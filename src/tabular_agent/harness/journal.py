"""run_journal.json — the resumable memory (spec §19).

Two layers, deliberately separated by *who has the information*:

* **Facts** — ``append_experiment`` is called by ``train_tabular_model`` as a
  side effect of actually training. The experiment record is a by-product of the
  experiment, not an agent chore, so it cannot be skipped or misremembered.
* **Interpretation** — ``append_note`` is the thin agent-written layer
  (hypothesis / why / next ideas). Not wired into a tool yet; the writer exists
  so the schema is stable from the start.

The journal *is* the run state. A fresh agent after a context reset reads this
file plus the task spec and knows what has been tried and what to beat.
"""

import json
import os

from ..metrics import greater_is_better

SCHEMA_VERSION = 1


def journal_path(run_dir: str) -> str:
    return os.path.join(run_dir, "run_journal.json")


def load(run_dir: str) -> dict:
    path = journal_path(run_dir)
    if not os.path.exists(path):
        return {
            "schema_version": SCHEMA_VERSION,
            "task": None,
            "baseline": None,
            "best": None,
            "tables": [],
            "leaderboard": [],
            "log": [],
            "stop": {"refusals": 0, "finalized": False},
        }
    with open(path) as f:
        return json.load(f)


def save(run_dir: str, journal: dict) -> str:
    """Write the journal atomically.

    ``open(path, "w")`` truncates before it writes, so a crash, OOM or kill
    between the two leaves an empty or half-written file — and the journal *is*
    the run state, so that loses the entire run, including trained artifacts that
    can no longer be attributed. Write a sibling temp file and ``os.replace`` it:
    the rename is atomic, so a reader sees either the old journal or the new one,
    never a partial one. ``fsync`` before the rename so the bytes are on disk and
    not just in the page cache when the machine goes down mid-run.
    """
    path = journal_path(run_dir)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(journal, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def set_task(run_dir: str, *, target: str, task_type: str, metric: str) -> dict:
    journal = load(run_dir)
    journal["task"] = {"target": target, "task_type": task_type, "metric": metric}
    save(run_dir, journal)
    return journal


def set_split(run_dir: str, record: dict) -> dict:
    """Record how the input was divided into work and validation halves.

    Written once and never updated. Two runs over the same file are only
    comparable if they validated against the same rows, and this is the only
    place that question can be answered after the fact — the halves themselves
    are inside a run directory that may be deleted or moved.
    """
    journal = load(run_dir)
    if journal.get("split") is None:
        journal["split"] = record
        save(run_dir, journal)
    return journal


def set_work_probe(run_dir: str, scored: dict) -> dict:
    """The ruler's score for the data as given, before any table was built.

    Written once. It is the reference every derived table is implicitly measured
    against, and it has no row in ``tables`` because no tool produced it — the
    work half is what the run started with.
    """
    journal = load(run_dir)
    if journal.get("work_probe") is None:
        journal["work_probe"] = scored
        save(run_dir, journal)
    return journal


def set_holdout(run_dir: str, record: dict) -> dict:
    """The sealed split's verdict, written once at finalize (0023).

    Deliberately **not** on the leaderboard and deliberately write-once. These
    scores exist to tell a reader afterwards how much the search gamed the
    validation half; the moment they can be consulted twice, or ranked by,
    they become another number to optimise against and the third split has
    bought nothing. Write-once is what makes "scored exactly once" auditable
    from the journal alone rather than a property of the calling code.
    """
    journal = load(run_dir)
    if journal.get("holdout") is None:
        journal["holdout"] = record
        save(run_dir, journal)
    return journal


def next_table_id(run_dir: str) -> str:
    """Claim the next free table id, atomically, by creating its file first.

    Read-then-write is what this used to be, and it collided. Two `fit_encoding`
    calls running at once each read ``len(tables)``, each computed the same `t7`,
    each wrote `tables/t7.parquet`, and the second silently replaced the first —
    four times in one Rossmann run, with the journal left claiming a column lives
    in a file that does not contain it.

    ``O_CREAT | O_EXCL`` makes the claim the same operation as the check: the
    filesystem refuses the second creator, so the loser takes the next id instead
    of sharing one. It holds across *processes* too, which no in-process lock and
    no framework setting can — and the framework setting we were relying on
    turned out to limit concurrent agent runs, not concurrent tool calls.

    The empty placeholder is harmless: every writer goes through
    :func:`io.save_parquet`, which replaces the path atomically.
    """
    tables_dir = os.path.join(run_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    n = len(load(run_dir).get("tables", [])) + 1
    while True:
        try:
            fd = os.open(
                os.path.join(tables_dir, f"t{n}.parquet"),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError:
            n += 1
            continue
        os.close(fd)
        return f"t{n}"


def append_table(run_dir: str, row: dict) -> dict:
    """Record a derived feature table (§19.2, same discipline as experiments).

    Written as a side effect of actually producing the table, so an agent
    resuming after a context reset can see which feature tables exist and how
    each was made without re-deriving it.
    """
    journal = load(run_dir)
    journal.setdefault("tables", []).append(row)
    save(run_dir, journal)
    return journal


def update_table(run_dir: str, path: str, fields: dict) -> dict:
    """Merge ``fields`` into an already-appended table row.

    For facts about a table that are only measurable after it exists — credit
    costs a fit per contributing table, so it is asked for, not produced on the
    way past. Keeping it here rather than in the leaderboard makes it survive
    a context reset: it is a property of the *table*, true regardless of which
    experiment happened to ask.

    The raw input table has no row (nothing derived it) and is silently ignored.
    """
    journal = load(run_dir)
    for row in journal.get("tables") or []:
        if row.get("path") == path:
            row.update(fields)
            save(run_dir, journal)
            break
    return journal


def next_experiment_id(run_dir: str) -> str:
    """Claim the next free experiment id by creating its directory, atomically.

    Same race as :func:`next_table_id` and a worse outcome: two concurrent
    training calls computing the same `e7` would write their fold models, OOF
    predictions and `meta.json` into one directory, so the artifacts backing a
    leaderboard row would belong to a different experiment than the row does —
    and `finalize_run` bundles from those artifacts.

    ``os.mkdir`` raises if the directory exists, which makes the claim and the
    check the same operation.
    """
    experiments_dir = os.path.join(run_dir, "experiments")
    os.makedirs(experiments_dir, exist_ok=True)
    n = len(load(run_dir)["leaderboard"]) + 1
    while True:
        try:
            os.mkdir(os.path.join(experiments_dir, f"e{n}"))
        except FileExistsError:
            n += 1
            continue
        return f"e{n}"


def append_experiment(run_dir: str, row: dict) -> dict:
    """Append one leaderboard row and re-derive ``baseline`` and ``best``.

    ``best`` is recomputed from the whole leaderboard rather than compared
    against the previous best, so it stays correct even if a row is edited or
    the journal is rebuilt.

    Ranked by ``val_score`` — the held-out half — not by ``cv_score``. Both are
    on the row, and they disagree: cross-validation is what the search optimises
    against, so it is the number the search can inflate. A row whose validation
    score is missing cannot be ranked and so cannot be ``best``; it stays on the
    leaderboard with its reason, because a missing score is a fact about the run.
    """
    journal = load(run_dir)
    journal["leaderboard"].append(row)

    metric = (journal.get("task") or {}).get("metric") or row.get("metric")
    if metric is not None:
        scored = [r for r in journal["leaderboard"] if r.get("val_score") is not None]
        baselines = [r for r in scored if r.get("model_type") == "baseline"]
        candidates = [r for r in scored if r.get("model_type") != "baseline"]

        pick = max if greater_is_better(metric) else min
        if baselines:
            b = pick(baselines, key=lambda r: r["val_score"])
            journal["baseline"] = {
                "experiment_id": b["id"],
                "val_score": b["val_score"],
                "cv_score": b.get("cv_score"),
            }
        # A run whose only experiment is the baseline still needs a `best` for
        # finalize_run to have something to package.
        pool = candidates or baselines
        if pool:
            top = pick(pool, key=lambda r: r["val_score"])
            journal["best"] = {
                "experiment_id": top["id"],
                "val_score": top["val_score"],
                "cv_score": top.get("cv_score"),
                "artifact_path": top["artifact_path"],
            }

    save(run_dir, journal)
    return journal


def update_experiment(run_dir: str, exp_id: str, fields: dict) -> dict:
    """Merge ``fields`` into an already-appended leaderboard row.

    Exists because the two numbers that say whether an *idea* worked —
    ``vs_source_table`` and the per-table credit — can only be computed once the
    row is in the journal, and they are worth keeping. Without them a resuming
    agent has
    every score but no attribution: it can see that e54 is the best, and not what
    made it better than e51.

    Never touches ``best``/``baseline``: those derive from ``val_score``, and this
    only ever writes fields alongside it.
    """
    journal = load(run_dir)
    for row in journal["leaderboard"]:
        if row.get("id") == exp_id:
            row.update(fields)
            break
    else:
        raise KeyError(f"no experiment {exp_id!r} in the leaderboard")
    save(run_dir, journal)
    return journal


def ancestry(journal: dict, table_path: str) -> list[str]:
    """``table_path`` and every derived table behind it, newest first.

    Stops at the raw input table, which has no row in ``tables`` because nothing
    derived it.
    """
    sources = {t["path"]: t.get("source_table") for t in journal.get("tables") or []}
    chain = []
    seen = set()
    current = table_path
    while current in sources and current not in seen:
        seen.add(current)
        chain.append(current)
        current = sources[current]
    return chain


def attribution(journal: dict, table_path: str) -> list[dict]:
    """Which links in a table's history were actually measured (spec §19.4).

    A feature table that was built and then extended without ever being scored
    leaves a gap: the score of the later table cannot be attributed to its own
    columns rather than its source's, because nothing ever measured the source.
    The run is still correct — the model that ships is the model that scored
    best — but the *explanation* has a hole in it, and the report should say so
    rather than let the winning table take credit for all of its columns.

    Since 0035 the ruler scores every table as it is written, so the gap this
    exists to report is normally empty and the numbers along a lineage are
    directly comparable: one estimator, one set of validation rows, one column
    per link. It stays because a table the ruler could not measure still leaves
    the same hole, and because "normally empty" is a property of the current
    design rather than a guarantee.

    Nothing here constrains the search. It only records what the run does and
    does not entitle you to conclude.
    """
    by_path = {t["path"]: t for t in journal.get("tables") or []}
    out = []
    for path in ancestry(journal, table_path):
        table = by_path.get(path, {})
        score = table.get("probe_score")
        out.append(
            {
                "table": path,
                "added": table.get("new_columns") or [],
                "method": table.get("method"),
                # Kept as a count so the report's wording still works; every
                # scored table has exactly the one measurement the ruler made.
                "n_experiments": 1 if score is not None else 0,
                "best_score": score,
                "measured": score is not None,
            }
        )
    return out


def columns_by_table(journal: dict, table_path: str) -> list[dict]:
    """The columns of ``table_path``, grouped by which table in its history added them.

    The grouping is the unit of *decision*, which is why credit is only ever
    reported this way: `fit_encoding` and `transform_features` each write exactly
    one table, so a table's ``new_columns`` is precisely one tool call's output.
    The agent cannot add half a tool call, so per-column numbers are finer than
    any move available to it, and inviting per-column conclusions is how an
    interaction partner gets dropped for scoring nothing on its own.

    That one-to-one is also why this needs no vocabulary of its own. An earlier
    version called each group a *block*, which named something that already had a
    name — and left the agent holding two words for one thing, next to
    ``source_table`` and ``parent`` for another.

    Oldest first. The raw input table has no row here, so its own columns are
    whatever no derived table claims; callers add it.
    """
    by_path = {t["path"]: t for t in journal.get("tables") or []}
    out = []
    for path in reversed(ancestry(journal, table_path)):
        row = by_path.get(path, {})
        out.append(
            {
                "table": path,
                "method": row.get("method"),
                "columns": list(row.get("new_columns") or []),
            }
        )
    return out


def stop_state(run_dir: str) -> dict:
    """Refusal bookkeeping for the stop rule (§19.3), tolerant of old journals."""
    return load(run_dir).get("stop") or {"refusals": 0, "finalized": False}


def record_refusal(run_dir: str) -> int:
    """Count a training call the harness turned down; return the new total."""
    journal = load(run_dir)
    state = journal.get("stop") or {"refusals": 0, "finalized": False}
    state["refusals"] = state.get("refusals", 0) + 1
    journal["stop"] = state
    save(run_dir, journal)
    return state["refusals"]


def record_table_refusal(run_dir: str, table: str) -> int:
    """Count a refused request for one exhausted table; return that table's total.

    Per table rather than per run: being redirected off `t1` and then asking for
    `t2` is the agent doing what it was told, and must not look like a loop.
    """
    journal = load(run_dir)
    state = journal.get("stop") or {"refusals": 0, "finalized": False}
    per_table = state.setdefault("table_refusals", {})
    per_table[table] = per_table.get(table, 0) + 1
    journal["stop"] = state
    save(run_dir, journal)
    return per_table[table]


def mark_finalized(run_dir: str) -> dict:
    journal = load(run_dir)
    state = journal.get("stop") or {"refusals": 0, "finalized": False}
    state["finalized"] = True
    journal["stop"] = state
    save(run_dir, journal)
    return journal


def append_note(run_dir: str, note: dict) -> dict:
    journal = load(run_dir)
    journal["log"].append(note)
    save(run_dir, journal)
    return journal
