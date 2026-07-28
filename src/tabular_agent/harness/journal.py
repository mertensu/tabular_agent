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
            "leaderboard": [],
            "log": [],
        }
    with open(path) as f:
        return json.load(f)


def save(run_dir: str, journal: dict) -> str:
    path = journal_path(run_dir)
    with open(path, "w") as f:
        json.dump(journal, f, indent=2)
    return path


def set_task(run_dir: str, *, target: str, task_type: str, metric: str) -> dict:
    journal = load(run_dir)
    journal["task"] = {"target": target, "task_type": task_type, "metric": metric}
    save(run_dir, journal)
    return journal


def next_experiment_id(run_dir: str) -> str:
    return f"e{len(load(run_dir)['leaderboard']) + 1}"


def append_experiment(run_dir: str, row: dict) -> dict:
    """Append one leaderboard row and re-derive ``baseline`` and ``best``.

    ``best`` is recomputed from the whole leaderboard rather than compared
    against the previous best, so it stays correct even if a row is edited or
    the journal is rebuilt.
    """
    journal = load(run_dir)
    journal["leaderboard"].append(row)

    metric = (journal.get("task") or {}).get("metric") or row.get("metric")
    if metric is not None:
        scored = [r for r in journal["leaderboard"] if r.get("cv_score") is not None]
        baselines = [r for r in scored if r.get("model_type") == "baseline"]
        candidates = [r for r in scored if r.get("model_type") != "baseline"]

        pick = max if greater_is_better(metric) else min
        if baselines:
            b = pick(baselines, key=lambda r: r["cv_score"])
            journal["baseline"] = {"experiment_id": b["id"], "cv_score": b["cv_score"]}
        # A run whose only experiment is the baseline still needs a `best` for
        # finalize_run to have something to package.
        pool = candidates or baselines
        if pool:
            top = pick(pool, key=lambda r: r["cv_score"])
            journal["best"] = {
                "experiment_id": top["id"],
                "cv_score": top["cv_score"],
                "artifact_path": top["artifact_path"],
            }

    save(run_dir, journal)
    return journal


def append_note(run_dir: str, note: dict) -> dict:
    journal = load(run_dir)
    journal["log"].append(note)
    save(run_dir, journal)
    return journal
