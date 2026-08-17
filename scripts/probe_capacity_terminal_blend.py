"""Capacity on the thing that actually ships: the terminal blend, not one probe fit.

0067 measured capacity on Rossmann's `t13` — the chain that shipped — and found
**+0.0068**, a tenth of what the same knob is worth on a bare table. That number
decides where this project spends its next month, and it was produced by **one
lgbm scored on the validation half**.

That is not what ships. What ships is `finalize_run`'s output: four families fitted
on the winning table, averaged equal-weight (0044), scored on the sealed holdout.
And a fact checked while writing 0067 and not acted on:

    models.py:419-432   lgbm / xgb / catboost defaults: n_estimators=200
    probe.py:110        PROBE_PARAMS = {"n_estimators": 200}

**Every capacity number in this repo is a probe reading on a single lgbm.** The
terminal fits — the models that are actually served — have never been run at a
real budget on any dataset, ever. This script is the first reading of that cell.

## Why the blend might not inherit the gain

An equal-weight average of four models already buys some of what capacity buys:
variance reduction. Three under-trained boosted trees with different inductive
biases, averaged, land somewhere a single under-trained tree does not. If that
overlap is large, raising every member's budget adds less to the blend than it
added to the lone lgbm — and the +0.0068 headline shrinks on the way to the
serving path.

It could also go the other way. Averaging cannot manufacture structure none of the
members found; if 200 trees is too few to reach 1,115 store effects, all four
members miss them and the average misses them too.

Nothing in this repo distinguishes those two stories, and they imply different
priorities.

## The prediction, registered before running

* **The blend gains less than the single lgbm's +0.0068 on Rossmann** — the
  substitution argument above. Point estimate **+0.003 to +0.006** sealed.
* **Restaurant's `t7` gains more than Rossmann's `t13` does**, because 0054's
  mechanism is per-entity identity and restaurant has 829 shop effects behind a
  *string* id that `_feature_frame` drops — so the members have further to reach.
  Held loosely: t7 is a well-featured table and 0067's substitution finding cuts
  the other way.
* **The sealed gain is smaller than the validation gain on both**, because a bigger
  model fits the validation half's period better than the next period (0060: a
  sealed score is a lower bound at the horizon you will be graded at).

**The falsifier, and it closes the line: a sealed gain below +0.002 on both
datasets.** That would mean capacity is a property of the *probe* and not of the
harness's output, that 0067's +0.0068 does not reach the serving path at all, and
that the whole capacity thread has been measuring an instrument rather than a
product.

## What is varied, and what is deliberately not

One axis (0017): `n_estimators` and `learning_rate`, applied **identically to all
three boosted families**. That is 0039's "common budget" — the record that first
named this and left it open, on the grounds that enumerating library defaults
measures the libraries rather than the families. `elasticnet` has no budget and is
untouched, so it is a passive control: its score must not move between arms.

The two budgets are 0067's own grid points, not new ones. `1000 @ lr 0.10` and
`5000 @ lr 0.03` were the two best on `t13` (+0.0057 and +0.0068) at a fivefold
difference in cost, and reusing them is what makes this reading comparable to the
single-lgbm one it is meant to be read against. **Neither is a setting to adopt** —
0054's "do not raise the constant" stands, and where a budget comes from is the
open decision (0054 item 2), not something this script may answer.

## How it is driven

Each arm gets a fresh run directory holding a **copy of the source run's split**,
byte for byte, plus the one shipped table and its lineage — so every arm is scored
on the rows the original run was scored on, and a recut split cannot masquerade as
a capacity effect. Then `finalize_run` does the rest: the enumerated terminal set,
the blend, the sealed score. Nothing here reimplements it.

**The `current` arm is the control on the machinery.** It must reproduce the source
run's own journalled val and sealed scores, per model, or the harness is not being
driven the way the run drove it and the budget arms mean nothing. Fatal if it
fails. One deviation is known and stated: `fit_time_budget_s` is raised so a
5,000-tree fit is not abandoned mid-way. At 200 trees it never bound, so the
control is unaffected by it.

The single-lgbm probe sweep is run alongside, from `probe_capacity.sweep`, imported
rather than restated. On Rossmann it reproduces `_runs/capacity_v7_chain.csv` and
is therefore a second control (0055: a rename made every earlier artifact
unreplayable, and nothing noticed for four hours). On Restaurant it is new — no
capacity reading on `t7` exists.

**No submissions.** Rossmann's board is spent (0045) and Restaurant's is spent for
this question (0066). Both readings below are internal.
"""
import contextlib
import json
import os
import shutil
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from tabular_agent import models  # noqa: E402
from tabular_agent.harness import journal  # noqa: E402
from tabular_agent.harness.session import RunSession  # noqa: E402
from tabular_agent.harness.stop import StopPolicy  # noqa: E402
from tabular_agent.harness.tools import finalize_run, inspect_data, probe_table  # noqa: E402
from tabular_agent.metrics import greater_is_better  # noqa: E402
from tabular_agent.taskspec import TaskSpec  # noqa: E402

from probe_capacity import sweep  # noqa: E402

RUNS = os.path.join(REPO, "_runs")
OUT_ROOT = os.path.join(RUNS, "capacity_terminal_blend")

#: The arms. `current` is not a candidate — it is the harness as it stands, and
#: every other row is read against it. See the docstring for why these two and not
#: a fresh grid.
BUDGETS: list[tuple[str, dict | None]] = [
    ("current  200 @ lr 0.10", None),
    ("        1000 @ lr 0.10", {"n_estimators": 1000, "learning_rate": 0.1}),
    ("        5000 @ lr 0.03", {"n_estimators": 5000, "learning_rate": 0.03}),
]

#: The families a budget means anything to. `elasticnet` is excluded because it has
#: no boosting rounds — which makes it the passive control: if its score moves
#: between arms, something other than capacity moved.
BUDGETED = ("lgbm", "xgb", "catboost")

#: Raised from the source runs' 20 minutes so a 5,000-tree fit is not abandoned
#: by `TrainingTooSlow`. At 200 trees the terminal fits took 12-18s (Rossmann v7)
#: and 4-8s (Restaurant v2), so this cannot affect the control arm.
FIT_TIME_BUDGET_S = 60 * 60


@contextlib.contextmanager
def _swap_budget(params: dict | None):
    """Run the harness's own terminal fits at a different budget, then put it back.

    Wraps `models._make_model` rather than editing the defaults in place, and
    layers the budget **on top of** whatever params the caller passed. That last
    detail is load-bearing: `_terminal_fits` hands the lgbm candidate
    `dict(PROBE_PARAMS)`, which pins `n_estimators=200`, so a patch that merely
    changed the *default* would leave the blend's lgbm member — the probe's own
    fit, and the one that makes 0044's invariant hold — at 200 trees while the
    other two moved. That arm would look like a blend at budget and would not be
    one.

    `models.fit_model` resolves the global at call time, so rebinding the module
    attribute is enough. `probe._make_model` is deliberately *not* patched: the
    probe scores the same single table in every arm, and holding it fixed keeps
    the ruler that picks `best_table` identical across arms.
    """
    if not params:
        yield
        return
    original = models._make_model

    def patched(model_type, task_type, p, seed, metric=None):
        if model_type in BUDGETED:
            p = dict(p or {})
            p.update(params)
        return original(model_type, task_type, p, seed, metric)

    models._make_model = patched
    try:
        yield
    finally:
        models._make_model = original


# --------------------------------------------------------------------------
# The two datasets, each configured exactly as the run that produced its table
# configured it. Deviating here would entangle the capacity reading with a
# configuration change, which is the one thing this script exists not to do.
# --------------------------------------------------------------------------

def _rossmann(run_dir: str) -> tuple[RunSession, dict]:
    from prep_rossmann import (
        ABOUT, COLUMN_NOTES, METRIC, OUT, SPLIT_ON, TEST_OUT,
    )

    session = RunSession(
        run_dir, OUT,
        TaskSpec(target="Sales", metric=METRIC, notes=ABOUT,
                 column_notes=COLUMN_NOTES, forecast_horizon=1.0),
        stop_policy=StopPolicy(max_tables=25),
        val_fraction=0.05, holdout_fraction=0.05, split_on=SPLIT_ON,
        serving_columns=[c for c in pd.read_parquet(TEST_OUT).columns if c != "Id"],
        fit_time_budget_s=FIT_TIME_BUDGET_S,
    )
    return session, {}


def _restaurant(run_dir: str) -> tuple[RunSession, dict]:
    from prep_restaurant import (
        ABOUT, COLUMN_NOTES, FORECAST_HORIZON, METRIC, OUT, SPLIT_ON, STREAMS, TEST_OUT,
    )

    session = RunSession(
        run_dir, OUT,
        TaskSpec(target="visitors", metric=METRIC, notes=ABOUT,
                 column_notes=COLUMN_NOTES, forecast_horizon=FORECAST_HORIZON),
        stop_policy=StopPolicy(max_tables=25),
        val_fraction=0.05, holdout_fraction=0.05, split_on=SPLIT_ON,
        serving_columns=[c for c in pd.read_parquet(TEST_OUT).columns if c != "id"],
        fit_time_budget_s=FIT_TIME_BUDGET_S,
        streams=STREAMS,
    )
    return session, {"streams": STREAMS}


DATASETS = {
    # name: (builder, source run, the table that shipped, the ordering column)
    "rossmann": (_rossmann, "rossmann_v7", "t13", "Date"),
    "restaurant": (_restaurant, "restaurant_v2", "t7", "visit_date"),
}


def control_numbers(source_run: str) -> dict[str, dict]:
    """What the source run recorded per model, read off its journal, not typed.

    0061's leaderboard constants were wrong for weeks because they were typed
    beside the file that contradicted them. The control has to come from the
    artifact it is checking against.
    """
    j = journal.load(os.path.join(RUNS, source_run))
    out = {}
    for e in (j.get("holdout") or {}).get("scores", []):
        mt = e.get("model_type", "")
        if mt and not mt.startswith("blend"):
            out[mt] = {"val": e.get("val_score"), "sealed": e.get("holdout_score")}
    return out


def build_arm(name: str, arm_dir: str) -> RunSession:
    """A run directory holding the source run's split and exactly one table."""
    builder, source_run, table, _ = DATASETS[name]
    source = os.path.join(RUNS, source_run)
    if os.path.exists(arm_dir):
        shutil.rmtree(arm_dir)
    os.makedirs(os.path.join(arm_dir, "tables"))
    # Copied, not recut. The rows of the shipped table are the source run's work
    # half; a recut split would score a table built on different rows, which is
    # the one way a capacity sweep can silently stop being a capacity sweep.
    shutil.copytree(os.path.join(source, "data"), os.path.join(arm_dir, "data"))
    for suffix in (".parquet", ".pipeline.json"):
        shutil.copy(
            os.path.join(source, "tables", f"{table}{suffix}"),
            os.path.join(arm_dir, "tables", f"{table}{suffix}"),
        )

    session, _ = builder(arm_dir)
    inspect_data(session)

    # The one table on the board, scored by the run's own ruler so `best_table`
    # has something to be the argmax of. There is no choice for it to make, which
    # is the point — the choice here is the budget.
    table_path = os.path.join(arm_dir, "tables", f"{table}.parquet")
    scored = probe_table(session, table_path)
    with open(os.path.join(arm_dir, "tables", f"{table}.pipeline.json")) as f:
        n_steps = len(json.load(f)["steps"])
    journal.append_table(arm_dir, {
        "id": table, "path": f"tables/{table}.parquet",
        "source_table": "data/work.parquet",
        "method": f"copied from {source_run}",
        "new_columns": [], "n_steps": n_steps, **scored,
    })
    return session


def shipped_numbers(run_dir: str) -> dict:
    """Every terminal candidate and the blend, val and sealed, from the journal."""
    j = journal.load(run_dir)
    rows = (j.get("holdout") or {}).get("scores", [])
    return {
        e.get("model_type"): {"val": e.get("val_score"), "sealed": e.get("holdout_score")}
        for e in rows
    }


def blend_key(scores: dict) -> str | None:
    return next((k for k in scores if k and k.startswith("blend")), None)


def run_dataset(name: str) -> list[dict]:
    builder, source_run, table, order_col = DATASETS[name]
    source = os.path.join(RUNS, source_run)
    if not os.path.isdir(source):
        print(f"  missing {source} — skipping {name}", file=sys.stderr)
        return []

    print(f"\n{'=' * 74}\n=== {name}: {source_run}'s {table}, the table that shipped ===")

    # ---- Reading 1: the single lgbm, the way every capacity number so far was
    # made. On Rossmann this reproduces `_runs/capacity_v7_chain.csv`; on
    # Restaurant no such reading exists.
    probe_dir = os.path.join(OUT_ROOT, name, "probe")
    session = build_arm(name, probe_dir)
    print(f"\n  -- the single-lgbm probe, for the comparison the blend is read against")
    probe_rows = sweep(name, session, os.path.join(probe_dir, "tables", f"{table}.parquet"),
                       order_col, f"{source_run} {table}")

    # ---- Reading 2: the terminal blend, which is what ships.
    metric = session.resolved().metric
    higher = greater_is_better(metric)
    control = control_numbers(source_run)
    results: dict[str, dict] = {}
    rows: list[dict] = []

    for label, params in BUDGETS:
        print(f"\n  -- terminal fits @ {label.strip()}")
        arm_dir = os.path.join(OUT_ROOT, name, label.strip().replace(" ", "_"))
        session = build_arm(name, arm_dir)
        with _swap_budget(params):
            finalize_run(session)
        scores = shipped_numbers(arm_dir)
        results[label] = scores
        for mt, s in scores.items():
            sealed = "n/a" if s["sealed"] is None else f"{s['sealed']:.5f}"
            print(f"     {mt:<42} val {s['val']:.5f}  sealed {sealed}")

    # ---- The control on the machinery, before anything is read.
    print(f"\n  {'-' * 70}\n  CONTROL — does the 200-tree arm reproduce {source_run}?")
    got = results[BUDGETS[0][0]]
    ok = True
    for mt, want in control.items():
        have = got.get(mt)
        if have is None:
            print(f"    {mt:<12} ** missing from the rebuild **")
            ok = False
            continue
        for field in ("val", "sealed"):
            a, b = want[field], have[field]
            match = a is not None and b is not None and abs(a - b) < 1e-5
            print(f"    {mt:<12} {field:<7} {source_run} {a:.6f}  rebuilt {b:.6f}  "
                  f"{'OK' if match else '** MISMATCH **'}")
            ok = ok and match
    if not ok:
        print("\n    The rebuild does not reproduce the run it copies. The budget arms\n"
              "    are not a controlled comparison and nothing below should be read.")
        return []

    # ---- The reading.
    print(f"\n  {'-' * 70}\n  THE READING ({metric}, "
          f"{'higher' if higher else 'lower'} is better)")
    base = results[BUDGETS[0][0]]
    base_blend = blend_key(base)
    header = f"    {'budget':<24} {'blend val':>10} {'gain':>9} {'blend sealed':>13} {'gain':>9}"
    print(header)
    for label, _ in BUDGETS:
        scores = results[label]
        key = blend_key(scores)
        if key is None or base_blend is None:
            print(f"    {label:<24} no blend was formed")
            continue
        line = f"    {label:<24}"
        for field in ("val", "sealed"):
            now, was = scores[key][field], base[base_blend][field]
            if now is None or was is None:
                line += f" {'n/a':>10} {'':>9}" if field == "val" else f" {'n/a':>13} {'':>9}"
                continue
            gain = (now - was) if higher else (was - now)
            width = 10 if field == "val" else 13
            line += f" {now:>{width}.5f} {gain:>+9.5f}"
        print(line)
        rows.append({
            "dataset": name, "source_run": source_run, "table": table,
            "budget": label.strip(), "arm": "blend",
            "val": scores[key]["val"], "sealed": scores[key]["sealed"],
            "val_gain": None if scores[key]["val"] is None else
                        ((scores[key]["val"] - base[base_blend]["val"]) if higher
                         else (base[base_blend]["val"] - scores[key]["val"])),
            "sealed_gain": None if scores[key]["sealed"] is None else
                           ((scores[key]["sealed"] - base[base_blend]["sealed"]) if higher
                            else (base[base_blend]["sealed"] - scores[key]["sealed"])),
        })

    # The passive control: elasticnet has no budget, so its score must not move.
    moved = [
        label for label, _ in BUDGETS[1:]
        if results[label].get("elasticnet", {}).get("val") is not None
        and base.get("elasticnet", {}).get("val") is not None
        and abs(results[label]["elasticnet"]["val"] - base["elasticnet"]["val"]) > 1e-9
    ]
    print(f"\n    elasticnet (no budget, passive control): "
          f"{'unchanged in every arm — OK' if not moved else f'** MOVED in {moved} **'}")

    # Per-member, so a blend that gains less than its members is visible as such
    # rather than inferred.
    print(f"\n    per-member sealed, so the averaging is not hidden:")
    print(f"    {'model':<42}" + "".join(f"{label.strip():>18}" for label, _ in BUDGETS))
    for mt in base:
        if mt is None:
            continue
        cells = ""
        for label, _ in BUDGETS:
            s = results[label].get(mt, {}).get("sealed")
            cells += f"{'n/a':>18}" if s is None else f"{s:>18.5f}"
        print(f"    {mt:<42}{cells}")
        rows.append({
            "dataset": name, "source_run": source_run, "table": table,
            "budget": "per-member", "arm": mt,
            "val": None, "sealed": None, "val_gain": None, "sealed_gain": None,
        })

    for r in probe_rows:
        rows.append({
            "dataset": name, "source_run": source_run, "table": table,
            "budget": r["budget"].strip(), "arm": "probe lgbm",
            "val": r["score"], "sealed": None,
            "val_gain": r["gain_vs_current"], "sealed_gain": None,
        })

    if probe_rows:
        best_probe = max(r["gain_vs_current"] for r in probe_rows)
        best_blend = max((r["sealed_gain"] or 0.0) for r in rows if r["arm"] == "blend")
        print(f"\n    single lgbm, best probe gain    {best_probe:+.5f}")
        print(f"    blend, best sealed gain         {best_blend:+.5f}")
        if best_blend < 0.002:
            print("\n    FALSIFIED on this dataset: capacity does not reach the serving\n"
                  "    path. The gain is a property of the probe, not of what ships.")
        elif best_blend < best_probe:
            print("\n    The blend inherits part of the gain, not all of it — an\n"
                  "    equal-weight average was already buying some of what capacity buys.")
        else:
            print("\n    The blend gains at least as much as the single fit: averaging was\n"
                  "    not substituting for capacity, and the prediction was wrong.")
    return rows


def main() -> int:
    wanted = sys.argv[1:] or list(DATASETS)
    unknown = [w for w in wanted if w not in DATASETS]
    if unknown:
        print(f"unknown dataset(s): {unknown}; known: {list(DATASETS)}", file=sys.stderr)
        return 1
    os.makedirs(OUT_ROOT, exist_ok=True)

    all_rows: list[dict] = []
    for name in wanted:
        all_rows += run_dataset(name)

    if all_rows:
        out = os.path.join(RUNS, "capacity_terminal_blend.csv")
        fresh = pd.DataFrame(all_rows)
        if os.path.exists(out):
            old = pd.read_csv(out)
            old = old[~old.dataset.isin(set(fresh.dataset))]
            fresh = pd.concat([old, fresh], ignore_index=True)
        fresh.to_csv(out, index=False)
        print(f"\nwrote {out}  ({len(fresh)} rows)")

    print("\nThese are internal scores — validation and the sealed holdout. No\n"
          "submission is written: Rossmann's board is spent (0045) and Restaurant's\n"
          "is spent for this question (0066). 0054's 'do not raise the constant'\n"
          "stands; where a budget comes from is still the open decision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
