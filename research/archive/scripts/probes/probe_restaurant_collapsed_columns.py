"""The controlled test v1-vs-v2 could not be: t7 against t8, one variable apart.

## Why this exists

Since 0054 this project has repeated a claim about Recruit Restaurant that it has
never measured: the precomputed `air_reserve_*` / `hpg_reserve_*` columns are cut
at the moment the forecast is issued, so a training row can count a table booked
the same morning while a forecast row cannot — and the column is therefore
*systematically larger in training than at serving*.

Restaurant v2 (0065) put that claim within reach and did not settle it. The agent
diagnosed the cut itself and built **t8**, which is **t7 with the six collapsed
reservation columns dropped and nothing else changed**. The probe scored t8 0.0085
*worse*, `best_table` is the argmax, and t7 shipped.

Then Kaggle scored v2 at private **0.55124** against v1's **0.54398**, and the
registered prediction — that v2 would decay less from sealed to private — was
refuted: v2 decayed **+0.04553** against v1's **+0.04222**.

**But v1 against v2 is not a controlled comparison.** Two different agent
trajectories, 17 tables against 13, different feature sets; only the harness
configuration was held still. That comparison cannot separate "the honest columns
did not help" from "v2's agent built a slightly worse model".

t7 against t8 can. Same chain, same rows, same split, one variable: whether the
six collapsed columns are present.

## The registered prediction

**t7 will beat t8 on the private board too**, and the "trap" this dataset was kept
for will turn out to be a null result on it.

The reasoning, stated so it can be held against the outcome: two external
refutations already exist. 0060 predicted Kaggle would punish the far edge more
and found the field decays *more* than we do (+0.0446 against +0.0310), so the
decay is a property of the task and not of these columns. v2 then decayed slightly
more than v1 while carrying the honest columns. A third reading in the same
direction would settle it.

**If t8 wins instead**, the mechanism is real, the probe is measuring the wrong
thing on exactly the rows that matter, and 0065's central claim survives its own
refutation — which would make the argmax-shipping defect it names urgent rather
than theoretical.

Either way this is a *reading of a mechanism*, not a search for a better score. It
is run once, and it spends the Restaurant private board for this question (0045).

## How it works, and why it is not a reimplementation

Each arm gets a fresh run directory holding a **copy of v2's own split** — the
same work/val/holdout parquets, byte for byte, so the two arms are scored on
identical rows — plus the one table it is about and that table's lineage.

Then `finalize_run` does the rest. It is the real shipping path: the enumerated
terminal set, the equal-weight blend (0044), the sealed score, the bundle. Nothing
here reimplements it, because a reimplementation that quietly differs is exactly
how a controlled experiment stops being one.

**The t7 arm is the control on the machinery itself.** It must reproduce v2's
journalled numbers — blend val 0.51971, sealed 0.50571 — or the harness is not
being driven the way v2 drove it and the t8 arm means nothing. That check runs
first and is fatal.
"""
import json
import os
import shutil
import sys

import pandas as pd

from tabular_agent.harness import journal
from tabular_agent.harness.session import RunSession
from tabular_agent.harness.stop import StopPolicy
from tabular_agent.harness.tools import finalize_run, inspect_data, probe_table
from tabular_agent.pipeline import load_bundle, predict_bundle
from tabular_agent.taskspec import TaskSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_restaurant import (  # noqa: E402
    ABOUT,
    COLUMN_NOTES,
    FORECAST_HORIZON,
    METRIC,
    OUT,
    SPLIT_ON,
    STREAMS,
    TEST_OUT,
    leaderboard,
    leaderboard_rank,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_RUN = os.path.join(REPO, "_runs", "restaurant_v2")
OUT_ROOT = os.path.join(REPO, "_runs", "restaurant_collapsed_probe")

#: v2's own numbers for the arm that shipped, read off its journal by the check
#: below rather than typed here — these are only what the check *expects to find*.
V2_SHIPPED = {"val": 0.519707, "sealed": 0.505713, "private": 0.55124}

ARMS = {
    "t7": "with the six collapsed reservation columns",
    "t8": "same chain, those columns dropped",
}


def build_arm(arm: str) -> RunSession:
    """A run directory holding v2's split and exactly one table."""
    run_dir = os.path.join(OUT_ROOT, arm)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(os.path.join(run_dir, "tables"))
    # The split, copied rather than recut. Recutting would give the same rows in
    # this case and there is no reason to depend on that: the point of the arm is
    # that both are scored on the rows v2 was scored on.
    shutil.copytree(os.path.join(SOURCE_RUN, "data"), os.path.join(run_dir, "data"))
    for suffix in (".parquet", ".pipeline.json"):
        shutil.copy(
            os.path.join(SOURCE_RUN, "tables", f"{arm}{suffix}"),
            os.path.join(run_dir, "tables", f"{arm}{suffix}"),
        )

    session = RunSession(
        run_dir, OUT,
        TaskSpec(target="visitors", metric=METRIC, notes=ABOUT,
                 column_notes=COLUMN_NOTES, forecast_horizon=FORECAST_HORIZON),
        stop_policy=StopPolicy(max_tables=25),
        val_fraction=0.05, holdout_fraction=0.05, split_on=SPLIT_ON,
        serving_columns=[c for c in pd.read_parquet(TEST_OUT).columns if c != "id"],
        fit_time_budget_s=20 * 60,
        streams=STREAMS,
    )
    inspect_data(session)

    # The one table on the board, scored by the run's own ruler so `best_table`
    # has something to be the argmax of. There is no choice for it to make here,
    # which is the point — the choice is the arm.
    table_path = os.path.join(run_dir, "tables", f"{arm}.parquet")
    scored = probe_table(session, table_path)
    journal.append_table(run_dir, {
        "id": arm,
        "path": f"tables/{arm}.parquet",
        "source_table": "data/work.parquet",
        "method": "copied from restaurant_v2",
        "new_columns": [],
        "n_steps": len(json.load(
            open(os.path.join(run_dir, "tables", f"{arm}.pipeline.json"))
        )["steps"]),
        **scored,
    })
    return session


def shipped_numbers(run_dir: str) -> dict:
    j = journal.load(run_dir)
    record = j.get("holdout") or {}
    blend = next(
        (e for e in record.get("scores", []) if e.get("model_type", "").startswith("blend")),
        None,
    )
    return {
        "probe": (j.get("tables") or [{}])[0].get("probe_score"),
        "blend_val": (blend or {}).get("val_score"),
        "blend_sealed": (blend or {}).get("holdout_score"),
        "all": [
            {"model": e.get("model_type"), "val": e.get("val_score"),
             "sealed": e.get("holdout_score")}
            for e in record.get("scores", [])
        ],
    }


def write_submission(run_dir: str, arm: str) -> str:
    bundle = load_bundle(os.path.join(run_dir, "bundle"))
    test = pd.read_parquet(TEST_OUT)
    streams = {name: pd.read_parquet(spec["path"]) for name, spec in STREAMS.items()}
    preds = predict_bundle(bundle, test.drop(columns=["id"]), streams=streams)
    path = os.path.join(OUT_ROOT, f"submission_{arm}.csv")
    pd.DataFrame({"id": test["id"], "visitors": preds.clip(min=0)}).to_csv(
        path, index=False
    )
    return path


def main() -> int:
    if not os.path.isdir(SOURCE_RUN):
        print(f"missing {SOURCE_RUN} — run scripts/run_restaurant_v2.py first",
              file=sys.stderr)
        return 1
    os.makedirs(OUT_ROOT, exist_ok=True)

    results = {}
    for arm, what in ARMS.items():
        print(f"\n{'=' * 70}\n{arm}: {what}")
        session = build_arm(arm)
        finalize_run(session)
        results[arm] = shipped_numbers(session.run_dir)
        for row in results[arm]["all"]:
            sealed = row["sealed"]
            print(f"  {row['model']:<40} val {row['val']:.5f}  sealed "
                  f"{'n/a' if sealed is None else f'{sealed:.5f}'}")
        path = write_submission(session.run_dir, arm)
        sub = pd.read_csv(path)
        print(f"  submission: {path}  ({len(sub)} rows, mean "
              f"{sub.visitors.mean():.2f})")

    # The control on the machinery. If the t7 arm does not reproduce what v2
    # recorded, the two arms were not driven the way v2 was driven and nothing
    # below is a controlled comparison.
    print(f"\n{'=' * 70}\nCONTROL — does the t7 arm reproduce v2's shipped numbers?")
    got = results["t7"]
    for key, want in (("blend_val", V2_SHIPPED["val"]), ("blend_sealed", V2_SHIPPED["sealed"])):
        ok = got[key] is not None and abs(got[key] - want) < 1e-5
        print(f"  {key:<14} v2 {want:.6f}  rebuilt "
              f"{'n/a' if got[key] is None else f'{got[key]:.6f}'}  "
              f"{'OK' if ok else '** MISMATCH **'}")
        if not ok:
            print("\n  The rebuild does not reproduce the run it copies. The t8 arm "
                  "is not a controlled comparison and its submission should not be "
                  "scored.")
            return 1

    print(f"\n{'=' * 70}\nTHE COMPARISON — one variable, six columns")
    print(f"  {'':6} {'probe':>10} {'blend val':>10} {'blend sealed':>13}")
    for arm in ARMS:
        r = results[arm]
        print(f"  {arm:<6} {r['probe']:>10.5f} {r['blend_val']:>10.5f} "
              f"{r['blend_sealed']:>13.5f}")
    d_probe = results["t8"]["probe"] - results["t7"]["probe"]
    d_sealed = results["t8"]["blend_sealed"] - results["t7"]["blend_sealed"]
    print(f"\n  t8 - t7:  probe {d_probe:+.5f}   sealed {d_sealed:+.5f}   "
          "(rmsle, lower is better, so positive = t8 worse)")
    print("\n  Neither number can settle it: both are measured on rows drawn from")
    print("  the same uncut stream as training (0065). The private board can.")
    print(f"\n  Submit both CSVs. v2's t7 blend already scored private "
          f"{V2_SHIPPED['private']} ({leaderboard_rank(V2_SHIPPED['private'])}),")
    print(f"  so the t7 arm here is also a check on that. private board: "
          f"{leaderboard(True)}")
    with open(os.path.join(OUT_ROOT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
