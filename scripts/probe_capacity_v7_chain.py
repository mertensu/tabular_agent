"""The untested cell: Rossmann v7's own chain at a real training budget.

0061 named this and nothing has measured it:

> The trap in reading that optimistically: capacity *was* already spent, on a
> bare feature block, and that arm sits at top 69% — behind v7's chain at 200
> trees (top 58%). **The untested cell is still v7's chain at a real budget.**

## Why this is the cell that matters

0054 measured capacity on two Rossmann tables and neither is the one that
matters:

| table | 200 trees | best budget | gain |
|---|---|---|---|
| baseline (raw columns, `Customers` dropped) | 0.21810 | 0.15749 | **+0.0606** |
| v8's engineered feature *block* (code only) | 0.23331 | 0.15213 | **+0.0812** |
| **v7's t13 — the chain that actually shipped** | **0.13202** | ? | **?** |

The first two start from tables that score 0.22–0.23. v7's chain already scores
**0.13202** at 200 trees, better than either reaches *with* capacity. So the two
existing readings cannot answer the only question that decides where this project
should spend its next month:

    do capacity and feature work ADD UP, or is the feature work doing
    capacity's job by hand?

If they add up, v7's chain at a real budget goes somewhere no run of this project
has been, and the harness's fixed 200 trees is the whole story of being mid-table.
If they substitute, the chain has saturated, capacity buys little on top of it,
and the gap to the median is made of something we have not identified yet.

v8's t24 could not be swept in 0054 because 0051's rename broke its lineage
replay. **v7's t13 predates `fit_history` entirely** — its eleven steps are
transforms, a row filter and target/frequency encoders — so it replays at HEAD,
which is why this measurement is possible now and was not then.

## The prediction, registered before running

* **Mostly substitutes, with a real but modest gain: +0.010 to +0.030 rmspe.**
  The reasoning: both mechanisms are reaching the same structure. v7's chain
  encodes per-store, per-store-weekday and per-store-promo effects *explicitly*;
  capacity lets a tree ensemble discover them by splitting. A table that already
  carries them should have less left for capacity to find than a bare one, so the
  gain should be well below the bare table's +0.0606.
* **Even the low end of that range matters.** The gap from v7's shipped blend to
  the private median is **+0.0085** (0061). A gain of +0.010 on the probe would be
  the largest single mechanism this project has available, and it comes from one
  parameter rather than from more vocabulary.
* **Early stopping lands near the best fixed budget**, as it did on all four
  datasets in 0054.

**The falsifier: a gain below +0.005 at every budget.** That would mean a
well-featured table has saturated, that 0054's large numbers are a property of
*impoverished* tables only, and that capacity is not the answer to "why are we
mid-table". This line would then be closed and the priority list rewritten.

## What this is not

Not a tuning run and not a setting to adopt. It measures whether a knob matters on
the one table where it has never been tried. 0054's rule stands: **do not raise
the constant** — early stopping asked for 162 to 16,926 rounds across four
datasets, and picking Rossmann's is 0022 one level up. Where the budget should
come from is a design decision that needs its own record.
"""
import os
import shutil
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from prep_rossmann import (  # noqa: E402
    ABOUT,
    COLUMN_NOTES,
    METRIC,
    OUT,
    SPLIT_ON,
    TEST_OUT,
    leaderboard,
)
# Imported rather than restated, so the grid, the monkeypatch and the
# early-stopping arm are byte-identical to the ones that produced
# `_runs/capacity_probe.csv`. A sweep that redefined any of them would produce
# numbers that look comparable and are not.
from probe_capacity import BUDGETS, sweep  # noqa: E402
from tabular_agent.harness.session import RunSession  # noqa: E402
from tabular_agent.io import load_table  # noqa: E402
from tabular_agent.taskspec import TaskSpec  # noqa: E402

RUNS = os.path.join(REPO, "_runs")
SOURCE_RUN = os.path.join(RUNS, "rossmann_v7")
RUN_DIR = os.path.join(RUNS, "capacity_rossmann_v7")

#: v7's best table, and the one experiment e3 — the run that shipped — trained on.
TABLE = "t13"

#: What 0054 already measured, for the comparison the result has to be read
#: against. Read off `_runs/capacity_probe.csv` at run time rather than typed, so
#: this cannot drift from the file the way 0061's leaderboard constants did.
REFERENCE_TABLES = ("baseline table", "v8 feature block")


def build_session() -> tuple[RunSession, str]:
    """v7's split and v7's best table, in a run directory of their own.

    The split is **copied, not recut**. The rows of `t13` are v7's work half, so
    a recut split would score a table built on different rows — which is the one
    way a capacity sweep can silently stop being a capacity sweep.
    """
    if os.path.exists(RUN_DIR):
        shutil.rmtree(RUN_DIR)
    os.makedirs(os.path.join(RUN_DIR, "tables"))
    shutil.copytree(os.path.join(SOURCE_RUN, "data"), os.path.join(RUN_DIR, "data"))
    for suffix in (".parquet", ".pipeline.json"):
        shutil.copy(
            os.path.join(SOURCE_RUN, "tables", f"{TABLE}{suffix}"),
            os.path.join(RUN_DIR, "tables", f"{TABLE}{suffix}"),
        )

    serving = [c for c in pd.read_parquet(TEST_OUT).columns if c != "Id"]
    session = RunSession(
        RUN_DIR, OUT,
        TaskSpec(target="Sales", metric=METRIC, notes=ABOUT,
                 column_notes=COLUMN_NOTES, forecast_horizon=1.0),
        val_fraction=0.05, holdout_fraction=0.05, split_on=SPLIT_ON,
        serving_columns=serving,
    )
    return session, os.path.join(RUN_DIR, "tables", f"{TABLE}.parquet")


def reference_rows() -> pd.DataFrame:
    path = os.path.join(RUNS, "capacity_probe.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df[(df.dataset == "rossmann") & (df.table.isin(REFERENCE_TABLES))]


def main() -> int:
    if not os.path.isdir(SOURCE_RUN):
        print(f"missing {SOURCE_RUN}", file=sys.stderr)
        return 1

    session, table_path = build_session()
    work = load_table(session.data_path)
    table = load_table(table_path)
    print(f"=== rossmann v7, {TABLE} — the chain that shipped ===")
    print(f"  work {len(work):,} rows | val {len(load_table(session.val_path)):,} rows")
    print(f"  table {len(table):,} rows x {table.shape[1]} cols "
          f"(fewer rows than work: v7's chain filters Open == 1)")

    rows = sweep("rossmann", session, table_path, "Date", f"v7 chain ({TABLE})")
    if not rows:
        print("\nno budget could be scored — the lineage did not replay at HEAD.")
        return 1

    print(f"\n{'=' * 70}\nAGAINST WHAT 0054 ALREADY MEASURED")
    ref = reference_rows()
    if not ref.empty:
        for label in REFERENCE_TABLES:
            sub = ref[ref.table == label]
            if sub.empty:
                continue
            best = sub.loc[sub.score.idxmin()]
            first = sub[sub.budget.str.contains("current")].iloc[0]
            print(f"  {label:<22} 200 trees {first.score:.5f} -> best "
                  f"{best.score:.5f}  ({best.gain_vs_current:+.5f}, "
                  f"{best.budget.strip()})")
    got = pd.DataFrame(rows)
    best = got.loc[got.score.idxmin()]
    first = got.iloc[0]
    print(f"  {'v7 chain (' + TABLE + ')':<22} 200 trees {first.score:.5f} -> best "
          f"{best.score:.5f}  ({best.gain_vs_current:+.5f}, {best.budget.strip()})")

    print(f"\n{'=' * 70}\nTHE READING")
    gain = float(best.gain_vs_current)
    if gain < 0.005:
        print(f"  FALSIFIED at {gain:+.5f}: a well-featured table has saturated, and")
        print("  0054's large numbers are a property of impoverished tables. Capacity")
        print("  is not the answer to being mid-table; the priority list needs rewriting.")
    else:
        print(f"  Capacity is worth {gain:+.5f} on the chain that actually shipped.")
        print("  v7's shipped blend sits +0.00846 from the private median (0061), so")
        print(f"  this is {gain / 0.00846:.1f}x that gap — from one parameter.")
    print("\n  This is a probe score on v7's validation half, NOT a Kaggle score.")
    print("  It says the knob matters, not what the harness should do about it —")
    print("  0054's 'do not raise the constant' stands, and where an early-stopping")
    print("  block comes from is still the open decision.")
    print(f"\n  private board, for scale: {leaderboard(True)}")

    out = os.path.join(RUNS, "capacity_v7_chain.csv")
    got.to_csv(out, index=False)
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
