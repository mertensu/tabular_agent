"""Does searching over feature *subsets* buy anything, on the tables that shipped?

The Rossmann winner's report names feature selection as the device his solution
turned on, and this project has no version of it:

> *"With all features together, the model easily overfit the train set, resulting
> in suboptimal performance on the test set. Therefore, I needed a way to reduce
> the feature set... The handpicking cost me a lot of time and I realized that it
> was biased by my ideas. Therefore I decided to create some models on random
> selections of the features."*

Every tool in this harness makes a table **wider**. `select_features` is the first
that can make one narrower, and this script is its first reading against real
tables rather than tests.

## Why this needs measuring rather than assuming

The first smoke test is the reason. On house-prices, `majority of top 5` beat all
36 columns on the selection block — 0.13753 against 0.14070 — and then scored
**0.00731 worse on the validation half**, at roughly two paired standard errors.
The search found a subset that suits its own rows. With 1,168 rows and a 234-row
selection block that is exactly what should be expected, and it is the failure
mode the tool has to be measured for.

So the question is not "does the selection block improve" — it will, by
construction, because that is what the argmax is taken on. The question is
**whether the improvement survives onto rows the search never saw.**

`probe_vs_source_table` answers it: the selected table against its own parent, on
the validation half, with 0059's paired interval around the difference. That
comparison is honest here in a way it is not for most tables, because the subset
was chosen on rows held out of the table itself and `val` was never involved.

## Reading 1 — two things measured and one of them was my mistake

Run once with an individually-scored argmax and a selection block that was a fixed
**fraction** of the table. Both were wrong, in different ways, and both readings
are kept here because they are what the current design was chosen against.

| | selection block | validation gain |
|---|---|---|
| block = 20% of the table (~6 months on Rossmann) | wins by 0.019 | **−0.0200** |
| block = the validation block's span (44 days) | wins by 0.0071 | **−0.0026** |
| the same, Restaurant `t7` | wins by 0.0100 | +0.0004 |

**The block must be a span, not a fraction.** A fifth of Rossmann's work half is
six months while the run is ranked over forty-five days, so the search answered a
question nobody asked and dropped `DayOfWeek` and `Promo`. 0043 and 0060 both say
this in advance; the fix is in `_selection_split`.

**And then the search still bought nothing**, and 0059's interval says why rather
than leaving it a mystery: Rossmann's gain of −0.0026 sits against a **block sd of
0.0050**, Restaurant's +0.0004 against 0.0023, and both come back
`sign_stable: False`. The margin the argmax was choosing by on the block (0.0071)
is barely wider than the block's own noise. **It was selecting noise.**

## What changed for reading 2, and it is one thing (0017)

The criterion, from individual score to **pair-ensemble score** — which is what the
winner actually did and what the first version failed to copy:

> *"I ran over 500 random models and systematically calculated the validation error
> on each pair-ensemble of models. From the best model pairs I built a larger
> ensemble... Then I got the idea to take the features from all of the selected
> models together and combine them into one model: this turned out to work very
> well."*

A subset now goes forward only if it improves *another* subset when averaged with
it. That selects for complementarity, which luck supplies far less readily than it
supplies a good individual score, and it is 0044's argument one level down.

The selection block is unchanged, the draws are unchanged, the seed is unchanged.
Only the ranking criterion moved, so this reading is attributable.

## The prediction, registered before running reading 2

* **Rossmann `t13` turns positive but small: 0.000 to +0.004.** The mechanism the
  pair criterion removes is selection noise, and the noise it was drowning in was
  0.0050 of block sd against a −0.0026 reading — so recovering to a small positive
  is the most this change alone can be worth.
* **Restaurant `t7` stays near zero.** It only dropped 3 of 58 columns before, so
  there was little selection noise to remove; and its block sd is 0.0023.
* **The chosen candidate narrows less than before.** A pair criterion keeps
  columns that are useful in *some* good model rather than in the single best one,
  so the union should be wider than the 30 columns the majority-of-singles chose.

**The falsifier, and it closes this line: `probe_vs_source_table` still negative on
Rossmann.** That would mean the pair criterion is not the missing piece either,
that the difference from the winner is scale (500 models and 125,000 pairs against
96 and 4,560) or the width of the starting table rather than the criterion, and
that a tool of this shape does not pay on tables an agent built one probed step at
a time.

## What this is not

Not a search for a better Rossmann score — no submission is written, and both
boards are spent (0045, 0066). The tables are instruments: what is being measured
is whether the harness can express a choice it previously could not, and whether
that choice survives contact with rows it was not made on.
"""
import os
import shutil
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from tabular_agent.harness import journal  # noqa: E402
from tabular_agent.harness.session import RunSession  # noqa: E402
from tabular_agent.harness.stop import StopPolicy  # noqa: E402
from tabular_agent.harness.tools import (  # noqa: E402
    inspect_data,
    probe_table,
    select_features,
)
from tabular_agent.taskspec import TaskSpec  # noqa: E402

RUNS = os.path.join(REPO, "_runs")
OUT_ROOT = os.path.join(RUNS, "feature_selection_probe")

#: How hard to search. Larger than the tool's default because these are wide
#: tables and the whole question is whether the search finds anything; a null
#: result at 48 draws would be ambiguous between "nothing to find" and "not
#: enough draws".
N_CANDIDATES = 96


def _rossmann(run_dir: str) -> RunSession:
    from prep_rossmann import ABOUT, COLUMN_NOTES, METRIC, OUT, SPLIT_ON, TEST_OUT

    return RunSession(
        run_dir, OUT,
        TaskSpec(target="Sales", metric=METRIC, notes=ABOUT,
                 column_notes=COLUMN_NOTES, forecast_horizon=1.0),
        stop_policy=StopPolicy(max_tables=25),
        val_fraction=0.05, holdout_fraction=0.05, split_on=SPLIT_ON,
        serving_columns=[c for c in pd.read_parquet(TEST_OUT).columns if c != "Id"],
        fit_time_budget_s=60 * 60,
    )


def _restaurant(run_dir: str) -> RunSession:
    from prep_restaurant import (
        ABOUT, COLUMN_NOTES, FORECAST_HORIZON, METRIC, OUT, SPLIT_ON, STREAMS, TEST_OUT,
    )

    return RunSession(
        run_dir, OUT,
        TaskSpec(target="visitors", metric=METRIC, notes=ABOUT,
                 column_notes=COLUMN_NOTES, forecast_horizon=FORECAST_HORIZON),
        stop_policy=StopPolicy(max_tables=25),
        val_fraction=0.05, holdout_fraction=0.05, split_on=SPLIT_ON,
        serving_columns=[c for c in pd.read_parquet(TEST_OUT).columns if c != "id"],
        fit_time_budget_s=60 * 60,
        streams=STREAMS,
    )


DATASETS = {
    "rossmann": (_rossmann, "rossmann_v7", "t13"),
    "restaurant": (_restaurant, "restaurant_v2", "t7"),
}


def build(name: str) -> tuple[RunSession, str]:
    """The source run's split and its shipped table, copied rather than recut."""
    builder, source_run, table = DATASETS[name]
    source = os.path.join(RUNS, source_run)
    run_dir = os.path.join(OUT_ROOT, name)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(os.path.join(run_dir, "tables"))
    shutil.copytree(os.path.join(source, "data"), os.path.join(run_dir, "data"))
    for suffix in (".parquet", ".pipeline.json"):
        shutil.copy(
            os.path.join(source, "tables", f"{table}{suffix}"),
            os.path.join(run_dir, "tables", f"{table}{suffix}"),
        )

    session = builder(run_dir)
    inspect_data(session)
    table_path = os.path.join(run_dir, "tables", f"{table}.parquet")
    scored = probe_table(session, table_path)
    journal.append_table(run_dir, {
        "id": table, "path": f"tables/{table}.parquet",
        "source_table": "data/work.parquet",
        "method": f"copied from {source_run}",
        "new_columns": [], "n_steps": 0, **scored,
    })
    return session, f"tables/{table}.parquet"


def run(name: str) -> dict | None:
    builder, source_run, table = DATASETS[name]
    if not os.path.isdir(os.path.join(RUNS, source_run)):
        print(f"  missing _runs/{source_run} — skipping {name}", file=sys.stderr)
        return None

    print(f"\n{'=' * 74}\n=== {name}: {source_run}'s {table} ===")
    session, table_rel = build(name)
    parent = journal.load(session.run_dir)["tables"][0]
    print(f"  parent probe score {parent['probe_score']:.5f}")

    out = select_features(session, table=table_rel, n_candidates=N_CANDIDATES, seed=0)

    print(f"\n  on the selection block ({out['selection_block_rows']:,} rows held out "
          f"of the table):")
    for c in out["candidates"]:
        print(f"    {c['choice']:<28} {c['selection_score']:.5f}   "
              f"{c['n_columns']} columns")

    if not out.get("selected"):
        print(f"\n  {out['note']}")
        return {"dataset": name, "selected": False}

    print(f"\n  chosen: {out['chosen']} — dropped {out['n_dropped']} of "
          f"{out['n_columns_considered']} columns")
    print(f"  dropped: {', '.join(out['dropped_columns'][:12])}"
          f"{' ...' if out['n_dropped'] > 12 else ''}")

    # The reading. Everything above is the search scoring itself; this is the
    # first number produced on rows the search never touched.
    gain = out.get("probe_vs_source_table")
    interval = out.get("probe_vs_source_interval") or {}
    print(f"\n  ON VALIDATION — rows the search never saw:")
    print(f"    parent          {parent['probe_score']:.5f}")
    print(f"    selected        {out['probe_score']:.5f}")
    print(f"    gain            {gain:+.6f}")
    se = interval.get("paired_se")
    if se:
        print(f"    paired se       {se:.6f}   ({abs(gain) / se:.1f} se, "
              f"mode {interval.get('mode')})")
    return {
        "dataset": name, "selected": True, "chosen": out["chosen"],
        "n_dropped": out["n_dropped"], "n_considered": out["n_columns_considered"],
        "parent_probe": parent["probe_score"], "selected_probe": out["probe_score"],
        "gain": gain, "paired_se": se,
        "dropped_columns": "|".join(out["dropped_columns"]),
    }


def main() -> int:
    wanted = sys.argv[1:] or list(DATASETS)
    unknown = [w for w in wanted if w not in DATASETS]
    if unknown:
        print(f"unknown dataset(s): {unknown}; known: {list(DATASETS)}", file=sys.stderr)
        return 1
    os.makedirs(OUT_ROOT, exist_ok=True)

    rows = [r for r in (run(name) for name in wanted) if r]
    scored = [r for r in rows if r.get("selected")]
    if scored:
        print(f"\n{'=' * 74}\nTHE READING")
        for r in scored:
            print(f"  {r['dataset']:<12} dropped {r['n_dropped']:>3} of "
                  f"{r['n_considered']:>3}   gain {r['gain']:+.6f}"
                  + (f"  ({abs(r['gain']) / r['paired_se']:.1f} se)"
                     if r.get("paired_se") else ""))
        if all(r["gain"] <= 0 for r in scored):
            print("\n  FALSIFIED: the selection block cannot pick a subset that")
            print("  generalises. The house-prices reading was the rule, not a")
            print("  small-data artifact, and this capability does not work as built.")
        out = os.path.join(RUNS, "feature_selection_probe.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n  wrote {out}")

    print("\n  These are probe scores on the validation half. No submission is")
    print("  written — both boards are spent (0045, 0066), and what is being")
    print("  measured is a capability, not a score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
