"""The `_latest` pair ablation, on a second dataset. Does 0071 generalise?

## Why this exists

0071 measured `fit_history`'s unrequested `_latest` / `_latest_ago` pair at
**+0.01428** probe rmspe on Rossmann — six times the agent gap and twelve times
the capability gap it was looking for, at 5/5 blocks and 2.5 block sd. It then
declined to change the library, for the reason this project keeps re-learning:
that is n=1, and 0069 is three days old and is exactly the mistake of acting on
one reading.

The notebook written before this script found the *mechanism* behind Rossmann's
number, and the mechanism is what makes a second dataset necessary rather than
merely nice:

* `fit_history` freezes **per fold**, correctly, so a training row's `_latest`
  comes from its own fold boundary — 4,041 distinct values, 0.1% zeros, ages
  1..305 days. It is a genuinely informative column and a tree will split on it.
* Every served row reads the **final** freeze, so `_latest` is one value per group
  for the whole horizon — 21 distinct values across 1,115 stores, ages 1..45.
* Rossmann's final freeze happened to land on **1 May 2015**, a public holiday on
  which 1,095 of 1,115 stores were shut, so the column read `0.0` for 98% of
  stores.

The model was trained on one column and served another that shares its name. That
is why the ablation is worth 0.014 and not the 0.0014 an idle column costs (0069):
the pair is not useless, it is confidently wrong.

**How much of that is the defect and how much is the holiday is precisely what one
dataset cannot say.** Restaurant's freeze is 2017-03-18, an ordinary Saturday on
which 772 of 829 shops reported. If the price survives a freeze with no accident
in it, the defect is structural; if it collapses, the defect is a lottery, which
is a different finding with a different fix.

## What is measured

v2's **t7** — the table that shipped (0065) — against the same table with exactly
two columns dropped: `air_store_id_visitors_latest` and `..._latest_ago`. Nothing
else changes: same rows, same split, same lineage, both scored by the run's own
ruler.

* **The probe pair**, in one session, so the ablated table is a child of t7 and
  `probe_vs_source_table` carries 0059's block interval for free. This is the
  reading that corresponds to Rossmann's +0.01428.
* **Two sealed arms**, each a run directory with exactly one table on the board,
  driven through the real `finalize_run`. Separate arms because `finalize_run`
  ships the **argmax** of the probe scores — the gotcha that made 0071's first
  printed "capability gap" wrong. With one table per arm there is no argmax to go
  astray.
* **A control**, and it is fatal: the t7 arm must reproduce v2's journalled blend
  val 0.519707 / sealed 0.505713. If it does not, the harness is not being driven
  the way v2 drove it and the ablated arm is not a controlled comparison.

Restaurant's t7 also differs from Rossmann's table in a way that cuts *against*
the pair mattering, and that is deliberate: it carries **six window means** from
the same encoder (7/14/30/60/90/180-day), so the group's recent level is already
described six times over. Rossmann's table had one window per history call.

## The registered prediction

**Direction: dropping the pair still helps** — the train/serve distribution gap is
structural and does not need a holiday to exist.

**Magnitude: much smaller than Rossmann's +0.01428. I expect +0.000 to +0.005**,
for three reasons stated before the run:

1. no closure accident at the freeze, so the served value is a plausible number
   rather than a zero;
2. the spine is **ragged** (0054), so `_latest_ago` already varies at training
   time and the two distributions sit closer together;
3. six window means leave the pair less unique room.

**And the interval may well come back `sign_stable: False`.** That is not a
failure of the experiment — it *is* the finding: the defect would be real and its
price dataset-dependent, which means the fix's default has to be argued from the
mechanism rather than read off a score.

**If the ablation comes back negative** — the pair helps here — then 0049's
argument holds wherever persistence is strong, and the library change must default
to *keeping* the pair and merely allow declining it. That outcome would be worth
more than a confirmation.

## What this is not

Not a submission. Nothing here goes to Kaggle: the question is about `src/`, and
0066 already spent the Restaurant private board on a different one. The diagnostic
section below reports the served-side shape of the column, which is the part that
transfers to any dataset with a freeze point.
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from prep_restaurant import (  # noqa: E402
    ABOUT,
    COLUMN_NOTES,
    FORECAST_HORIZON,
    METRIC,
    OUT,
    SPLIT_ON,
    STREAMS,
    TEST_OUT,
)
from tabular_agent.encoders import HistoryEncoder  # noqa: E402
from tabular_agent.harness import journal  # noqa: E402
from tabular_agent.harness.session import RunSession  # noqa: E402
from tabular_agent.harness.stop import StopPolicy  # noqa: E402
from tabular_agent.harness.tools import (  # noqa: E402
    finalize_run,
    inspect_data,
    probe_table,
    transform_features,
)
from tabular_agent.taskspec import TaskSpec  # noqa: E402

SOURCE_RUN = os.path.join(REPO, "_runs", "restaurant_v2")
OUT_ROOT = os.path.join(REPO, "_runs", "restaurant_latest_pair")

#: The table that shipped in v2, and the pair it carries from its t3 history step.
BASE = "t7"
PAIR = ["air_store_id_visitors_latest", "air_store_id_visitors_latest_ago"]

#: What v2 recorded for the arm that shipped. Read as *what the control expects to
#: find*, not as an input to anything.
V2_SHIPPED = {"probe": 0.521351, "val": 0.519707, "sealed": 0.505713}

#: Rossmann's reading, for the comparison this whole script exists to make.
ROSSMANN = {"with": 0.140982, "without": 0.126702, "gain": 0.014280,
            "block_sd": 0.005760, "stable": True}

DROP_PAIR = (
    "def transform_features(df):\n"
    f"    return df.drop(columns={PAIR!r}, errors='ignore')\n"
)


def make_session(run_dir: str, tables: dict[str, str]) -> RunSession:
    """A run directory holding v2's split, byte for byte, plus named tables.

    `tables` maps the id it should have here to the id it has in v2. The split is
    copied rather than recut for the reason `probe_restaurant_collapsed_columns.py`
    gives: recutting would give the same rows in this case and there is no reason
    to depend on that.
    """
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(os.path.join(run_dir, "tables"))
    shutil.copytree(os.path.join(SOURCE_RUN, "data"), os.path.join(run_dir, "data"))
    for local, source in tables.items():
        for suffix in (".parquet", ".pipeline.json"):
            shutil.copy(os.path.join(SOURCE_RUN, "tables", f"{source}{suffix}"),
                        os.path.join(run_dir, "tables", f"{local}{suffix}"))
    return make_session_from(run_dir)


def make_session_from(run_dir: str) -> RunSession:
    """The session itself, for a directory whose data and tables are in place."""
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
    return session


def register(session: RunSession, table_id: str, source: str) -> dict:
    """Probe a copied table and put it on this run's board."""
    run_dir = session.run_dir
    path = os.path.join(run_dir, "tables", f"{table_id}.parquet")
    scored = probe_table(session, path)
    journal.append_table(run_dir, {
        "id": table_id,
        "path": f"tables/{table_id}.parquet",
        "source_table": "data/work.parquet",
        "method": f"copied from restaurant_v2 {source}",
        "new_columns": [],
        "n_steps": len(json.load(
            open(os.path.join(run_dir, "tables", f"{table_id}.pipeline.json"))
        )["steps"]),
        **scored,
    })
    return scored


# --------------------------------------------------------------------------
# 1. The diagnostic: what the column looks like on each side of the freeze.
# --------------------------------------------------------------------------

def served_side_shape() -> None:
    """The mechanism, before the price — and the part that names no dataset.

    Refits the same history the run's t3 fitted, on v2's own work half, and
    compares the column the model **trains** on against the column it is
    **served**. On Rossmann those two are different columns wearing one name; the
    question here is whether that is a property of `fit_history` or of a holiday.
    """
    work = pd.read_parquet(os.path.join(SOURCE_RUN, "data", "work.parquet"))
    val = pd.read_parquet(os.path.join(SOURCE_RUN, "data", "val.parquet"))
    t7 = pd.read_parquet(os.path.join(SOURCE_RUN, "tables", f"{BASE}.parquet"),
                         columns=[*PAIR, SPLIT_ON])

    enc = HistoryEncoder(by=["air_store_id"], value="visitors", order=SPLIT_ON,
                         windows=[(7, "mean"), (14, "mean"), (30, "mean"),
                                  (60, "mean"), (90, "mean"), (180, "mean")])
    enc.fit(work)
    served = enc.transform(val)

    freeze = pd.Timestamp(enc.frozen_at_, unit="D")
    on_freeze = work[work[SPLIT_ON] == work[SPLIT_ON].max()]
    print(f"\n  freeze point            : {freeze.date()} "
          f"({freeze.day_name()}), {len(on_freeze):,} of "
          f"{work['air_store_id'].nunique():,} shops reporting")

    print(f"\n  {'':<26} {'distinct':>9} {'zero':>7} {'age (days)':>14}")
    for label, values, ages in (
        ("training rows (per fold)", t7[PAIR[0]], t7[PAIR[1]]),
        ("served rows (final freeze)", served[PAIR[0]], served[PAIR[1]]),
    ):
        ages = ages.dropna()
        print(f"  {label:<26} {values.nunique():>9,} {(values == 0).mean():>7.1%} "
              f"{f'{ages.min():.0f}..{ages.max():.0f}':>14}")

    per_group = served.groupby("air_store_id")[PAIR[0]].nunique()
    print(f"\n  distinct values within one served group: max {per_group.max()} "
          f"(structural: the freeze does not move while the rows do)")

    scored = served[served["visitors"] > 0]
    for name in (PAIR[0], "air_store_id_visitors_30_mean"):
        c = scored[[name, "visitors"]].corr().iloc[0, 1]
        print(f"  corr({name:<34}, visitors) = {c:>6.3f}")


# --------------------------------------------------------------------------
# 2. The probe pair, in one session, so the interval comes with it.
# --------------------------------------------------------------------------

def probe_ablation() -> dict:
    print(f"\n{'=' * 74}\nTHE PROBE PAIR — t7, then t7 with the pair dropped")
    session = make_session(os.path.join(OUT_ROOT, "probe"), {BASE: BASE})
    base = register(session, BASE, BASE)
    print(f"  {BASE:<34} probe {base['probe_score']:.6f}")

    ok = abs(base["probe_score"] - V2_SHIPPED["probe"]) < 1e-5
    print(f"  v2 recorded                        probe {V2_SHIPPED['probe']:.6f}"
          f"   {'OK' if ok else '** MISMATCH **'}")
    if not ok:
        raise SystemExit("the copied table does not reproduce v2's probe score")

    out = transform_features(session, DROP_PAIR, table=f"tables/{BASE}.parquet")
    print(f"  {BASE} minus the pair{'':<17} probe {out['probe_score']:.6f}"
          f"   ({out['probe_vs_source_table']:+.6f} vs t7)")
    interval = out.get("probe_vs_source_interval") or {}
    for k, v in interval.items():
        print(f"      {k:<20} {v}")
    return {"with": base["probe_score"], "without": out["probe_score"],
            "gain": out["probe_vs_source_table"], "interval": interval,
            "ablated_table": out["table"], "run_dir": session.run_dir}


# --------------------------------------------------------------------------
# 3. The sealed arms, one table each so there is no argmax to go astray.
# --------------------------------------------------------------------------

def sealed_arm(name: str, source_run: str, source_table: str) -> dict:
    print(f"\n{'=' * 74}\nSEALED ARM {name}")
    run_dir = os.path.join(OUT_ROOT, name)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(os.path.join(run_dir, "tables"))
    shutil.copytree(os.path.join(SOURCE_RUN, "data"), os.path.join(run_dir, "data"))
    for suffix in (".parquet", ".pipeline.json"):
        shutil.copy(os.path.join(source_run, "tables", f"{source_table}{suffix}"),
                    os.path.join(run_dir, "tables", f"a1{suffix}"))

    session = make_session_from(run_dir)
    register(session, "a1", f"{os.path.basename(source_run)}/{source_table}")
    finalize_run(session)

    j = journal.load(run_dir)
    record = j.get("holdout") or {}
    blend = next((e for e in record.get("scores", [])
                  if (e.get("model_type") or "").startswith("blend")), None)
    for e in record.get("scores", []):
        sealed = e.get("holdout_score")
        print(f"    {e.get('model_type'):<42} val {e['val_score']:.5f}  "
              f"sealed {'n/a' if sealed is None else f'{sealed:.5f}'}")
    return {"blend_val": (blend or {}).get("val_score"),
            "blend_sealed": (blend or {}).get("holdout_score")}


def main() -> int:
    if not os.path.isdir(SOURCE_RUN):
        print(f"missing {SOURCE_RUN} — run scripts/run_restaurant_v2.py first",
              file=sys.stderr)
        return 1
    os.makedirs(OUT_ROOT, exist_ok=True)

    print(f"{'=' * 74}\nTHE MECHANISM — is the column the model trains on the one "
          f"it is served?")
    served_side_shape()

    probe = probe_ablation()

    arms = {
        "with_pair": sealed_arm("with_pair", SOURCE_RUN, BASE),
        "without_pair": sealed_arm(
            "without_pair", probe["run_dir"],
            os.path.splitext(os.path.basename(probe["ablated_table"]))[0]),
    }

    print(f"\n{'=' * 74}\nCONTROL — does the with_pair arm reproduce v2?")
    got = arms["with_pair"]
    fatal = False
    for key, want in (("blend_val", V2_SHIPPED["val"]),
                      ("blend_sealed", V2_SHIPPED["sealed"])):
        ok = got[key] is not None and abs(got[key] - want) < 1e-5
        fatal = fatal or not ok
        print(f"  {key:<14} v2 {want:.6f}  rebuilt "
              f"{'n/a' if got[key] is None else f'{got[key]:.6f}'}  "
              f"{'OK' if ok else '** MISMATCH **'}")
    if fatal:
        print("\n  The arm does not reproduce the run it copies, so the comparison "
              "below is not controlled. Read the probe pair only.")

    print(f"\n{'=' * 74}\nTHE READING (rmsle, lower is better)")
    print(f"  {'':<24} {'probe':>10} {'blend sealed':>14}")
    print(f"  {'t7, with the pair':<24} {probe['with']:>10.5f} "
          f"{arms['with_pair']['blend_sealed']:>14.5f}")
    print(f"  {'t7, pair dropped':<24} {probe['without']:>10.5f} "
          f"{arms['without_pair']['blend_sealed']:>14.5f}")
    d_sealed = (arms["with_pair"]["blend_sealed"] or np.nan) - (
        arms["without_pair"]["blend_sealed"] or np.nan)
    print(f"\n  the pair costs   probe {probe['gain']:+.5f}   sealed {d_sealed:+.5f}")

    print(f"\n  Rossmann (0071): probe {ROSSMANN['gain']:+.5f}, block sd "
          f"{ROSSMANN['block_sd']:.5f}, sign_stable {ROSSMANN['stable']}")
    iv = probe["interval"]
    if iv:
        print(f"  here           : probe {iv.get('gain'):+.5f}, block sd "
              f"{iv.get('block_sd'):.5f}, sign_stable {iv.get('sign_stable')} "
              f"({iv.get('blocks_favouring_a')}/{iv.get('n_blocks')} blocks)")

    print("\n  Read the two magnitudes against the registered prediction "
          "(+0.000..+0.005),")
    print("  and read the block interval before the point estimate. No submission "
          "is written.")

    with open(os.path.join(OUT_ROOT, "results.json"), "w") as f:
        json.dump({"probe": probe, "sealed": arms, "rossmann": ROSSMANN}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
