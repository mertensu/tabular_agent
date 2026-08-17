"""How often does the ruler misprice a step, across every run that can be re-ruled?

`probe_ruler_budget_rank.py` re-ruled Rossmann v7's chain at five budgets and found
that the probe does not merely rank two close tables in the wrong order — it
**inflates most feature steps**, and the sign flips are just where the inflation
exceeds a step's real value. v7's whole chain is worth 0.104 as the ruler tells it
and 0.032 at 3000-5000 rounds.

That is one chain on one dataset, and sizing a lever from n=1 is the error this
project keeps recording (0069 is the freshest). Before anything is built on it, the
question is whether mispricing is **common**, and whether it falls anywhere
predictable.

## What is measured

Every run in `_runs/` whose journal carries probe scores, re-ruled at two budgets:

    LOW   200 @ 0.10   the harness as it stands, and the control
    HIGH  3000 @ 0.03  the second arm

Per parent-to-child step this records the gain at both budgets, whether the sign
flipped, and the **deflation ratio** ``gain_low / gain_high`` — because the flips
are the visible tip and the deflation is the phenomenon. Then it groups by the tool
that produced the step and by the cardinality of the key that step encoded.

## Why 3000 @ 0.03 is the second arm

Chosen on a principle, not a score, because `probe.py` is explicit that tuning the
ruler against a dataset is the 0022 failure. The principle: *high enough that the
model can reach structure the agent might otherwise supply by hand.*

The two things that ruled out the alternatives are measurements rather than taste.
`1000 @ 0.10` is anomalous — at matched total shrinkage it is worse than both 3000
and 5000 @ 0.03 on six of v7's fourteen tables, which is the ordinary result that a
lower rate over more rounds generalises better, and it means rounds and rate are
not interchangeable the way `probe_capacity.BUDGETS` assumes. And `5000 @ 0.03`
demonstrably *overfits* three of v7's tables (`t3`, `t5`, `t7` all degrade from
3000 to 5000), so it is not a neutral instrument either.

**There is no budget at which the ruler is simply correct**, which is the deeper
finding of the v7 pass: each table has its own capacity optimum, and the optimum
depends on the table, which is the thing being ranked. `3000 @ 0.03` is not the
truth. It is a second opinion, which is all the proposed panel ever claims.

## The corpus, and why it is smaller than `_runs/`

The probe arrived with 0035, so every run before it has no probe scores to compare
against — `rossmann_v1`-`v6`, `house_prices` through `v8`, and `seguro_v1`/`v2` are
all empty of them and cannot enter. What is left is four datasets:

    rossmann_v8      22 steps   916k rows, rmspe, ordered
    restaurant_v1    17 steps   227k rows, rmsle, ordered
    restaurant_v2    13 steps   227k rows, rmsle, ordered
    seguro_v3/v4     19 steps   476k rows, auc, random split, binary
    house_prices_v9  10 steps   1,168 rows, rmsle, random split

`rossmann_v7` is deliberately excluded: it is the run the hypothesis was formed on,
and leaving it in would let the finding be carried by the chain that produced it.
Its numbers are in `_runs/ruler_budget_rank.csv` for anyone comparing.

**`rossmann_v8` is expected to fail partially and that is not a bug.** Scoring a
table means replaying its lineage over the validation rows, and 0051's rename of
`HistoryEncoder`'s output broke v8's replay (0055). Steps at or after its first
`fit_history` call will come back unscored. They are reported as *did not replay*
rather than counted, which keeps a missing number from reading as a zero.

## The prediction, registered before running

* **Mispricing is common: 15-25% of steps flip sign**, and a clear majority deflate
  by more than 2x. v7 had 2 of 13 flip at this budget, which is 15%, and v7's chain
  is unusually long in cheap late steps that barely move anything either way.
* **house-prices is the discriminator and should show almost nothing.** 1,168 rows,
  where 200 trees is roughly the right size, so the two budgets should nearly agree
  and the deflation ratios should sit near 1.0. If house-prices deflates as much as
  Rossmann, the effect is not about capacity being too small for the data and the
  mechanism argued from v7 is wrong.
* **Porto Seguro deflates.** 476k rows and a binary target scored by auc, so it
  tests whether the finding survives a different metric and a random split rather
  than only an ordered one.
* **Deflation tracks the cardinality of the encoded key**, not the tool's name. This
  is the hypothesis that replaced "target encodings are overvalued", which v7
  refuted: `t9->t10` and `t10->t11` are target encodings whose value *rises* with
  capacity. The refined claim is that a step is overvalued in proportion to how much
  of what it supplies the model could have reached by splitting — so encoding a
  1,115-way identity is nearly all substitutable, and encoding a 7-way one is not.

**The falsifier: under 5% of steps flip and the median deflation ratio sits near
1.0.** That would mean v7's chain was atypical, its fork was bad luck rather than a
defect, and the two-budget panel is an expensive way to flag noise. The instrument
would then be acquitted and the remaining candidate for the gap to the median is
architecture (0070).

## What this is not

Not a proposal to probe at 3000 rounds. It measures how often a second opinion would
have disagreed, which is the thing that decides whether a panel is worth its cost.
"The number that won here" is not a setting to adopt.
"""
import contextlib
import json
import os
import shutil
import sys
import time

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from tabular_agent import probe  # noqa: E402
from tabular_agent.harness.session import RunSession  # noqa: E402
from tabular_agent.harness.tools import probe_table  # noqa: E402
from tabular_agent.io import load_table  # noqa: E402
from tabular_agent.metrics import greater_is_better  # noqa: E402
from tabular_agent.taskspec import TaskSpec  # noqa: E402

RUNS = os.path.join(REPO, "_runs")
OUT_CSV = os.path.join(RUNS, "ruler_deflation_across_runs.csv")

LOW = ("200 @ 0.10", {"n_estimators": 200})
HIGH = ("3000 @ 0.03", {"n_estimators": 3000, "learning_rate": 0.03})

#: The runs to re-rule, in the order they are cheapest to fail on. house-prices
#: first because it takes seconds and is the prediction's discriminator; v8 last
#: because it is the slowest and the one expected to replay only partially.
CORPUS = [
    "house_prices_v9",
    "restaurant_v2",
    "restaurant_v1",
    "seguro_v3",
    "seguro_v4",
    "rossmann_v8",
]

#: Tolerance on the control. The journal rounds `probe_score` to six decimals, so
#: anything at or below half a unit in the last place is the same number.
CONTROL_TOL = 5e-7


@contextlib.contextmanager
def _swap_probe_params(params: dict):
    """The harness's own ruler at a different budget, then put back.

    Same monkeypatch as `probe_capacity._swap_probe_params`: the score has to come
    from `probe_table`, the function that produced every number in `_runs/`, or the
    readings are not comparable with the journals they are about.
    """
    original = probe.PROBE_PARAMS
    probe.PROBE_PARAMS = dict(params)
    try:
        yield
    finally:
        probe.PROBE_PARAMS = original


def _session_for(run: str, journal: dict, ids: list[str]) -> RunSession:
    """A run's own split and tables, copied into a directory of their own.

    Every argument is read from the journal rather than restated per dataset, which
    is what makes six runs across four datasets one code path. The split is
    **copied, not recut** — probe scores are measured on the validation half, and
    the same rows for every table is the entire basis of the comparison.

    `serving_columns` is deliberately left `None`. 0030's servability replay is
    then skipped, which is safe here for a narrow reason: only tables the journal
    *already scored* are re-ruled, so each one already passed that check during its
    own run. Reconstructing the serving schema per dataset would add four
    dataset-specific branches to a script whose point is that it has none.
    """
    run_dir = os.path.join(RUNS, f"deflation_{run}")
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(os.path.join(run_dir, "tables"))
    shutil.copytree(os.path.join(RUNS, run, "data"), os.path.join(run_dir, "data"))
    for tid in ids:
        for suffix in (".parquet", ".pipeline.json"):
            src = os.path.join(RUNS, run, "tables", f"{tid}{suffix}")
            if os.path.exists(src):
                shutil.copy(src, os.path.join(run_dir, "tables", f"{tid}{suffix}"))

    task = journal["task"]
    split = journal.get("split") or {}
    return RunSession(
        run_dir,
        split.get("source_path") or os.path.join(RUNS, run, "data", "work.parquet"),
        TaskSpec(target=task["target"], task_type=task.get("task_type"),
                 metric=task.get("metric")),
        val_fraction=split.get("val_fraction", 0.2),
        holdout_fraction=split.get("holdout_fraction"),
        split_on=split.get("split_on"),
    )


def _key_cardinality(run: str, table: dict, parent_id: str | None) -> tuple[str, int | None]:
    """What the step keyed on, and how many distinct values that key had.

    Read off the **parent** table, because that is the frame the step saw. Returns
    the widest key when a step encoded several columns at once: the hypothesis is
    about how much identity structure the step handed the model, and the widest key
    is what governs that.

    ``(name, None)`` when the step keyed on nothing (a transform, a row filter) or
    the parent is not on disk — absent rather than zero, so it cannot be averaged
    in as if it were a measurement.
    """
    keys = table.get("encoded_columns") or table.get("group_by") or []
    if not keys or parent_id is None:
        return ",".join(keys), None
    path = os.path.join(RUNS, run, "tables", f"{parent_id}.parquet")
    if not os.path.exists(path):
        return ",".join(keys), None
    try:
        df = pd.read_parquet(path, columns=[k for k in keys])
    except Exception:  # noqa: BLE001 - a key built by the step itself is not in the parent
        return ",".join(keys), None
    return ",".join(keys), int(max(df[k].nunique() for k in keys))


def _rule_one_run(run: str) -> tuple[list[dict], str]:
    """Both budgets over one run's scored tables. Returns (step rows, status)."""
    path = os.path.join(RUNS, run, "run_journal.json")
    if not os.path.exists(path):
        return [], "no journal"
    with open(path) as f:
        journal = json.load(f)
    scored = [t for t in (journal.get("tables") or []) if t.get("probe_score") is not None]
    if not scored:
        return [], "no probe scores (predates 0035)"

    metric = journal["task"].get("metric")
    ids = [t["id"] for t in scored]
    session = _session_for(run, journal, ids)
    print(f"\n=== {run} — {len(scored)} scored tables, metric {metric} ===")

    got: dict[str, dict[str, float]] = {}
    stale: list[str] = []
    for t in scored:
        tpath = os.path.join(RUNS, f"deflation_{run}", "tables", f"{t['id']}.parquet")
        per: dict[str, float] = {}
        for label, params in (LOW, HIGH):
            started = time.monotonic()
            with _swap_probe_params(params):
                out = probe_table(session, tpath)
            score = out.get("probe_score")
            if score is None:
                print(f"  {t['id']:<5} {label:<12} did not replay — "
                      f"{str(out.get('probe_unavailable'))[:80]}")
                break
            per[label] = float(score)
            if label == LOW[0]:
                delta = abs(float(score) - float(t["probe_score"]))
                if delta >= CONTROL_TOL:
                    print(f"  {t['id']:<5} CONTROL MISMATCH: {score:.6f} vs journal "
                          f"{t['probe_score']:.6f}")
                    stale.append(t["id"])
                    break
            print(f"  {t['id']:<5} {label:<12} {score:.6f}"
                  f"   ({time.monotonic() - started:.0f}s)"
                  + ("  == journal" if label == LOW[0] else ""))
        if len(per) == 2:
            got[t["id"]] = per

    if stale:
        # A wrong split or a wrong TaskSpec would move every score, and the whole
        # run's readings would be self-consistent and meaningless. Refusing the run
        # is the only safe response; refusing the mismatched tables alone would keep
        # the ones that happened to agree by luck.
        return [], f"control failed on {len(stale)} table(s) — run excluded"

    rows = []
    for t in scored:
        parent = os.path.basename(t.get("source_table") or "").replace(".parquet", "")
        if t["id"] not in got or parent not in got:
            continue
        lo = got[parent][LOW[0]] - got[t["id"]][LOW[0]]
        hi = got[parent][HIGH[0]] - got[t["id"]][HIGH[0]]
        if greater_is_better(metric):
            lo, hi = -lo, -hi
        key, card = _key_cardinality(run, t, parent)
        rows.append({
            "run": run, "metric": metric, "step": f"{parent}->{t['id']}",
            "method": t.get("method"), "key": key, "key_cardinality": card,
            "gain_low": round(lo, 6), "gain_high": round(hi, 6),
            "flipped": (lo > 0) != (hi > 0),
            # Only meaningful when the low arm says the step helped, which is the
            # case the harness acts on. A ratio through zero is not a ratio.
            "deflation": round(lo / hi, 3) if hi and lo > 0 and hi > 0 else None,
        })
    return rows, f"{len(rows)} comparable steps of {len(scored)} tables"


def main() -> int:
    wanted = sys.argv[1:] or CORPUS
    all_rows, status = [], {}
    for run in wanted:
        try:
            rows, note = _rule_one_run(run)
        except Exception as e:  # noqa: BLE001 - one bad run must not lose the others
            rows, note = [], f"{type(e).__name__}: {e}"
        status[run] = note
        all_rows += rows
        print(f"  -> {note}")

    print("\n" + "=" * 72)
    print("STATUS BY RUN")
    for run, note in status.items():
        print(f"  {run:<20} {note}")

    if not all_rows:
        print("\nno comparable steps — nothing to read.")
        return 1
    df = pd.DataFrame(all_rows)

    print("\n" + "=" * 72)
    print("1. HOW OFTEN THE TWO BUDGETS DISAGREE")
    print(f"  {len(df)} steps across {df.run.nunique()} runs")
    print(f"  sign flips: {int(df.flipped.sum())} "
          f"({df.flipped.mean() * 100:.0f}% of steps)")
    helped = df[df.gain_low > 0]
    print(f"  of the {len(helped)} steps the ruler says helped, "
          f"{int((helped.gain_high <= 0).sum())} do not help at capacity")
    d = df.deflation.dropna()
    if len(d):
        print(f"  deflation ratio (low/high, steps positive at both): "
              f"median {d.median():.2f}, "
              f"{int((d > 2).sum())} of {len(d)} above 2x")

    print("\n" + "=" * 72)
    print("2. BY RUN — the cross-dataset reading, and house-prices is the test")
    for run, g in df.groupby("run"):
        dd = g.deflation.dropna()
        print(f"  {run:<20} {len(g):>3} steps   flips {int(g.flipped.sum()):>2}"
              f"   median deflation "
              f"{f'{dd.median():.2f}' if len(dd) else '  n/a'}")

    print("\n" + "=" * 72)
    print("3. BY TOOL — v7 refuted the tool-family story; does the corpus agree?")
    for method, g in df.groupby("method"):
        dd = g.deflation.dropna()
        print(f"  {str(method):<16} {len(g):>3} steps   flips {int(g.flipped.sum()):>2}"
              f"   median deflation "
              f"{f'{dd.median():.2f}' if len(dd) else '  n/a'}")

    print("\n" + "=" * 72)
    print("4. BY KEY CARDINALITY — the replacement hypothesis")
    keyed = df[df.key_cardinality.notna() & df.deflation.notna()]
    if len(keyed) < 4:
        print(f"  only {len(keyed)} keyed steps with a usable ratio — not readable.")
    else:
        rho = keyed.key_cardinality.corr(keyed.deflation, method="spearman")
        print(f"  spearman(key cardinality, deflation) = {rho:+.3f}  (n={len(keyed)})")
        print("  the widest and narrowest keys measured:")
        for _, r in keyed.nlargest(3, "key_cardinality").iterrows():
            print(f"    {r.run:<16} {r.step:<12} {str(r.key)[:34]:<34} "
                  f"card {int(r.key_cardinality):>6}  deflation {r.deflation:>7.2f}")
        for _, r in keyed.nsmallest(3, "key_cardinality").iterrows():
            print(f"    {r.run:<16} {r.step:<12} {str(r.key)[:34]:<34} "
                  f"card {int(r.key_cardinality):>6}  deflation {r.deflation:>7.2f}")

    print("\n" + "=" * 72)
    print("THE READING")
    flip_rate = df.flipped.mean()
    med = d.median() if len(d) else float("nan")
    # The verdict is about the corpus, and this script is routinely invoked on one
    # run to smoke-test it. Printing "FALSIFIED" off a single dataset — which the
    # first version did, on the one dataset the prediction says should show
    # nothing — is 0065's flattering summariser in the reporting layer again.
    if set(wanted) != set(CORPUS):
        print(f"  PARTIAL CORPUS ({', '.join(wanted)}) — no verdict.")
        print("  The falsifier is defined over all four datasets; a subset cannot")
        print("  reach it, least of all house-prices, which the prediction says")
        print("  should show nothing either way.")
    elif flip_rate < 0.05 and (med != med or abs(med - 1) < 0.25):
        print("  FALSIFIED. The two budgets agree almost everywhere, v7's chain was")
        print("  atypical, and a panel would be an expensive way to flag noise. The")
        print("  instrument is acquitted; the remaining candidate is architecture (0070).")
    else:
        print(f"  The two budgets disagree on the sign of {flip_rate * 100:.0f}% of steps,")
        print(f"  and the median step the ruler likes is inflated {med:.2f}x.")
        print("  Mispricing is not a property of v7's chain. Whether a panel is the")
        print("  right response is a design decision and wants its own record — this")
        print("  measures how often a second opinion would have disagreed, nothing more.")
    print("\n  All numbers are probe scores on each run's own validation half. No")
    print("  intervals: a paired interval needs both budgets' stored predictions,")
    print("  which this script overwrites per budget by design (0036's instrument")
    print("  keys its cache on the table, not on the ruler).")

    df.to_csv(OUT_CSV, index=False)
    print(f"\n  wrote {OUT_CSV}  ({len(df)} steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
