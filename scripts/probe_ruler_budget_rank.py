"""Does the ruler rank tables the same way at a real budget as at 200 trees?

Every decision a run makes goes through the probe: one lgbm, `n_estimators=200`,
fixed seed, scored on the run's validation half. A run makes 15-25 such decisions
— keep this table or its parent, branch here or there, and finally which table to
ship. What ships is not that model. It is a blend of three families at whatever
budget the terminal fit uses.

0054 measured the *sign* of one thing's value reversing between those two
regimes: v8's feature block is worth -0.0152 against the baseline at 200 trees
and +0.0087 at 5000 @ 0.03. That is a systematic shift, not noise. If the ruler
shifts like that in general, then every table decision this project has ever made
was taken on an instrument measuring a different model than the one that ships,
and no amount of widening the interval around it helps — an interval measures
variance, and this would be bias.

This script is the cheapest decisive test. It re-probes **every table of
Rossmann v7's chain** at higher budgets and compares the ranking against the
200-tree scores already in that run's journal.

## Why v7, and why every table rather than one

v7 is the run whose tables still replay at HEAD — its steps are transforms, a row
filter and target/frequency encoders, all of which predate `fit_history`, so
0051's rename does not break the lineage the way it breaks v8's `t24` (0055).
It is the only complete chain in `_runs/` on which this measurement is possible.

Every table, because the quantity of interest is a *ranking*, and a ranking needs
the whole ladder. Fourteen of v7's fifteen tables carry a probe score; `t1` is
unservable (it drops `Customers` without a guard) and was never scored at any
budget, so it cannot enter.

## What is measured, and it is three different things

1. **The control.** The 200-tree arm must reproduce the journal's `probe_score`
   to six decimals. If it does not, the copy-the-split machinery is broken and
   nothing else on this page means anything.
2. **Global rank correlation** between the 200-tree ranking and each higher
   budget's. This is the headline number the plan asked for.
3. **The decisions, one at a time.** A run never consults the global ranking. It
   consults `probe_vs_source_table` — *is this child better than the parent I
   derived it from* — and at the end it takes the argmax. So this reports the
   sign of all thirteen parent-to-child steps at every budget, counts the ones
   that flip, and prints which table each budget's argmax would have shipped.

Point 3 is the one that decides anything, and it is why a high Spearman would not
be an acquittal. The ladder falls from 0.236 to 0.132; the early steps are worth
0.02-0.04 each and no budget will reorder them. Every close call in this project
lives in the last six steps, which sit within 0.005 of each other.

## The prediction, registered before running

* **The global rank correlation is high — Spearman above 0.9.** The chain is
  dominated by four large early steps that nothing will reorder.
* **And it does not matter, because the flips are concentrated where the
  decisions are.** I expect **at least two of the last six parent-to-child steps
  to change sign** between 200 trees and 1000 @ 0.10. Those are the steps whose
  gains (`t9` -0.0044, `t10` +0.0049, `t11` +0.0002, `t12` +0.0001, `t13`
  +0.0016, `t14` -0.0011, `t15` -0.0011) are at or below the size 0054's shift
  operates at.
* **The argmax moves off `t13`.** `t14` and `t15` add columns that a 200-tree
  model cannot pay for and a 1000-tree model can. This is the sharpest form of
  the claim: `t13` is the table v7 actually shipped, and its 0.13813 private
  score is the best number this project has.

**The falsifier: zero sign flips in the last six steps and the same argmax at
every budget.** That would mean the ruler is fine, the cause named in the plan is
wrong, and the remaining candidate for the gap to the median is architecture
(0070).

## What the first pass found, and why there is a second

The prediction above is left as written — it is what was believed before the
measurement, and the pattern it got wrong is the point. The control passed on all
fourteen tables. Then:

* **Global Spearman 0.8769 (1000 @ 0.10) and 0.9429 (5000 @ 0.03).** Half right.
* **The tail-flip prediction is refuted: one step, not two, at each budget.**
* **The argmax moves at 5000 @ 0.03** (`t13` -> `t12`) and not at 1000 @ 0.10.

And the reasoning behind the prediction was wrong in a way the prediction itself
concealed. **The flips are in the largest steps, not the smallest.** `t2->t3`
(a row filter) reads +0.01398 / +0.00225 / -0.00903; `t3->t5` (a target encoding)
reads +0.04636 / -0.01008 / +0.00536. Those swing by 0.023 and 0.056, against
tail flips that swing by 0.0027 and 0.0022 — at or inside the ~0.005 block sd
this project has measured on Rossmann steps. A noise model predicts the reverse
pattern exactly.

The consequence for v7: at the chain's largest fork the ruler ranked the target
branch (`t5`, +0.046) above the frequency branch (`t4`, +0.018), and at both
higher budgets `t4` is the better table by 0.017 and 0.021. Every table from `t6`
to the `t15` end of the chain, including the `t13` that shipped, descends from
`t5`. Whether the other branch would have *finished* better is not shown by this
and is a separate measurement — extrapolating a step-level reading to a
chain-level outcome is the uncontrolled comparison 0066 and 0067 both record.

## The second prediction, registered before the two added budgets run

The first pass cannot say what the instability is *in*, because its two
non-incumbent budgets differ in rounds and in learning rate at once. So:

* **The instability is in capacity, not learning rate.** Holding rounds at 1000
  and moving the rate from 0.10 to 0.03 should leave the step signs largely
  alone; holding the rate at 0.03 and moving rounds from 1000 to 5000 should
  reproduce the flips. The mechanism argued for the flips — a target encoding
  supplies per-group structure a model can also reach by splitting, and a row
  filter removes rows whose rule a model can afford to learn — is about how much
  the model can *fit*, and total shrinkage is the axis that governs that.
* **`1000 @ 0.03` will look more like `200 @ 0.10` than like `5000 @ 0.03`**, on
  the same grounds: 1000 x 0.03 is a third of 1000 x 0.10's total shrinkage, so
  it is the *least* capable budget on the grid after the incumbent.

If instead the flips track the learning rate with rounds held, the mechanism
above is wrong, the finding is about shrinkage interacting with encoded columns
rather than about capacity, and the panel proposed off this measurement would be
varying the wrong axis.

## What this is not

Not a proposal to probe at 1000 trees. A ruler that costs three times more per
table is a budget decision with its own consequences (a run makes 15-25 of these
calls), and "the number that won here" is not that decision. This measures
whether the instrument is biased, not what to do about it.
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

from prep_rossmann import (  # noqa: E402
    ABOUT,
    COLUMN_NOTES,
    METRIC,
    OUT,
    SPLIT_ON,
    TEST_OUT,
)
from tabular_agent import probe  # noqa: E402
from tabular_agent.harness.session import RunSession  # noqa: E402
from tabular_agent.harness.tools import probe_table  # noqa: E402
from tabular_agent.io import load_table  # noqa: E402
from tabular_agent.metrics import greater_is_better  # noqa: E402
from tabular_agent.taskspec import TaskSpec  # noqa: E402

RUNS = os.path.join(REPO, "_runs")
SOURCE_RUN = os.path.join(RUNS, "rossmann_v7")
RUN_DIR = os.path.join(RUNS, "ruler_budget_rank")
OUT_CSV = os.path.join(RUNS, "ruler_budget_rank.csv")

#: The budgets. `current` is the control, not a candidate — every other column is
#: read against it, and its job is to reproduce the journal.
#:
#: The full grid, and the first pass was run on three of these five. That pass
#: found the ranking unstable and **could not say in what**: it compared
#: `1000 @ 0.10` against `5000 @ 0.03`, which differ in rounds *and* in learning
#: rate, so an instability could have come from either. That is the confound 0017
#: forbids, walked into for the sake of comparability with
#: `probe_capacity.BUDGETS` — the grid is still taken from there rather than
#: invented, but all five rows are now used instead of three.
#:
#: The two added rows are what separate the axes:
#:
#:   1000 @ 0.10  vs  1000 @ 0.03   — rounds held, rate varies
#:   1000 @ 0.03  vs  3000 @ 0.03  vs  5000 @ 0.03   — rate held, rounds vary
#:
#: `200 @ 0.10` remains the control rather than a candidate: it is the harness as
#: it stands, and it is what every score in `_runs/` is denominated in.
BUDGETS: list[tuple[str, dict]] = [
    ("200 @ 0.10", {"n_estimators": 200}),
    ("1000 @ 0.10", {"n_estimators": 1000, "learning_rate": 0.1}),
    ("1000 @ 0.03", {"n_estimators": 1000, "learning_rate": 0.03}),
    ("3000 @ 0.03", {"n_estimators": 3000, "learning_rate": 0.03}),
    ("5000 @ 0.03", {"n_estimators": 5000, "learning_rate": 0.03}),
]

#: The two comparisons the added rows exist for, as (label, a, b) with the axis
#: that differs. Printed as its own reading, because "the ranking is unstable in
#: capacity" and "the ranking is unstable in learning rate" are different
#: findings that would want different fixes.
AXIS_PAIRS: list[tuple[str, str, str]] = [
    ("learning rate (rounds held at 1000)", "1000 @ 0.10", "1000 @ 0.03"),
    ("rounds (rate held at 0.03)", "1000 @ 0.03", "5000 @ 0.03"),
]

#: The control budget, whose scores must match the journal.
CONTROL = BUDGETS[0][0]

#: How many trailing steps count as "where the decisions are". Six, because that
#: is where v7's ladder flattens: every step from `t10` on moves the probe by
#: less than 0.005, and the argmax that shipped is inside this window.
TAIL_STEPS = 6


@contextlib.contextmanager
def _swap_probe_params(params: dict):
    """Run the harness's own ruler at a different budget, then put it back.

    Same monkeypatch as `probe_capacity._swap_probe_params`, and for the same
    reason: the score has to come from `probe_table`, the function that produced
    every number in `_runs/`. A reimplementation that differed in one detail
    would look exactly like a budget effect.
    """
    original = probe.PROBE_PARAMS
    probe.PROBE_PARAMS = dict(params)
    try:
        yield
    finally:
        probe.PROBE_PARAMS = original


def journal_chain() -> list[dict]:
    """v7's scored tables, in the order it built them, each with its parent.

    Read from the journal rather than restated, so this cannot drift from the run
    it is about. Tables without a `probe_score` are dropped — `t1` is unservable
    and has no 200-tree number to compare against, so there is nothing to rank.
    """
    with open(os.path.join(SOURCE_RUN, "run_journal.json")) as f:
        tables = json.load(f)["tables"]
    scored = []
    for t in tables:
        if t.get("probe_score") is None:
            print(f"  skipping {t['id']}: {t.get('probe_unavailable', 'no probe score')[:70]}")
            continue
        parent = t.get("source_table") or ""
        scored.append({
            "id": t["id"],
            "parent": os.path.basename(parent).replace(".parquet", ""),
            "journal": float(t["probe_score"]),
        })
    return scored


def build_session(ids: list[str]) -> RunSession:
    """v7's split and all of v7's tables, in a run directory of their own.

    The split is **copied, not recut** (`probe_capacity_v7_chain`'s pattern). The
    validation half is what every probe score is measured on, and it is the same
    rows for every table — that constancy is the entire basis of the comparison
    this script is making. A recut split would silently compare each table
    against different rows than its journal number was measured on, and the
    control would be the only thing that noticed.
    """
    if os.path.exists(RUN_DIR):
        shutil.rmtree(RUN_DIR)
    os.makedirs(os.path.join(RUN_DIR, "tables"))
    shutil.copytree(os.path.join(SOURCE_RUN, "data"), os.path.join(RUN_DIR, "data"))
    for tid in ids:
        for suffix in (".parquet", ".pipeline.json"):
            src = os.path.join(SOURCE_RUN, "tables", f"{tid}{suffix}")
            if os.path.exists(src):
                shutil.copy(src, os.path.join(RUN_DIR, "tables", f"{tid}{suffix}"))

    serving = [c for c in pd.read_parquet(TEST_OUT).columns if c != "Id"]
    return RunSession(
        RUN_DIR, OUT,
        TaskSpec(target="Sales", metric=METRIC, notes=ABOUT,
                 column_notes=COLUMN_NOTES, forecast_horizon=1.0),
        val_fraction=0.05, holdout_fraction=0.05, split_on=SPLIT_ON,
        serving_columns=serving,
    )


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation, computed here rather than pulled in from scipy.

    Average ranks for ties, which cannot arise between two distinct probe scores
    but can if a budget scores two tables identically.
    """
    def ranks(xs: list[float]) -> list[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            shared = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = shared
            i = j + 1
        return out

    ra, rb = ranks(a), ranks(b)
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")


def gain(child: float, parent: float, metric: str) -> float:
    """A step's value, in the direction that means "better" for this metric."""
    return (child - parent) if greater_is_better(metric) else (parent - child)


def main() -> int:
    if not os.path.isdir(SOURCE_RUN):
        print(f"missing {SOURCE_RUN}", file=sys.stderr)
        return 1

    print("=== rossmann v7 — the whole chain, re-ruled ===")
    chain = journal_chain()
    ids = [t["id"] for t in chain]
    session = build_session(ids)
    metric = session.resolved().metric
    print(f"  {len(chain)} scored tables: {', '.join(ids)}")
    print(f"  work {len(load_table(session.data_path)):,} rows | "
          f"val {len(load_table(session.val_path)):,} rows\n")

    scores: dict[str, dict[str, float]] = {label: {} for label, _ in BUDGETS}
    for label, params in BUDGETS:
        print(f"  --- {label} ---")
        for t in chain:
            path = os.path.join(RUN_DIR, "tables", f"{t['id']}.parquet")
            started = time.monotonic()
            with _swap_probe_params(params):
                out = probe_table(session, path)
            elapsed = time.monotonic() - started
            score = out.get("probe_score")
            if score is None:
                print(f"    {t['id']:<5} unscored — {out.get('probe_unavailable')}")
                continue
            scores[label][t["id"]] = float(score)
            note = ""
            if label == CONTROL:
                delta = abs(float(score) - t["journal"])
                note = "  == journal" if delta < 5e-7 else f"  !! journal {t['journal']:.6f}"
            print(f"    {t['id']:<5} {score:.6f}   ({elapsed:.0f}s){note}")
        print()

    # ---------------------------------------------------------------- control
    print("=" * 72)
    print("CONTROL — the 200-tree arm against v7's own journal")
    bad = [t for t in chain
           if abs(scores[CONTROL].get(t["id"], float("nan")) - t["journal"]) >= 5e-7]
    if bad:
        print(f"  FAILED on {len(bad)} of {len(chain)} tables: "
              f"{', '.join(t['id'] for t in bad)}")
        print("  The split or the lineage did not reproduce. Nothing below is readable.")
        return 1
    print(f"  passed: all {len(chain)} tables reproduce to six decimals.")

    # -------------------------------------------------------- global ranking
    print("\n" + "=" * 72)
    print("1. GLOBAL RANK CORRELATION against the 200-tree ranking")
    common = [t["id"] for t in chain if all(t["id"] in scores[l] for l, _ in BUDGETS)]
    base = [scores[CONTROL][i] for i in common]
    for label, _ in BUDGETS[1:]:
        rho = spearman(base, [scores[label][i] for i in common])
        print(f"  {label:<12} spearman {rho:+.4f}   (n={len(common)})")

    # ------------------------------------------------- the actual decisions
    print("\n" + "=" * 72)
    print("2. THE DECISIONS — each parent-to-child step, at every budget")
    print("   (this is `probe_vs_source_table`, the number the agent acts on)")
    header = f"  {'step':<14}" + "".join(f"{l:>14}" for l, _ in BUDGETS)
    print(header)
    steps = []
    for t in chain:
        if not t["parent"] or t["parent"] not in scores[CONTROL]:
            continue  # a table derived from the work half has no scored parent
        row = {"step": f"{t['parent']}->{t['id']}", "child": t["id"]}
        for label, _ in BUDGETS:
            if t["id"] in scores[label] and t["parent"] in scores[label]:
                row[label] = gain(scores[label][t["id"]],
                                  scores[label][t["parent"]], metric)
        steps.append(row)
        cells = "".join(f"{row.get(l, float('nan')):>+14.5f}" for l, _ in BUDGETS)
        flip = "" if len({row.get(l, 0) > 0 for l, _ in BUDGETS}) == 1 else "   <- FLIPS"
        print(f"  {row['step']:<14}{cells}{flip}")

    tail = steps[-TAIL_STEPS:]
    print(f"\n  sign flips vs the {CONTROL} arm:")
    for label, _ in BUDGETS[1:]:
        allflip = [s for s in steps
                   if label in s and CONTROL in s and (s[label] > 0) != (s[CONTROL] > 0)]
        tailflip = [s for s in allflip if s in tail]
        print(f"    {label:<12} {len(allflip)} of {len(steps)} steps"
              f"   |   {len(tailflip)} of the last {len(tail)}: "
              f"{', '.join(s['step'] for s in tailflip) or 'none'}")

    # ------------------------------------------------------- which axis is it
    # The reading the first pass could not produce. Each pair moves exactly one
    # of (rounds, learning rate) and holds the other, so a disagreement can be
    # attributed. Reported as flips *between the pair*, not against the control:
    # the question is which axis the ranking is unstable in, and the incumbent
    # differs from both members of both pairs on both axes.
    print("\n" + "=" * 72)
    print("2b. WHICH AXIS — one moves, the other is held (0017)")
    for axis, a, b in AXIS_PAIRS:
        both = [s for s in steps if a in s and b in s]
        flips = [s for s in both if (s[a] > 0) != (s[b] > 0)]
        worst = max((abs(s[a] - s[b]), s["step"]) for s in both) if both else (0, "-")
        print(f"  {axis}")
        print(f"    {a} vs {b}: {len(flips)} of {len(both)} steps flip"
              f"   ({', '.join(s['step'] for s in flips) or 'none'})")
        print(f"    largest disagreement {worst[0]:.5f} on {worst[1]}")

    # -------------------------------------------------------------- the argmax
    print("\n" + "=" * 72)
    print("3. THE ARGMAX — which table each budget would have shipped")
    better = max if greater_is_better(metric) else min
    picks = {}
    for label, _ in BUDGETS:
        pick = better(scores[label], key=lambda i: scores[label][i])
        picks[label] = pick
        print(f"  {label:<12} {pick}  ({scores[label][pick]:.6f})")
    moved = len(set(picks.values())) > 1

    print("\n" + "=" * 72)
    print("THE READING")
    tail_flips = max(
        len([s for s in tail
             if label in s and CONTROL in s and (s[label] > 0) != (s[CONTROL] > 0)])
        for label, _ in BUDGETS[1:]
    )
    if tail_flips == 0 and not moved:
        print("  FALSIFIED. The ruler orders v7's chain the same way at every budget,")
        print("  and every budget ships the same table. The probe is not the reason")
        print("  this project is above the median, and the remaining candidate named")
        print("  in the plan is architecture (0070).")
    else:
        print(f"  {tail_flips} of the last {len(tail)} decisions change sign with the budget"
              + (", and the argmax moves." if moved else ", though the argmax holds."))
        print("  The probe measures a different model than the one that ships, and it")
        print("  does so where the decisions are close — which is where all of them are")
        print("  after the first four steps. An interval around this number measures its")
        print("  variance; this is a shift. The open question becomes at which budget the")
        print("  ranking stabilises, and whether that is affordable 15-25 times per run.")
    print("\n  These are probe scores on v7's validation half, not Kaggle scores, and")
    print("  the chain is one run of one dataset. What generalises is the mechanism —")
    print("  a ruler fitted at a budget nothing ships — not the size of the numbers.")

    rows = [{"table": t["id"], "parent": t["parent"], "journal": t["journal"],
             **{l: scores[l].get(t["id"]) for l, _ in BUDGETS}} for t in chain]
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\n  wrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
