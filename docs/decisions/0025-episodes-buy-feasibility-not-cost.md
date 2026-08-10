# 0025 — Episodes buy feasibility, not cost, and orientation scales with the cap

- **Date:** 2026-08-10
- **Status:** accepted, **not yet implemented**
- **Supersedes:** two claims in [0024](0024-the-episode-stop-is-structural.md) —
  the orientation tax being "roughly a fixed 2-3 calls", and the implication that
  episodes address the cost of a long run. **0024's decisions all stand**: the
  episode stop is structural, resumed agents orient themselves with
  `explore_journal`, the brief is required, the loop guard is mandatory.
- **Related:** [0022](0022-the-yardstick-must-measure-the-ranked-quantity.md)
  (whose lesson is applied to this record's own numbers)

## Why this record exists

0024 was written from v5, which ran episodes capped at 6 tool calls. v6 ran the
same loop at 16, and two of 0024's supporting claims turned out to be artefacts
of the small cap. The decisions they supported survive; the reasoning behind them
does not, and that reasoning was about to be used to size the real thing.

## Correction 1: the tax scales with the cap, it does not amortise

0024: *"The tax is roughly a fixed 2-3 calls, so it is proportional to
`1 / episode_length` and becomes ruinous exactly where short episodes are most
wanted."* That was inferred from the *opening* call sequences of v5's episodes.
Counting orientation across whole episodes — `explore_journal` plus every
`inspect_data` after the first — gives:

| cap | orientation share per episode |
|---|---|
| v5, cap 6 | 43%, 27%, 89% |
| v6, cap 16 | 26%, **50%**, 26%, 43% |

**Roughly a third of every episode, at both caps.** Given more room, the agent
orients more thoroughly rather than proportionally less — v6's episode 3 spent 8
of 20 calls on `explore_journal` alone.

This inverts the practical advice. Under 0024's model, a longer episode dilutes a
fixed cost and the fix is a bigger cap. Under the measurement, a bigger cap buys
proportionally more orientation, and the fix has to attack the *cause*: the
agent re-derives context because nothing tells it where it is. That makes the
derived brief the primary lever rather than a nicety — which is the same
conclusion 0024 reached, now for a reason that holds.

## Correction 2: episodes are more expensive per experiment at this scale

0024's problem statement leaned on context growing linearly and cost therefore
growing quadratically, implying episodes fix the cost. Measured:

| | v3 (one conversation) | v6 (5 episodes) |
|---|---|---|
| experiments | 24 | 18 |
| requests | 48 | 90 |
| total input tokens | 1,095,599 | **1,196,194** |
| **per experiment** | **45,650** | **66,455** |

**Episodes cost 46% more per experiment here.** Every episode pays the system
prompt and tool schemas again, re-runs `inspect_data`, and re-reads history that
the single conversation still had in front of it.

What episodes actually buy is **survivability**. The single-run cost model
`total ≈ ½ x slope x turns²` predicts 1.04M against v3's observed 1.10M, so it is
sound enough to extrapolate:

| experiments | one conversation | episodes |
|---|---|---|
| 24 | 1.1M, peak 44k | 1.2M, peak 28k |
| ~35 | **crossover** — episodes become cheaper per experiment | |
| ~60-80 | peak passes the window: **impossible** | still bounded |
| 150 | ~40M tokens, peak ~270k (dead) | ~10M, peak ~28k |

## The decision

**Episodes are the mechanism for long runs and the wrong mechanism for short
ones.** Do not put the loop on the default path. It should switch on when a run
is *budgeted* past roughly 35 experiments, and a 10-experiment run should keep
using one conversation, where it is both cheaper and better informed.

This is a sharper claim than 0024 made and it is falsifiable: if a 40-experiment
episode run does not come in cheaper per experiment than a 40-experiment single
run, the cost model is wrong.

## What this record does *not* claim

v6's best `val_score` was 0.12525 against v3's 0.11354, which invites "episodes
cost quality". **That inference is not available.** Per 0022, `val_score` on 292
rows has a standard error of ~0.010, so a 0.012 gap is barely one standard error
— and drawing a conclusion from a difference inside the noise is precisely the
mistake 0022 exists to record. Whether episodes cost quality is **unmeasured**,
and measuring it needs either many more rows or repeated runs.

## What it costs

**Good**

- The overnight run has a mechanism with a quantified regime of validity rather
  than an assumption.
- Peak context is bounded by *episode* length, not run length — v6's peaks
  (18k, 21k, 23k, 28k, 19k) rise and fall with what happened inside each episode
  and show no run-length trend.

**Bad**

- **A third of every episode is spent re-deriving state**, and no fix for that is
  implemented. The brief is the lever and it is still unbuilt.
- Two of the three runs behind these numbers had different budgets, so the
  comparison is not perfectly controlled. The cost model matching v3 to within
  6% is the main reason to trust the extrapolation.
- Nothing here is implemented; the loop is still a script.

## Where it will live

- The same places 0024 named. Additionally: whatever decides *whether* to run
  episodes needs the experiment budget, which is `StopPolicy.max_experiments` —
  already on the session.
