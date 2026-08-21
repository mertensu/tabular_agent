# results/

Small numeric **readings** carved out of the gitignored `_runs/` directory.

`_runs/` holds ~11 GB of agent output — journals, transcripts, and fitted models
— and is disposable: `scripts/run_*.py` and `scripts/probe_*.py` reproduce it.
But the *findings* those runs produced are a few KB of CSV each, and losing them
to a reproduce-from-scratch step is a false economy. Those CSVs live here, in
version control, so the numbers a decision record cites can be inspected without
re-running anything.

These are **readings, not truth**: a CSV records what one run measured on one
day. The reasoning about what a reading *means* lives in `docs/decisions/`; this
directory is only the raw table behind a claim.

**These are reductions, and they cannot answer the block-level questions.** Each
row here is a scalar summary. In particular, none of these files — nor any stored
run artifact — carries the per-block gains, `block_sd`, the candidate-vs-incumbent
contrast, or the panel vote matrix. `blockwise_difference()` computes those
transiently and only its scalar `gain` survives into the journal (and `probe_se`
is `None` on the ordered runs, so `block_sd` was not even kept for the run's own
threshold); validation predictions are not persisted, so nothing is recoverable
offline. So `ruler_deflation_across_runs.csv` supports **budget-deflation** and
capacity-survival analysis (its `gain_low`/`gain_high` are a table re-ruled at a
low vs. a high budget) but **cannot** test whether `gain > k·block_sd`
discriminates better than `gain > 0`. And even the `block_sd` in the code is the
**candidate-vs-parent** contrast, while the stop rule judges
**candidate-vs-global-incumbent** — the wrong geometry regardless. Answering that
needs an *instrumented re-run* that computes `blockwise_difference()` on both
contrasts and persists both full dicts per step; see `HANDOFF.md`.

| File | What it measures | Produced by |
|------|------------------|-------------|
| `capacity_probe.csv` | Score vs. tree budget / learning rate per table, across datasets — the "capacity, not features" probe | `scripts/probe_capacity.py` |
| `capacity_v7_chain.csv` | Same probe over the Rossmann v7 feature chain | `scripts/probe_capacity_v7_chain.py` |
| `capacity_terminal_blend.csv` | Blend-arm val vs. sealed scores at capacity | `scripts/probe_capacity_terminal_blend.py` |
| `feature_selection_probe.csv` | Gain from dropping columns, with paired SE and the dropped set | `scripts/probe_feature_selection.py` |
| `ruler_budget_rank.csv` | How table ranking shifts as the ruler's budget grows (200→5000 trees) | `scripts/probe_ruler_budget_rank.py` |
| `ruler_deflation_across_runs.csv` | Per-step gain low/high and deflation factor across runs | `scripts/probe_ruler_deflation_across_runs.py` |
| `winner_salesmodel.csv` | Per-model and blend val/sealed scores for the winner-style sales model | `scripts/probe_winner_salesmodel.py` |

To refresh one, re-run its script (it writes into `_runs/`) and copy the CSV
here.
