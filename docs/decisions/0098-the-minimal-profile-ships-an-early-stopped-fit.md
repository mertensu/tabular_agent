# 0098 — The minimal profile ships an early-stopped fit, not the probe

- **Date:** 2026-11-03
- **Status:** accepted and implemented (`tools._finalize_minimal_run_unlocked` fits
  `probe.fit_fitted_ruler` on the chosen table and ships it). Test:
  `tests/test_minimal_profile.py::test_minimal_ships_the_early_stopped_fit_not_the_probe`.
- **Decided by:** Ulf
- **Revises:** the minimal profile's "one bounded model contract for table ranking
  and shipping" (`simplify-v0.2`), and so returns it to
  [0035](0035-the-search-is-over-features.md)'s part 1: the probe "only has to rank,
  never to ship". The rest of 0035 stands. The full profile is not changed.
- **Evidence:** `research/capacity_ship_minimal/` ("Reading r1 + r2", decision rule
  met twice on Porto), `research/credit_default_tabfm/` (README, both readings).

## The problem

`ta.Agent()` shipped the probe itself: LightGBM, 200 rounds, learning rate 0.1, the
same fit on 1,000 rows as on 600,000. A fixed budget is wrong in both directions.
On UCI credit default an early-stopped fit at learning rate 0.03 stopped at 35–70
rounds on 1,000 rows and 155–212 on 24,000, and the shipped probe lost to it by
0.019 / 0.023 / 0.011 AUC at 1k / 5k / 24k on the same chosen table. The capacity
study's House/Porto reading had already met its preregistered rule ("ship a fitted
model; the probe stays a ruler"); only the table choice was built (6499326), and
only for the full profile.

The probe also hid what the agent built: measured with the probe, the credit runs'
features were worth +0.002 to +0.005; measured with the early-stopped fit, +0.004 to
+0.009.

## The decision

**The probe ranks tables. What ships on the chosen table is the probe's family
stopped early.**

- At finalize the chosen table is fitted with `fit_fitted_ruler`: LightGBM at
  learning rate 0.03, up to 5,000 rounds, stopped after 200 without improvement on
  15% of the training rows (the latest 15% when the run is ordered), then refitted
  on all training rows at that many rounds. The evaluation score in the report is
  this fit's, on the validation rows.
- The refit on training + validation rows (0092) keeps the stopped round count.
- If the early-stopped fit fails, the probe ships, as before.
- Which table is chosen does not change: the minimal profile still takes the best
  probe score.

## Why this is not tuning

One setting, the number of rounds, is set by a fixed rule, not searched. It is read
on training rows, so the validation rows that rank tables are not used to fit it,
and the sealed rows are untouched. The agent has no part in it (0029). The learning
rate, ceiling and patience are constants, not chosen per dataset.

## What it costs

- **One more fit at finalize**, at a slower learning rate: seconds on 24,000 rows,
  about a minute on a large table.
- **The stopping slice is 15% of the training rows.** On 1,000 rows that is 120
  rows, so the round count is itself noisy.
- **The probe still chooses the table**, and it ranks only loosely like a fitted
  model (Spearman 0.06–0.96 on House/Porto). A table the fitted model would prefer
  can still be passed over.
- **Depth and regularization stay fixed.** At 1k and 5k credit rows a tuned XGBoost
  is still 0.004–0.009 AUC ahead.

## Rejected

- **Choosing the table with the fitted ruler too, as the full profile does.** One
  early-stopped fit per scored table; on 60 tables of a large dataset that is most
  of the 30-minute budget. Separate decision if the probe's choice is shown to cost.
- **A hyperparameter search at finalize** (the 30 Optuna trials of the credit
  baseline). That is tuning, against rows the search could read, and a larger
  change than the evidence asks for.
