# 0100 — The full profile ships one early-stopped LightGBM by default

- **Date:** 2026-11-04
- **Status:** accepted and implemented (`Agent` / `ForecastingAgent` `blend=False`,
  `RunSession.blend`, `tools._terminal_fits`). Test:
  `tests/test_blend_ships.py::test_without_the_blend_one_early_stopped_lightgbm_ships`.
- **Decided by:** Ulf
- **Revises:** [0044](0044-what-ships-is-the-average-not-the-argmax.md), as a default
  only: the blend, its weights (0080) and its members (0087, 0099) remain, behind
  `blend=True`. Not measured against the early-stopped blend: see below.
- **Evidence:** `research/ieee_cross_row/` (r1, k1).

## The problem

0099 brought every blend member to capacity, and the first run with it showed the
cost. On 420,000 IEEE-CIS rows, over 30 minutes into the blend's fits only its LightGBM
had finished. XGBoost appears to have been dropped by 0099's budget check, with
CatBoost next; the reason is not logged and the run was stopped before this could be
confirmed. Dropping them would leave two LightGBMs and an elastic net weighted by
stability rather than score. Keeping them means a limit of an hour or more and an
estimated 3–4 hours at finalize, after a search of about 30 minutes.

## The decision

**By default the full profile ships what the minimal profile ships (0098): the
fitted ruler's fit on the chosen table.** Same learning rate, same stopping rule,
refitted on training + validation rows (0092). The table is still chosen by the
fitted ruler under 0094.

`blend=True` keeps the whole member set and 0099's stopping rule, for the long
version. It has not been run to completion on a large table.

## What this gives up

- **0044's argument against choosing a single model does not apply**, because nothing
  is chosen: one model is fitted, by a rule fixed in advance. But the diversification
  it paid for is gone.
- **On Rossmann the 200-round blend beat every single model** by 0.004–0.006
  (`research/capacity_ship_minimal/`); on IEEE r1 it lost to the single model on
  validation (0.9088 against 0.9125). Neither compares a single model with a blend at
  capacity. That measurement is still open.
- **On IEEE-CIS the single model matched the 200-round blend on one set of features**:
  Kaggle private 0.9164 against 0.9159, public 0.9382 against 0.9383, on r1's table
  refitted on all labelled rows.
- **The first run under this default scored 0.8997 private**, below 0.9023 for the
  raw columns. That difference is its table, not its model: it never built features an
  earlier run did (`research/ieee_cross_row/`).

## Rejected

- **A longer limit as the default.** Hours at finalize for a gain nobody has measured.
- **Removing the blend.** The open measurement needs it.
