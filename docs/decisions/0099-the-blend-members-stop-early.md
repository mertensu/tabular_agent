# 0099 — The full profile's blend members stop early, not at 200 rounds

- **Date:** 2026-11-03
- **Status:** accepted and implemented (`tools._terminal_fits` with `_StoppingRows`,
  `probe.stopped_rounds`). Tests:
  `tests/test_blend_ships.py::test_every_boosted_member_runs_at_the_rounds_it_stopped_at`,
  `tests/test_fitted_ruler.py::test_every_boosted_family_stops_where_held_back_rows_stop_improving`.
- **Decided by:** Ulf
- **Revises:** the member parameters of
  [0087](../../research/archive/decisions/0087-the-blend-ships-a-fixed-diverse-config-per-family-not-just-the-default.md)
  and of 0035's enumerated terminal fits (library defaults at 200 rounds). The
  members, the blend (0044) and its weights (0080) are unchanged. Extends
  [0098](0098-the-minimal-profile-ships-an-early-stopped-fit.md) to the full profile.
- **Evidence:** `research/ieee_cross_row/` (r1 readings).

## The problem

Every boosted member of the shipped blend ran at its library's defaults with
`n_estimators=200`. That is not one model size in three libraries. On IEEE-CIS r1's
chosen table (230k rows), stopped early on the latest 15% of the training rows:

| family | learning rate | rounds at stop | AUC on the stopping rows |
|---|---|---|---|
| lightgbm | 0.1 (default) | 227 | 0.926 |
| lightgbm | 0.03 | 697 | 0.929 |
| xgboost | 0.3 (default) | 128 | 0.893 |
| xgboost | 0.03 | 714 | 0.931 |
| catboost | auto | 2,431 | 0.933 |
| catboost | 0.03 | 3,097 | 0.930 |

At 200 rounds catboost is a tenth of its size and xgboost at learning rate 0.3 sits
0.04 below itself at 0.03; brought to capacity the three families are level. The
shipped blend scored 0.9088 on the run's validation rows, every member between
0.889 and 0.902; the early-stopped LightGBM that chose the table scored 0.9125 on
the same rows. The blend measured the libraries' defaults, not the families — the
comment on `TERMINAL_MODEL_TYPES` already said so and left it for a record.

## The decision

**Each boosted member runs at the fitted ruler's learning rate (0.03) for as many
rounds as it improves on the latest 15% of the chosen table's training rows**, with
the ruler's patience (200) and ceiling (5,000). The round count is read once per
member, then the member is cross-fitted and refitted as before.

- The default lgbm member is the fitted ruler's own fit on the chosen table, at the
  rounds `terminal_selection` recorded, as it was the probe's fit before.
- 0087's second configurations keep their subsampling and seed, and get their own
  round count.
- A family whose stopping fit fails keeps its library defaults.
- Elasticnet is unchanged.

## Why this is not tuning

As in 0098: one setting, the number of rounds, set by a fixed rule on training rows.
The learning rate, patience and ceiling are the fitted ruler's constants, the same
for every family and every dataset. Nothing reads the validation rows, and the agent
has no part in it.

## What it costs

- **One stopping fit per boosted member** (five, plus the ruler's rounds read back),
  and the members themselves are larger. On IEEE r1's table lgbm and xgb come to a
  few minutes per member.
- **Catboost can fall out of the blend on a large table.** At 2,400–3,100 rounds it
  is about ten minutes per fit on IEEE, an hour for its folds. When the stopping
  fit's time times the number of fits exceeds `fit_time_budget_s`, the member is
  skipped before its folds start, like any fit over budget. The blend then has fewer
  families.
- **The stopping rows are 15% of the training rows**, so on small tables the round
  count is noisy (0098).

## Rejected

- **A faster learning rate for catboost alone**, to keep it inside the budget. A
  per-family constant picked on the one dataset where it was too slow.
- **Each library's default learning rate, stopped early.** Cheaper, but xgboost at
  0.3 stops at 128 rounds and 0.04 below its own capacity: the defaults would still
  decide what the blend measures.
- **Replacing the blend with the single early-stopped fit.** On Rossmann the blend
  beat every single model by 0.004–0.006 (`research/capacity_ship_minimal/`); on
  IEEE it lost to one. Fixing the members first separates the two questions.
