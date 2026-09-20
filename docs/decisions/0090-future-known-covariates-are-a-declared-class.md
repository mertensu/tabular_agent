# 0090  Future-known covariates are a declared class, and their windowed aggregates are a tool

- **Date:** 2026-09-20
- **Status:** accepted, implemented (`DataPlan.future_known`,
  `FutureWindowEncoder` + `future_window` lineage step,
  `fit_future_covariate_window` tool, forecast-only). Acceptance met: on the
  Energy Challenge, a fixed-LGBM feature ablation over the research builder's own
  four folds moved mean RMSE 55.94 → 52.96 (−2.98) with the tool, against the
  research builder's own 52.76 (−3.19) — the same gain family, ~93% of the gap
  closed. Tests in `tests/test_future_window.py`.
- **Related:** [0089](0089-forecasting-is-an-explicit-public-contract.md) (the
  forecast contract), [0047](0047-the-forecast-horizon-is-a-column-not-a-constant.md)
  (the horizon gates history), [0011](0011-stability-check.md) (row-local
  transforms are rejected if batch-dependent), [0062](0062-a-stream-is-a-declared-second-input-not-a-hidden-join.md)
  (observation-time-safe past inputs), [0088](0088-a-run-may-propose-a-capability-gap-but-not-change-the-product.md)
  (a run may propose a gap); empirical proof in `research/energy_horizon_features/`.

## What happened

On the Energy Forecasting Data Challenge a dataset-specific research builder
(`research/energy_horizon_features/`) found that **aggregates of the fully
provided future weather over the serving horizon** were the single largest lever
(one fixed LGBM, feature delta only: mean fold RMSE 53.77 → 51.37, −2.40), and it
carried the account's best Kaggle result. A generic `ForecastingAgent` run on the
same data (branch `claude/calendar-event-features`) could not build them and
landed ~2.7 RMSE worse.

This is not dataset-specific. The forecasting literature separates **past
covariates** (known only up to the cutoff) from **future covariates** (known over
the horizon at issue time: calendar, planned promotions, prices, weather
forecasts). Rolling/aggregate versions of future covariates — "total irradiance
this day", "promos in the next 7 days", "mean forecast temperature over the next
3 hours" — are a standard, high-value GBDT-forecasting family (M5, the Rossmann
winner, Darts `future_covariates`).

The agent cannot express the family, for a structural reason:

- `transform_features` is **row-local**; any window aggregate depends on other
  rows in the batch and 0011 rejects it, correctly.
- `fit_history` aggregates the **target backward** — trailing windows frozen in
  the past, gated to be at least one horizon old. It cannot aggregate an
  *exogenous* column, and it cannot look **forward** across the horizon, which is
  the whole point of a future covariate.
- `fit_stream_aggregate` is for *past* second inputs by observation time.

The deeper cause is the leakage model itself: it is binary — the target (leaky,
needs history/encoding) versus everything else (row-local, available at serving).
It has no notion that *a particular serving column's future values are known*, so
it cannot tell a legal forward aggregate from an illegal one.

## The decision

Add the missing class as a **declaration** plus a **tool**, mirroring how
`fit_history` made target history a first-class, replayable step rather than an
escape-hatch trick.

1. **`DataPlan.future_known`** — a per-serving-column declaration (a set/list of
   column names, subset of `serving_columns`). It states that the column's values
   over the forecast horizon are available at issue time. It is the user's
   assertion about the data, exactly like `serving_columns` and
   `forecast_horizon`; the harness never infers it. `validate()` checks the names
   are serving columns. Persisted in the run manifest and the bundle so replay
   applies the same contract.

2. **`fit_future_covariate_window`** — a tool that builds, per series key,
   serving-legal windowed aggregates of a declared future-known covariate. Only a
   column in `future_known` is eligible; the tool refuses any other column with
   the reason (the target, `load`, an undeclared covariate). Aggregate families:
   - **whole-period** keyed `(series, block, calendar-unit)` — e.g. day/week
     statistics that may look forward *within the delivered block*
     (`day_irr_sum`, `day_t2m_max`, `day_daylight_frac`);
   - **trailing** within the block, reset at the serving-block start
     (`roll4_t2m`, `roll16_irr` — the block-local rolling the energy builder
     used);
   - **forward-within-horizon** — a window ahead of the row bounded by the
     horizon (`next_3h_temp_mean`, `promos_next_7d`).
   Statistics: mean/sum/min/max/std/count, as `fit_history` already offers.

3. **A `future_window` lineage step** (a new encoder, sibling to
   `HistoryEncoder`) that stores what it needs to replay — the series key, the
   covariate, the window spec and calendar unit — and recomputes the aggregate at
   predict time from serving inputs alone. Like history it is bounded by
   construction and re-run by `Pipeline.transform`; unlike history it reads the
   covariate, never the target, and its windows are **not** gated by the horizon
   (a future covariate is known across it — that is the definition).

4. **Exposed to `ForecastingAgent` only.** The IID `Agent` has no horizon and no
   future-known columns; the tool is part of the forecast toolset, and the
   forecast brief names it beside `fit_history` as the exogenous counterpart to
   target history.

### The block boundary is the invariant

Every aggregate is keyed within one series and one delivered serving block, and
resets at the block start — never crossing a `dataset_id` or a block boundary.
The target, `load`, `P` and split helpers never enter the computation (asserted,
as the research builder asserts). This is what makes a forward-looking aggregate
serving-legal: at serving the model is handed one block and computes the same
block-local statistic it computed in training.

## The realism caveat, recorded so it is not forgotten

Serving-legal under a competition contract that *delivers realized future
covariates* (as Kaggle does here) is not the same as production-realistic. In
operation a future covariate is a **forecast issued at the cutoff**, whose quality
decays with lead — a 14-day-ahead weather forecast is nearly useless while a
1-day-ahead one is excellent. Training honestly therefore needs the covariate
*as it was known at each historical issue time* (archived forecasts by
issue-time + lead), not the realized values, or the model learns on information it
will not have and skews train-versus-serve. The tool builds the aggregates it is
given; whether those inputs are realized or as-of-issue is a property of the data
the user supplies, and `future_known` is the user asserting they are legitimately
known. The docstring and `inspect_data` must say this plainly.

## Acceptance test

On the Energy Forecasting Data Challenge, a `ForecastingAgent` run that declares
the weather columns `future_known` and is offered `fit_future_covariate_window`
should reproduce the research builder's gain family and close most of the ~2.7
RMSE gap to `research/energy_horizon_features/`, measured on the internal holdout
and confirmed on the Kaggle leaderboard. On a dataset with no future-known
columns declared, behaviour is byte-identical to today.

## What was rejected

- **Let `transform_features` do windows.** It would break 0011 for every user, to
  reach one legal case; the legality is a property of the column's future
  availability, which the escape hatch cannot know.
- **Overload `fit_history`.** History is backward and target-bound by design;
  forcing a forward exogenous mode into it would blur the horizon gate that keeps
  target history honest.
- **Infer future-known from the data** (e.g. "the column is present in the test
  file over the horizon"). Presence is not permission: realized future weather is
  present in a Kaggle test file and is *not* known in production. The assertion
  must be the user's, like every other leakage-relevant fact in the plan.

## What it cost

One `DataPlan` field, one tool, one encoder/lineage step, a forecast-brief line,
and the persistence of the new declaration in manifest and bundle. The IID path
is untouched; a forecast that declares nothing keeps today's behaviour.
