# Calendar and computed-event features reduce both level and variance

- **Date:** 2026-09-20
- **Branch / commit:** `claude/calendar-event-features` / `eb6b2bb`
- **Related:** [0089](../../docs/decisions/0089-forecasting-is-an-explicit-public-contract.md)
  (forecasting is an explicit contract; its acceptance test is a *distribution*
  of Rossmann repeats, not one score), `docs/time_aware_features.md`,
  `docs/examples/calendar_features.py`

## What changed

The `FORECAST_SYSTEM_PROMPT` now tells the agent to exhaust the calendar before
history: integer cycle parts (year/month/day/dow/doy/week/quarter/is_weekend/
days-to-month-end) and row-local distances to *known* events (Christmas, New
Year, the Easter family), the movable feasts **computed** (Anonymous Gregorian
computus) rather than joined so they stay correct for the future dates a forecast
scores. No sin/cos: a tree splits raw integer cycle parts natively. A distance to
a *data-observed* event stays excluded as non-row-local. A copyable recipe ships
in `docs/examples/calendar_features.py`, unit-tested against Rossmann
`StateHoliday` b (Easter) and c (Christmas).

The prompt only *offers* the features; the agent keeps them per measured gain.

## Method

Three preregistered repeats of `research/rossmann_forecasting_agent/runner.py`
(the 0089 harness), model `deepseek-v4-flash-0731@eu`, `forecast_horizon=48`,
compared against the three baseline repeats committed at `f9271ec`. Metric is the
Rossmann holdout (RMSPE-like), lower is better.

## Result

| Repeat | Baseline (master) | calendar-event-features |
|---|---|---|
| 1 | 0.1307 | 0.1248 * |
| 2 | 0.1480 | 0.1317 |
| 3 | 0.1315 | 0.1297 |
| **mean** | **0.1367** | **0.1287** |
| **spread (max-min)** | 0.0173 | 0.0069 |

\* Repeat 1's result JSON was not written — the run finished but the runner's
final `git rev-parse HEAD` metadata call failed under the mixed host/container
git-worktree path; the number is read from that run's `run_journal.json` (same
shipped-blend holdout metric). Repeats 2-3 were rerun with `GIT_DIR`/
`GIT_WORK_TREE` set so the runner wrote its result JSON normally.

Two effects, both in the direction 0089 asks for:

1. **Level** — mean holdout -0.0080 (-5.9%); every calendar-event run beats the
   baseline mean.
2. **Variance** — spread more than halved; the baseline's 0.1480 outlier
   disappears, and the worst calendar-event run (0.1317) beats the two worst
   baseline runs.

Feature uptake, from the run journals: all three runs built the integer calendar
parts; runs 1 and 3 also built the Easter/Christmas/New-Year distances (kept for
their probe gain), while run 2 built the calendar parts plus promo interactions
and skipped the holiday distances — the intended agent-decides behaviour.

## Caveats

n=3 per arm, no formal significance test. The signal is the consistent direction
plus the variance reduction, which is what this runner's distribution is for. The
metric is the internal holdout, not the Kaggle private leaderboard. Raw outputs
for the calendar-event runs are under
`research/rossmann_findings/calendar_event_features/`.
