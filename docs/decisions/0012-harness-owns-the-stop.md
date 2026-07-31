# 0012 — The harness decides when the run is over

- **Date:** 2026-07-30
- **Status:** accepted (thresholds revised same day — see [0014](0014-two-level-stop.md))
- **Commit:** `6936d4b`

## The problem

An agent left to itself does not stop. From a real run:

```
e1  baseline  0.6903      <- lower is better (logloss)
e4  catboost  0.2744
e5  catboost  0.2779
e6  catboost  0.2987
...
e11 catboost  0.2779
```

Four experiments took it from 0.69 to 0.27. Seven more moved it by 0.0026. It
would have kept going until the context ran out, spending real money to re-roll a
die. Nothing objected, because nothing in the loop had an opinion about *done*.

## Background: why "no improvement" is not obvious

The tempting rule is a threshold: "stop when the gain is under 0.001". That needs
a different number per metric and per dataset, and getting it wrong fails
silently in both directions — too strict stops mid-climb, too loose never stops.

The right yardstick is already in the leaderboard. Every experiment reports
`fold_spread`: the standard deviation of its score across cross-validation folds
— how much *this exact model's* number moves when you merely change which rows it
trained on.

In the run above the spread is about 0.02, and the seven experiments bought
0.0026. **The improvement is an order of magnitude smaller than the noise in the
measurement it is made of.** Re-run e9 with a different fold seed and it could
easily come out worse than e4.

So: a gain counts only if it exceeds `min_gain_ratio` (0.25) times the
incumbent's own fold spread. Everything smaller is the same measurement taken
twice.

Two details that matter:

- **The incumbent tracks the true best**, not the last thing that counted. Without
  that, a creeping series of sub-noise gains could be laundered into progress by
  moving the comparison point each time.
- **Patience is greater than 1.** A run branches: it tries something worse, backs
  up, tries something else. A dip before a real improvement is the search working.
  This is early stopping, with the same trade-off as in boosting.

## The decision

**The harness owns the stop, in two layers.**

- **Advisory.** Every training result carries `since_improvement` and a note
  saying where the run stands. A cooperative model reads it and finalises on its
  own, never getting refused.
- **Structural.** Past the patience budget, `train_tabular_model` stops accepting
  training calls.

Guidance for the cooperative case, structure for the rest — the same split as
everywhere else here.

Putting the rule only in the system prompt was rejected for the reason in 0004:
"stop when gains stall" is **unfalsifiable**. A model that keeps going has not
violated anything anyone can point at.

The stop logic is a pure function of the leaderboard (`stop.evaluate`), so the
interesting histories can be unit-tested against a hand-written leaderboard
instead of by training twenty models.

## Consequences

**Good**

- The first capped run did 8 experiments instead of 11 and reached the same
  score.
- **The advisory turned out to do most of the work.** In a later run the
  structural refusal never fired at all — `refusals: 0` — because the note alone
  moved the agent on. The enforcement is a backstop that has yet to be needed
  against a cooperative model.

**Bad**

- The rule says the *search* stopped moving. It knows nothing about whether the
  ideas are exhausted, and a run that is leaking the target converges fast and
  stops early, pleased with itself.
- `max_experiments` is a budget, not a judgement: a run that hits it may have
  been mid-climb, and the report does not say so.
- The first thresholds (`patience=4`) turned out to punish exploration, and were
  revised the same day. See 0014.

## Where it lives in the code

- `src/tabular_agent/harness/stop.py` — `StopPolicy`, `evaluate`, `advisory`.
- `src/tabular_agent/harness/session.py` — the policy hangs off `RunSession`,
  because it is this run's budget, not a library constant.
- `tests/test_stop.py` — hand-written leaderboards, including the observed
  0.003-over-six-experiments case.
