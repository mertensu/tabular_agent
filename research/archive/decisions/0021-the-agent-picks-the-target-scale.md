# 0021 — The agent picks the target scale, from a fixed menu

- **Date:** 2026-08-10
- **Status:** accepted
- **Decided by:** Ulf ("can we allow the agent to transform the target using
  known transforms but not hardcode one?")
- **Related:** [0009](../../../docs/decisions/0009-target-hidden-from-transforms.md) (agent code never
  sees the target), [0017](0017-credit-by-ablation.md) (a comparison must differ
  by exactly one thing), [0022](../../../docs/decisions/0022-the-yardstick-must-measure-the-ranked-quantity.md)
  (which is what the first measurement of this feature actually taught us)

## The problem

A Kaggle notebook on house-prices, read against our own run, listed five basic
steps. The agent could do two of them. The one that stood out was **log-transform
the target**: our metric auto-detects as `rmsle` — squared error on the log scale
— while the models are fitted on raw dollars under squared loss. Those are
different objectives, and the model spends its capacity on the wrong one.

There was no path to it at all. 0009 hides the target from `transform_features`
by design, and nothing else touches the target.

## The decision

**A registry of stateless, invertible transforms, selected by name, per
experiment.**

    TARGET_TRANSFORMS = {"none": ..., "log1p": ..., "sqrt": ...}

Three properties, and each closes a way this could have gone wrong.

**A menu, not code.** The transform must be invertible and stateless, and
neither is checkable on arbitrary code. Free-form code operating on the target is
also precisely what 0009 exists to prevent. A name can be validated; a program
cannot.

**A `train_tabular_model` argument, not a run setting.** This was Ulf's framing
and it is the whole point of the record. Putting it on `TaskSpec` would mean *we*
decide once, at the top, that logging helps — an assumption nobody measures. As a
per-experiment argument it becomes an **experiment**: it lands on the leaderboard
row next to `model_type` and `params`, and `_fit_signature` includes it, so
0017's rule ("a comparison must differ by exactly one thing") covers it for free.

The first live run vindicated this immediately. The agent ran e2 (lgbm, `none`,
table t1 → 0.1366) and e4 (lgbm, `log1p`, same table → 0.1287) — a controlled
pair it constructed itself — then used `log1p` for all 21 remaining experiments.
Had this been a run setting we would have assumed the gain and never seen a
number. (What the number then turned out to be worth is 0022's subject, and it is
not a happy story.)

**Every score is reported on the original scale.** The models see `f(y)`;
predictions are inverted before any metric touches them and before they are
stored as out-of-fold values. Without this rule the leaderboard would rank
numbers in different units — a worse outcome than not having the feature. It also
keeps `explore_experiment`'s `residual` denominated in the units the agent asked
about; residuals in log-dollars would silently redefine every diagnostic call.

## What was rejected

- **`target_transform` on `TaskSpec`.** Above: it converts a measurable question
  into a project-level assumption.
- **Agent-written transform code.** Unverifiable invertibility, and a direct
  breach of 0009.
- **Box-Cox and Yeo-Johnson.** Their `lambda` is *estimated from the data*, which
  makes them learned state and a per-fold fitting problem. They belong to the
  `fit_*` family — imputation, outlier clipping, scaling — that the same Kaggle
  comparison showed we are missing entirely. Mixing them in here would have
  smuggled a leakage question into a change that otherwise has none.
- **Scoring on the transformed scale.** Cheaper (no inverse) and catastrophic: a
  transformed experiment would post a small `rmsle` because the numbers are
  smaller, and win.
- **Inverting per-model before blending.** The models share one target scale, so
  blending on that scale and inverting once is the same operation the
  cross-validated scores were computed with. Inverting first would make the
  bundle disagree with the leaderboard.

## What it costs

**Good**

- Real, if small: Kaggle 0.12368 → 0.12298. See 0022 before reading more into it.
- Three tools stay honest about it — `explain_table` takes the same argument, so
  a contribution is never measured against a fit that was never run.
- Old bundles keep working: `load_bundle` defaults a missing key to `"none"`.

**Bad**

- **Five call sites had to learn the inverse** (`train_cv`, `_score_on_val`,
  `predict_bundle`, oof storage, `explain_table`). Every one of them fails
  *silently and plausibly* if missed — predictions in log-dollars are still
  numbers, and for `rmsle` a missed inverse is flatteringly wrong rather than
  obviously wrong. This is tested rather than trusted, and it is the main risk
  the feature carries.
- The menu is short and the omissions are the useful ones (Box-Cox especially).
- One more dimension for the search to spend experiments on — and per 0022, one
  more dimension to overfit the validation half with.

## Where it lives in the code

- `src/tabular_agent/target.py` — the registry, domain checks, clipped inverse.
- `src/tabular_agent/models.py` — `train_cv` fits on `f(y)`, inverts before
  metrics and before storing oof.
- `src/tabular_agent/harness/tools.py` — `_score_on_val`, `_fit_signature`,
  the leaderboard row, `explain_table`, the bundle.
- `src/tabular_agent/pipeline.py` — `save_bundle`/`predict_bundle`.
- `tests/test_models.py` — `test_scores_stay_on_the_original_scale`,
  `test_oof_is_stored_on_the_original_scale`,
  `test_transform_round_trips_through_the_bundle`.
- `tests/test_harness_tools.py` — `test_the_bundle_predicts_in_the_original_units`,
  `test_vs_source_table_does_not_compare_across_target_scales`.
