# Decision records

Why this project is built the way it is.

Each file records **one decision**: the problem it solved, what was chosen, what
was rejected, and what it cost. They are written for someone who knows data
science but has never seen this codebase — project-specific vocabulary is
explained where it first matters.

## How to read these

A decision record is **history, not documentation**. It was true on the day it
was written and it is never edited to match later reality. If a decision is
later reversed, a new record supersedes it and the old one stays put, because
the reasoning that turned out to be wrong is usually the most useful part.

That means: **do not treat these as a description of how the system currently
works.** For that, read the code — the rationale for individual lines lives in
docstrings and comments next to the thing it explains, which is the only place
it cannot silently drift out of date. These records explain the *shape* the code
has, not its current contents.

## What this project is

`tabular_agent` is a **leakage-safe tabular machine-learning harness**. An LLM
agent is given a dataset and a target column, and works towards a model that
generalises. The central claim is that it is *leakage-safe by construction*: the
agent physically cannot commit the standard sins (training on the target,
fitting an encoder on data the model will be scored against) because the tools
it is given are the only way it can touch the data, and none of them can express
those mistakes.

Vocabulary used throughout:

| term | meaning |
|---|---|
| **run** | one agent session against one dataset, all of it inside a `run_dir/` |
| **table** | a file: the dataset plus columns added by one tool call. Named `t1`, `t2`, … |
| **experiment** | one model trained with cross-validation on one table. Named `e1`, `e2`, … |
| **journal** | `run_journal.json` — the run's record: task, leaderboard, tables |
| **lineage** | the steps needed to rebuild a table's columns from raw input, stored beside it |
| **bundle** | the deployable artifact: lineage + fitted model, predicts from raw rows |

## Index

### Architecture

- [0001 — Own a constrained harness rather than serve other agents](0001-own-the-harness.md)
- [0002 — PydanticAI as the harness framework](0002-pydantic-ai.md)
- [0003 — Tools take file paths, not DataFrames](0003-file-based-tools.md)
- [0004 — The tools write the journal, not the agent](0004-tools-write-the-journal.md)
- [0020 — The read-only sandbox has no kernel; the run dir is the state](0020-the-sandbox-has-no-kernel.md)
- [0024 — The episode stop is structural, and the brief is not optional](0024-the-episode-stop-is-structural.md)
  *(accepted, not yet implemented; two supporting claims superseded by 0025)*
- [0025 — Episodes buy feasibility, not cost](0025-episodes-buy-feasibility-not-cost.md)
  *(accepted, not yet implemented)*
- [0026 — The brief is rendered, never accumulated](0026-the-brief-is-rendered-not-remembered.md)
  *(accepted and implemented; its central claim is superseded by 0027)*
- [0027 — The brief did not cut the orientation tax](0027-the-brief-did-not-cut-the-orientation-tax.md)
  *(a measurement: orientation is a fraction of the episode, not an information deficit)*

### Keeping features honest

- [0005 — Ordinal level order is a required argument, never inferred](0005-ordinal-order-is-required.md)
- [0006 — Feature lineage per table, not per run](0006-lineage-per-table.md)
- [0009 — The target column is invisible to agent-written code](0009-target-hidden-from-transforms.md)
- [0010 — Store the transform code, not the columns it produced](0010-store-code-not-columns.md)
- [0011 — Statelessness is tested as a property, not blacklisted](0011-stability-check.md)
- [0021 — The agent picks the target scale, from a fixed menu](0021-the-agent-picks-the-target-scale.md)
  *(the menu stands; where the choice is made is superseded by 0035)*
- [0032 — Fitted group aggregates, and the first step that is not replayed](0032-fitted-aggregates-and-a-step-that-is-not-replayed.md)
  *(what 0011 rejects becomes legal once the value is stored; and why a row
  filter is train-time-only)*

### Surviving a real agent

- [0007 — A bad argument is a correction, not the end of the run](0007-retryable-arguments.md)
- [0008 — One tool call at a time](0008-serialise-tool-calls.md)
  *(the danger is real; its mechanism was decorative — see 0031)*
- [0031 — The serialisation guard was decorative; correctness moves into the primitives](0031-the-guard-was-decorative.md)
- [0013 — A refusal is a return value, not an exception](0013-refusal-as-return-value.md)
- [0033 — A ceiling in the wrong unit is no ceiling; the search is bounded, the fit is not](0033-a-ceiling-in-the-wrong-unit-is-no-ceiling.md)
  *(what a self-tuning model may cost at scale; and a wall-clock bound that still
  cannot bound a single fold)*

### Knowing when to stop, and what worked

- [0012 — The harness decides when the run is over](0012-harness-owns-the-stop.md)
- [0014 — Exploration is cheap; tuning one table is not](0014-two-level-stop.md)
  *(its two levels collapse into one under 0035: there are no model experiments
  left for the per-table budget to refuse)*
- [0015 — Attribution gaps are reported, not prevented](0015-report-attribution-gaps.md)
- [0016 — Feature credit per tool call, not per column](0016-block-level-credit.md)
  *(superseded by 0017)*
- [0017 — Credit is measured by refitting without the columns, not by importance](0017-credit-by-ablation.md)
- [0037 — Residuals need folds; the ranking does not](0037-residuals-need-folds-the-ranking-does-not.md)
  *(why `explore_table` refits across folds while the ranking path deliberately
  does not, and why in-sample residuals are not the cheap version)*
- [0019 — The leaderboard is a public leaderboard](0019-the-leaderboard-is-a-public-leaderboard.md)
- [0022 — The yardstick must measure the quantity being ranked](0022-the-yardstick-must-measure-the-ranked-quantity.md)
  *(accepted and now implemented; one claim about what the fix achieves is corrected by 0028)*
- [0028 — `val_se` is implemented; the bar it is a quarter of is still open](0028-the-yardstick-is-fixed-the-bar-is-not.md)
  *(accepted; `min_gain_ratio` deliberately left open pending a second dataset)*
- [0023 — The third split has evidence now](0023-the-third-split-has-evidence-now.md)
  *(accepted; implemented by 0041 — its "`build_report` already takes a
  `holdout_score` parameter" was stale when written)*
- [0034 — Breaking a `val_se` tie with `cv_score` was tested and rejected](0034-the-tiebreak-was-tested-and-rejected.md)
  *(a measurement: 1 win, 1 loss, 1 null — and why the real conclusion is 0023)*
- [0035 — The search is over features; models are enumerated, not searched](0035-the-search-is-over-features.md)
  *(accepted and implemented; supersedes 0021 on where the target scale is
  chosen, and collapses 0014's two-level stop — its central mechanism is
  corrected by 0036, and 0038 is the first run under it)*
- [0036 — The pairing asymmetry was not there; the probe is justified by coverage](0036-the-pairing-asymmetry-was-not-there.md)
  *(a measurement: every comparison is paired, model or table alike — and a
  candidate yardstick for 0028's open `min_gain_ratio`)*
- [0038 — The first feature-only run: less optimism, and a prediction that failed](0038-the-first-feature-only-run.md)
  *(a measurement: optimism +0.01127 -> +0.00458 on a pre-registered A/B; why a
  val-cv gap is not a diagnostic for selection noise — its "optimism belongs to
  the table" hypothesis is rejected by 0039)*
- [0039 — Seguro v3: two defects, and 0038's hypothesis falsified](0039-seguro-v3-two-defects-and-a-falsified-hypothesis.md)
  *(the ruler must be on the terminal ballot; removing a tool removes the
  guarantees it carried; optimism attaches to the selected measurement — and its
  "holds by construction" claim is corrected by 0040)*
- [0040 — Seguro v4: a near-copy of the ruler is not the ruler](0040-seguro-v4-a-near-copy-of-the-ruler-is-not-the-ruler.md)
  *(the invariant needs the probe's fit, not its model type; and the
  selected-measurement pattern holds a second time, confound still open)*
- [0043 — On ordered data the seal costs the freshest rows, not a random tenth](0043-on-ordered-data-the-seal-costs-the-freshest-rows.md)
  *(corrects 0041's cost section: an ordered seal trades recency, not sample
  size — v7's shipped model forecasts 3-5 months past its own data)*
- [0042 — The terminal argmax shipped the worst model, and the seal caught it](0042-the-terminal-argmax-shipped-the-worst-model.md)
  *(a measurement: ranking one contiguous period is n=1 in the unit that varies;
  equal-weight blending beat every single model on sealed rows — registered for
  the next run, deliberately not implemented; implemented by 0044 once Kaggle
  agreed)*
- [0044 — What ships is the average of the terminal fits, not the best of them](0044-what-ships-is-the-average-not-the-argmax.md)
  *(0042 implemented: why an average pays no selection cost, why no candidate is
  filtered out, why the run still records what the argmax would have shipped —
  and the inversion-order defect found on the way; its cost section is corrected
  by 0045)*
- [0045 — The cost of the unfiltered blend was underestimated 44x](0045-the-cost-of-the-unfiltered-blend-was-underestimated-44x.md)
  *(a measurement: the blend still beats the argmax on Kaggle, but including the
  weakest member costs 0.0036 where the seal said 0.00008 — and why that licenses
  a hypothesis rather than a filter)*
- [0041 — The sealed split scores the ballot, not just the winner](0041-the-sealed-split-scores-the-ballot-not-the-winner.md)
  *(0023 implemented: opt-in per run, why the holdout comes out of the work half,
  why an ordered split seals the latest period — and why every terminal candidate
  is scored, which is 0039's confound removed without a Kaggle submission)*
- [0046 — The mean is a lossy summary of a group, and the loss was not sayable](0046-the-mean-is-a-lossy-summary-of-a-group.md)
  *(a category's target spread, middle and size alongside its average; why only
  the mean is smoothed, why min/max/sum are off the menu, and a leak test that
  passed while the leak was real)*
- [0047 — The forecast horizon is a column, not a constant](0047-the-forecast-horizon-is-a-column-not-a-constant.md)
  *(accepted, **not implemented**: what a lag serves past the end of stored
  history, why staleness is per row, and why predict rows never read their own
  batch — the two questions `../time_aware_features.md` named as blocking)*
- [0048 — A target lag is not target encoding; the guard it needs is the horizon](0048-a-target-lag-is-not-target-encoding.md)
  *(accepted, **not implemented**: why out-of-fold construction buys nothing
  against a strictly backward lag, what it should be replaced by, and why target
  history is refused outright under random k-fold — `../time_aware_features.md`
  §3.4, the last question left open before slice 2)*
- [0049 — The last observation is a sample of size one](0049-the-last-observation-is-a-sample-of-size-one.md)
  *(a measurement, and a negative one: frozen history made Rossmann's probe worse,
  92% of it because the single freeze date was a public holiday with 98% of stores
  shut — why `_age` says how old a value is and nothing says how typical, and why
  that makes windows the fix rather than a bonus)*
- [0050 — The anchor is what matters, not the history](0050-the-anchor-is-what-matters-not-the-history.md)
  *(Rossmann v8: the four best tables in the run are history tables and so is the
  worst, separated by one parameter — why a boundary-anchored value is a sample of
  size one shared by the whole batch, why the declared floor makes an offset legal
  while the block span makes it useful, and a hardcoded verdict that asserted its
  own premise)*
- [0051 — One word for one idea, and a flag instead of an age](0051-one-word-for-one-idea-and-a-flag-instead-of-an-age.md)
  *(why history's columns are all `_ago`, why the offset's age became a 1/0 `_exact`
  flag while the latest-known one stayed a number, why no name says "days", and why
  `fit_history` now reports the block span the agent had no way to compute)*
- [0052 — A window is a worse estimator of a better idea](0052-a-window-is-a-worse-estimator-of-a-better-idea.md)
  *(slice 3 built and measured negative: the trailing window loses to 0046's
  unbounded mean of the same key, the periodic key is the whole gain, a window
  longer than the shortest fold prefix is train/serve skew — and the coverage
  number added the same day rated the broken window higher than the good one)*
- [0053 — The fold count moves a window by 0.14, and we cannot say why](0053-the-fold-count-moves-a-window-and-we-cannot-say-why.md)
  *(**revises 0052's mechanism, leaves its conclusion standing**: the same window
  scores −0.093 at `n_splits=5` and +0.047 at `n_splits=3` with identical
  train/serve counts, so the skew story is refuted — what holds is that a
  window's measured worth is an artifact of the fold count while the unbounded
  aggregate it loses to is invariant, and no replacement mechanism is offered)*
- [0054 — The budget is not a constant, and at 200 trees a feature block scores backwards](0054-the-budget-is-not-a-constant-and-at-200-trees-features-score-backwards.md)
  *(**closes the common-budget question 0039 left open**: the row-count hypothesis
  is refuted — capacity pays where there is high-cardinality *identity* to learn,
  not where there are many rows — and on Rossmann the agent's own feature block
  measures −0.015 at the harness's 200 trees and +0.009 at 5000, so the ruler that
  ranks every table has been changing the sign of what it measures; early stopping
  asked for 162 to 16,926 rounds across five readings, which is why no constant is
  the fix)*
- [0055 — A rename in the library makes every earlier artifact unreplayable](0055-a-rename-makes-every-earlier-artifact-unreplayable.md)
  *(0051's column rename landed four hours after v8's tables were written, so their
  stored lineages now emit names the tables do not have and 0030 correctly refuses
  them — the boundary of "the code is the artifact", and why nothing in the suite
  could have caught it)*
- [0056 — Kaggle ranked four models, and the block we rank on got it nearly backwards](0056-kaggle-ranked-four-models-and-our-leaderboard-ranked-them-wrong.md)
  *(four submissions on an external seal: capacity confirmed at +0.079 against a
  predicted +0.085, history refuted at −0.030 on its fifth and first external
  reading, the encoding pipeline worth nothing once the budget is right — and the
  finding that matters, that `val` ranks at Spearman +0.20 against Kaggle while the
  sealed holdout ranks at +1.00, so the block we rank on is the wrong one)*
- [0057 — The capabilities the agent does not have](0057-the-capabilities-the-agent-does-not-have.md)
  *(the design inventory, nothing implemented: lead-time-restricted aggregates —
  found independently by a second dataset and by a winner's writeup — signed
  distance to a named event, per-group trend extrapolation justified by GBDTs not
  extrapolating at all, and what each would demand of the harness; plus what is
  rejected and why, and why none of it should be built before 0056's ruler is
  fixed)*

### Keeping the agent's surface small

- [0018 — The result payload is a budget, and so is the vocabulary](0018-the-payload-is-a-budget.md)

### Who decides what

- [0029 — The harness owns the exam; folds respect the split](0029-the-harness-owns-the-exam.md)
  *(the evaluation protocol is ours, the modelling is the agent's — and what
  follows for time-ordered data)*
- [0030 — A model that could never be served is refused, not scored](0030-a-model-that-cannot-be-served-is-not-trained.md)
  *(the serving schema, and the Rossmann run that argued for it)*

## Related reading

Longer-form notes that are not decisions:

- `docs/oof_vs_final_encoder.md` — why the encoder shipped in the bundle is
  fitted on all rows while the values trained on are out-of-fold, and why that
  is not leakage. This is the foundational idea the whole project rests on.
- `docs/target_encoding_leakage_insight.md` — who gets lied to by leakage, the
  model or the metric.
- `docs/feature_attribution_and_agent_iteration.md` — the long version behind
  decisions 0015 and 0016. Predates 0017: its importance-share reasoning was
  superseded, and it has not been rewritten.
- `docs/time_aware_features.md` — the gap v7 exposed: no feature can look at
  another row in time, because the harness can carry a mapping across the
  fit/predict seam but not a history. A plan with two blocking questions, not a
  decision.
- `docs/superpowers/specs/2026-07-21-tabular-agent-mcp-design.md` — the original
  design spec. Its title is stale (see 0001); revisions are dated in place.
- `notebooks/` — hands-on walkthroughs, one per slice of the system, each
  driving the machinery by hand before letting an LLM near it.
