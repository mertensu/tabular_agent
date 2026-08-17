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

It also means the index has to do a job the records cannot. Because nothing here
is ever revised, **the only place that can say which reasoning is still standing
is this file** — so *Where each thread stands*, below, names the record that
currently holds each question and the ones it was built on top of. Read that
first and the reading order for everything else follows from it. Read the index
straight through in number order and you will read refuted mechanisms as though
they were current, which is the one failure mode this corpus has.

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

## Where each thread stands

**Routing, not summary.** One row per live question: the record that currently
holds it, and the ones it rests on. The *rests on* column is a reading order, not
a bibliography — the earlier records are where the mechanisms that failed are
written down, and several of them were refuted by the record in the middle
column.

Nothing in this table edits a record, and no record is authoritative about its
own successors. If this table and a record disagree, the record is what was
believed at the time and this table is what is believed now.

| thread | where it stands | rests on |
|---|---|---|
| **How a table is ranked** — the ruler | [0073](0073-the-ruler-is-mispriced-on-one-dataset-and-three-mechanisms-failed-to-say-why.md): the probe inflates most feature steps on a high-cardinality dataset, and no fixed budget is neutral because each table has its own optimum | 0022, 0028, 0034, 0035, 0036, 0038, 0039, 0040, 0054, 0056, 0058, 0059 |
| **Model capacity** | [0068](0068-capacity-reaches-the-shipped-blend-at-half-strength-and-validation-picks-the-wrong-budget.md): capacity reaches the shipped blend at half the probe's strength, and `val` picks the budget wrongly on both datasets | 0054, 0067 |
| **Where the search happens** | [0035](0035-the-search-is-over-features.md): the search is over features; models are enumerated, not searched | 0014, 0021, 0022 |
| **Feature credit** | [0017](0017-credit-by-ablation.md): credit is measured by refitting without the columns | 0015, 0016, 0037 |
| **The seal, and what an internal score is worth** | [0066](0066-the-collapsed-column-wins-everywhere-except-where-it-is-graded.md): a sealed score is a lower bound on a *difference* too, and can order two candidates backwards when the defect is in the data's provenance | 0019, 0023, 0041, 0043, 0060, 0061, 0065 |
| **What ships at the end of a run** | [0044](0044-what-ships-is-the-average-not-the-argmax.md): the equal-weight average of the terminal fits, not the argmax — with its cost corrected by [0045](0045-the-cost-of-the-unfiltered-blend-was-underestimated-44x.md) | 0042, 0045 |
| **Time-aware features** | [0072](0072-a-feature-is-an-average-over-a-stretch-not-the-value-of-one-row.md): `fit_history` emits trailing windows and nothing else — no last-known pair, no lags | 0047, 0048, 0049, 0050, 0051, 0052, 0053, 0071 |
| **Second inputs (streams)** | [0063](0063-the-grain-is-measured-from-the-table-not-required-of-the-caller.md): the grain is measured from the table, not required of the caller — and [0064](0064-month-ahead-is-the-same-idea-and-does-not-get-the-same-tool.md) refuses the single-input sibling | 0057, 0062, 0064 |
| **What the winner had that we do not** | [0070](0070-what-the-winner-had-that-we-cannot-express.md), narrowed by [0071](0071-the-gap-was-not-the-vocabulary-it-was-a-column-nobody-asked-for.md): the distance is **not** the feature vocabulary | 0057, 0070, 0071 |
| **Categoricals** | [0074](0074-a-categorical-declaration-is-a-fitted-step-not-a-flag.md) — **designed, not built**; [0075](0075-every-declared-categorical-it-could-see-it-used-and-every-one-it-could-not-it-ignored.md) splits its premise: the declaration channel works where it reaches the agent, and a second defect decides whether it reaches | 0011, 0073, 0074 |
| **What a fitted aggregate can reach** | [0076](0076-a-fitted-aggregate-loses-coverage-exactly-as-its-key-becomes-worth-having.md): coverage collapses as the key sharpens, and 0011 refuses leakage and transduction under one rule — **no fix proposed** | 0011, 0032, 0030 |

**Accepted and not built.** These are plans living in an archive, which is worth
seeing as a list rather than discovering one record at a time:

- [0024](0024-the-episode-stop-is-structural.md), [0025](0025-episodes-buy-feasibility-not-cost.md) — the episode stop. Recorded as unbuilt roughly fifty records ago.
- [0047](0047-the-forecast-horizon-is-a-column-not-a-constant.md), [0048](0048-a-target-lag-is-not-target-encoding.md) — both say `not implemented` in their own headers, and **that is now stale in one direction**: `TaskSpec.forecast_horizon` exists and is load-bearing (it gates window lengths in `encoders.py` and stream leads in `tools.py`). What did *not* land is the lag half, which 0072 went on to refuse outright. Checked against the code, 2026-08-17; the headers are left alone per the never-edit rule.
- [0074](0074-a-categorical-declaration-is-a-fitted-step-not-a-flag.md) — proposed, with its own build order.

**A note on `Status:`.** The header field is free text and has accumulated about
twenty distinct spellings across 74 records (`accepted`, `accepted, implemented`,
`accepted (a measurement)`, `built, measured, and the measurement is negative`,
…). So the corpus cannot answer "is this a rule I must honour, a measurement, or
an unbuilt plan" without being read. This table is the workaround; a fixed
vocabulary for *new* records would be the fix, and is not yet decided.

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

### Who decides what

- [0029 — The harness owns the exam; folds respect the split](0029-the-harness-owns-the-exam.md)
  *(the evaluation protocol is ours, the modelling is the agent's — and what
  follows for time-ordered data)*
- [0030 — A model that could never be served is refused, not scored](0030-a-model-that-cannot-be-served-is-not-trained.md)
  *(the serving schema, and the Rossmann run that argued for it)*

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
- [0055 — A rename in the library makes every earlier artifact unreplayable](0055-a-rename-makes-every-earlier-artifact-unreplayable.md)
  *(0051's column rename landed four hours after v8's tables were written, so their
  stored lineages now emit names the tables do not have and 0030 correctly refuses
  them — the boundary of "the code is the artifact", and why nothing in the suite
  could have caught it)*

### Surviving a real agent

- [0007 — A bad argument is a correction, not the end of the run](0007-retryable-arguments.md)
- [0008 — One tool call at a time](0008-serialise-tool-calls.md)
  *(the danger is real; its mechanism was decorative — see 0031)*
- [0013 — A refusal is a return value, not an exception](0013-refusal-as-return-value.md)
- [0031 — The serialisation guard was decorative; correctness moves into the primitives](0031-the-guard-was-decorative.md)
- [0033 — A ceiling in the wrong unit is no ceiling; the search is bounded, the fit is not](0033-a-ceiling-in-the-wrong-unit-is-no-ceiling.md)
  *(what a self-tuning model may cost at scale; and a wall-clock bound that still
  cannot bound a single fold)*

### Keeping the agent's surface small

- [0018 — The result payload is a budget, and so is the vocabulary](0018-the-payload-is-a-budget.md)

### Knowing when to stop, and crediting what worked

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

### The ruler: how a table is ranked

The project's central subject since about 0054, and the thread with the most
refuted mechanisms in it. Live conclusion: **0073**.

- [0022 — The yardstick must measure the quantity being ranked](0022-the-yardstick-must-measure-the-ranked-quantity.md)
  *(accepted and now implemented; one claim about what the fix achieves is corrected by 0028)*
- [0028 — `val_se` is implemented; the bar it is a quarter of is still open](0028-the-yardstick-is-fixed-the-bar-is-not.md)
  *(accepted; `min_gain_ratio` deliberately left open pending a second dataset)*
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
- [0054 — The budget is not a constant, and at 200 trees a feature block scores backwards](0054-the-budget-is-not-a-constant-and-at-200-trees-features-score-backwards.md)
  *(**closes the common-budget question 0039 left open**: the row-count hypothesis
  is refuted — capacity pays where there is high-cardinality *identity* to learn,
  not where there are many rows — and on Rossmann the agent's own feature block
  measures −0.015 at the harness's 200 trees and +0.009 at 5000, so the ruler that
  ranks every table has been changing the sign of what it measures; early stopping
  asked for 162 to 16,926 rounds across five readings, which is why no constant is
  the fix)*
- [0056 — Kaggle ranked four models, and the block we rank on got it nearly backwards](0056-kaggle-ranked-four-models-and-our-leaderboard-ranked-them-wrong.md)
  *(four submissions on an external seal: capacity confirmed at +0.079 against a
  predicted +0.085, history refuted at −0.030 on its fifth and first external
  reading, the encoding pipeline worth nothing once the budget is right — and the
  finding that matters, that `val` ranks at Spearman +0.20 against Kaggle while the
  sealed holdout ranks at +1.00, so the block we rank on is the wrong one)*
- [0058 — The ranking failure was one feature, not the block](0058-the-ranking-failure-was-one-feature-not-the-block.md)
  *(**revises 0056's mechanism and 0057's build order, leaves their measurements
  and inventory standing**: Ulf asked whether the ranking finding was
  dataset-specific and it was — drop the one history arm and `val` ranks fine, the
  arm decayed with distance from its freeze date exactly as 0049/0050 said, `val`'s
  block is the odd one on composition rather than on recency, and n=4 gives a
  perfect ordering 1 time in 24 anyway; what survives is a composition diagnostic
  and ranking stability across blocks, neither of which changes which block ranks)*
- [0059 — The interval belongs to the difference, not the score](0059-the-interval-belongs-to-the-difference-not-the-score.md)
  *(builds 0036's unbuilt instrument and 0058's block diagnostic: a run now reports
  an interval around `probe_vs_source_table` — a paired bootstrap over rows on a
  random split, the difference recomputed across consecutive blocks on an ordered
  one, the mode chosen by passing the ordering column rather than a flag because
  the validation half is **not** stored in its own order. Measured against the
  history arm Kaggle refuted: the row bootstrap called a difference that reverses
  externally 27 standard errors from zero, blocking widened it 16x and still called
  it stable over 44 days, and only a 90-day window exposed the decay — so the
  instrument's power is **window length, not block count**. 0036's pairing ratio
  replicates on a second dataset; `min_gain_ratio` is deliberately still not gated
  on any of it)*
- [0067 — Capacity is worth a tenth of that on a table that has features](0067-capacity-is-worth-a-tenth-of-that-on-a-table-that-has-features.md)
  *(0061's "untested cell", finally measured: v7's `t13` — the chain that actually
  shipped — swept at the same budgets 0054 used. **+0.0068**, against +0.0606 on a
  bare table and +0.0812 on a feature block. Capacity and feature work
  **substitute**: whichever you apply second is worth a tenth of what it was worth
  first. Narrows 0054 without contradicting a number in it, and refutes the
  priority argument built on it — capacity is 0.8x the gap to the private median,
  not the 3-7x the bare-table readings suggested. **Features and capacity are both
  bounded now and together they do not close the gap.** Also: the inner-tail
  early-stopping arm under-shoots badly here (1195 rounds, +0.0028), which is
  evidence about 0054's open item 2 rather than an answer to it)*
- [0068 — Capacity reaches the shipped blend at half strength, and validation picks the wrong budget](0068-capacity-reaches-the-shipped-blend-at-half-strength-and-validation-picks-the-wrong-budget.md)
  *(0067's knob, measured on what actually ships — the terminal blend on the sealed
  holdout, not one probe lgbm on validation. Rossmann **+0.0037** against the
  probe's +0.0068; Restaurant nothing. Capacity closes ~44% of Rossmann's gap, not
  80%. **The finding that outlives the numbers: `val` and the sealed rows disagree
  about which budget is best on Rossmann and disagree in sign on Restaurant**, so
  every design for 0054's open item 2 that lets validation choose is refuted before
  being built. The three GBDT families also disagree with each other, consistently
  across both datasets, so one common budget is the wrong shape)*
- [0073 — The ruler is mispriced on one dataset, and three mechanisms failed to say why](0073-the-ruler-is-mispriced-on-one-dataset-and-three-mechanisms-failed-to-say-why.md)
  *(re-ruled Rossmann v7's whole chain at five budgets and every other run in
  `_runs/` at two. **The probe inflates most feature steps, not just close calls:**
  v7's chain is worth 0.104 as the ruler tells it and 0.032 at capacity, and the
  argmax moves off the table that shipped. Two clean mechanisms — a row filter's
  cost is training data a small ruler cannot use, and v7's largest fork reverses so
  that the branch it rejected wins at every high budget. Replicates on `rossmann_v8`
  and appears on **none** of the other three datasets. Two beliefs died: rounds and
  learning rate are not interchangeable at matched shrinkage, and more capacity is
  not truer — three tables overfit between 3000 and 5000 rounds, so **each table has
  its own optimum and no fixed-budget ruler is neutral.** Three proposed mechanisms
  all failed, and the two-budget panel is **deliberately not built**: ~10x cost on
  every dataset for an effect seen on one. Also records that no model here has ever
  been told a column is categorical)*

### The seal, and the outside world

What an internal score is worth once something external grades it. Live
conclusion: **0066**.

- [0019 — The leaderboard is a public leaderboard](0019-the-leaderboard-is-a-public-leaderboard.md)
- [0023 — The third split has evidence now](0023-the-third-split-has-evidence-now.md)
  *(accepted; implemented by 0041 — its "`build_report` already takes a
  `holdout_score` parameter" was stale when written)*
- [0041 — The sealed split scores the ballot, not just the winner](0041-the-sealed-split-scores-the-ballot-not-the-winner.md)
  *(0023 implemented: opt-in per run, why the holdout comes out of the work half,
  why an ordered split seals the latest period — and why every terminal candidate
  is scored, which is 0039's confound removed without a Kaggle submission)*
- [0042 — The terminal argmax shipped the worst model, and the seal caught it](0042-the-terminal-argmax-shipped-the-worst-model.md)
  *(a measurement: ranking one contiguous period is n=1 in the unit that varies;
  equal-weight blending beat every single model on sealed rows — registered for
  the next run, deliberately not implemented; implemented by 0044 once Kaggle
  agreed)*
- [0043 — On ordered data the seal costs the freshest rows, not a random tenth](0043-on-ordered-data-the-seal-costs-the-freshest-rows.md)
  *(corrects 0041's cost section: an ordered seal trades recency, not sample
  size — v7's shipped model forecasts 3-5 months past its own data)*
- [0044 — What ships is the average of the terminal fits, not the best of them](0044-what-ships-is-the-average-not-the-argmax.md)
  *(0042 implemented: why an average pays no selection cost, why no candidate is
  filtered out, why the run still records what the argmax would have shipped —
  and the inversion-order defect found on the way; its cost section is corrected
  by 0045)*
- [0045 — The cost of the unfiltered blend was underestimated 44x](0045-the-cost-of-the-unfiltered-blend-was-underestimated-44x.md)
  *(a measurement: the blend still beats the argmax on Kaggle, but including the
  weakest member costs 0.0036 where the seal said 0.00008 — and why that licenses
  a hypothesis rather than a filter)*
- [0060 — The sealed block is optimistic at a horizon it cannot reach](0060-the-seal-is-optimistic-at-a-horizon-it-cannot-reach.md)
  *(the first agent run on Recruit Restaurant, scored externally: validation was
  **pessimistic** about the sealed block and the sealed block was **optimistic by
  0.042** about Kaggle's private half — four windows in time order, each a fair
  reading of itself and each flattering the next, so a sealed score is a **lower
  bound** on the error at the graded horizon rather than an estimate of it. Confirms
  0059's stated limitation on a second dataset. Also: 6 of 17 comparisons came back
  `sign_stable: False` — the block mode's first live firing — and the agent stopped
  early **citing it**, which is advisory information changing behaviour where 0027
  recorded three failures. And a mechanism refuted by having both boards: our
  horizon decay is **smaller** than the field's average, so it is a property of the
  task and says nothing about the reservation skew it was predicted to expose)*
- [0061 — We were measuring against the wrong board](0061-we-were-measuring-against-the-wrong-board.md)
  *(every Rossmann script hardcoded the **public** median 0.11677 while quoting
  **private** scores beside it. The private median is **0.12967**, so our best
  submission is **+0.0085** from median and top 58%, not the +0.021 the project has
  been repeating — a mechanism worth 0.01 now clears the bar where nothing under
  0.021 did. Found only because 0060 made the same mistake on Restaurant an hour
  earlier. Also, on **both** datasets the field degrades more than we do from the
  public to the private half of the test block (+0.0147 vs our +0.0088; +0.0446 vs
  our +0.0310) — cause unresolved, and regression to the mean is not ruled out.
  Corrects a reference line, no measurement)*
- [0065 — The agent found the leak, built the honest table, and the ruler overruled it](0065-the-agent-found-the-leak-and-the-ruler-overruled-it.md)
  *(**the first run with a declared stream**, and it is v1 with exactly one change.
  The agent reached both streams, chose a ladder of leads (7/14/30 — the
  registered prediction of "at the declared horizon" was wrong), and diagnosed the
  cut on its own off `column_notes`, building the table without the collapsed
  columns. **The probe scored that honest table 0.0085 worse — 2.9 block sd, every
  block agreeing — and `best_table` is the argmax, so the leaky table shipped.**
  Adding honest columns helped (+0.005); removing dishonest ones was punished.
  Supplying the capability is necessary and not sufficient: no block of these rows
  can measure the difference, because every one shares the cut. 0059's interval was
  confident and wrong for the second time. No fix chosen — three shapes named,
  spanning 0030, 0035 and 0041/0043)*
- [0066 — The collapsed column wins everywhere except where it is graded](0066-the-collapsed-column-wins-everywhere-except-where-it-is-graded.md)
  *(**the controlled test**, and the measurement this dataset was kept for since
  0054. t7 against t8, one variable — six collapsed reservation columns — on
  identical rows, both arms shipped through the real `finalize_run` path, with the
  t7 arm reproducing v2 to six decimals internally and digit-for-digit on Kaggle.
  **t8 − t7 reads +0.0085 (probe), +0.0112 (val), +0.0112 (sealed), +0.0023
  (public), −0.0015 (private)** — the collapsed column's advantage decaying
  monotonically with distance from training and reversing at the far half. The
  gradient is the evidence, not the flip. Refutes my own registered prediction and
  confirms 0065. **All three internal instruments were confidently wrong in the
  same direction, the sealed holdout worst of them** — so a sealed score is a lower
  bound on a *difference* too, and can order two candidates backwards when the
  defect is in the data's provenance. Magnitude is twelve ranks; the finding is
  direction. The board is now spent for this question)*

### Time-aware features and second inputs

Twelve records, five negative measurements and one external refutation. Live
conclusion: **0072** for history, **0063** for streams.

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
- [0062 — A stream is a second input, and its observation time is a column](0062-a-stream-is-a-second-input-and-its-observation-time-is-a-column.md)
  *(**accepted, being built** — the record 0057 item 1 asks for. A related file at
  a finer grain whose records carry both the time they are *about* and the time
  they became *knowable*: collapsing it to a column discards the second, and 36% of
  this dataset's reservations were booked with zero days' notice, so the column is
  systematically larger in training than at serving and no guard here sees it. The
  encoder stores the **recipe, not the numbers** (0010) and the stream is handed in
  again at predict; materialising it was rejected because it only works while the
  file already contains the future, which is 0050's frozen anchor. Also narrows
  0057's "found twice" claim — the principle was, the second-input plumbing was
  not)*
- [0063 — The grain is measured from the table, not required of the caller](0063-the-grain-is-measured-from-the-table-not-required-of-the-caller.md)
  *(settles a question 0062 left open. A stream is stamped finer than the table it
  feeds, so something must say which row an 18:00 booking belongs to. The first
  build left that to `prep_*.py`; Ulf rejected it — a rule every user must obey
  that the library never states, checks or records is a trap, and getting it wrong
  yields a column of zeros rather than an error. Instead `fit` **measures** the
  table's commonest row spacing, stores it as a width and a phase, and floors both
  sides onto it; the lead keeps full resolution. Believed only if every ordering
  value sits on that grid — otherwise it refuses and asks for `event_grain` to be
  declared. Why that is a measurement and not 0051's inference: it can be, and is,
  checked against the data it came from)*
- [0064 — "Month ahead" is the same idea and does not get the same tool](0064-month-ahead-is-the-same-idea-and-does-not-get-the-same-tool.md)
  *(**a decision not to build**. Ulf asked why the new tool is described as being
  for a second input when the Rossmann winner's month-ahead models have only one —
  and he is right: 0057 item 1 is two capabilities in one sentence, and only the
  bitemporal-stream half is built. Corrects 0062's claim that `fit_history`
  "substantially does it already": true in shape, false in scale, since the gap
  between a row and its history there comes from the fold geometry rather than a
  declaration — 0049 measured training ages 1–305 against validation ages 1–45.
  Refused anyway: found once not twice, on a feature family with five negative
  readings and one external refutation, and 0053 showed its *existing* knob cannot
  be searched, so a second one enlarges only what the agent must guess. The real
  objection is that the asymmetry belongs to the evaluation geometry (0041/0043),
  and a per-feature knob lets the agent work around the exam instead of fixing it.
  The discriminating test against the stream: there, no combination of existing
  tools could express the honest feature at all)*
- [0071 — The gap was not the vocabulary; it was a column pair nobody asked for](0071-the-gap-was-not-the-vocabulary-it-was-a-column-nobody-asked-for.md)
  *(0070's recommendation carried out: the winner's salesmodel rebuilt with real
  tool calls and measured. **Capability gap +0.0012, agent gap +0.0022, and
  `fit_history`'s unrequested `_latest` pair +0.0143** — the tool defect is twelve
  times the capability gap, at 5/5 blocks and 2.5 block sd, the first solid reading
  in three sessions. So the distance to the winner is not the feature vocabulary.
  Two unlooked-for findings: **the stability check tests row-locality, not
  batch-independence**, and misses any column that is a function of the set of
  distinct keys; and `Customers` is dropped in every run because 0030's servability
  rule makes dropping it the rational first move, which destroys the only thing it
  was good for. No library change made — n=1)*
- [0072 — A feature is an average over a stretch, not the value of one row](0072-a-feature-is-an-average-over-a-stretch-not-the-value-of-one-row.md)
  *(the rule that came out of 0071, and it is wider than the measurement was:
  **`fit_history` emits trailing windows and nothing else** — no last-known pair,
  no lags. The mechanism is that the per-fold freeze is **correct** and therefore
  silent: a stored single observation varies across four thousand values in
  training and is one frozen per-group constant at serving, so the model is fitted
  on one column and served another of the same name. Priced at +0.0143 on Rossmann
  (where the one draw fell on a public holiday) and +0.0022 probe / +0.0035 sealed
  on Restaurant, where it is `sign_stable: False` — the defect transfers, the price
  is a lottery. Supersedes 0049's and 0050's conclusion that pairing a value with
  its age makes it honest enough to keep. And the offset half turns up a sharper
  rule than the one it was built on: a 364-day lag and a 364-day window score
  0.829 against 0.818 on a coarse key — the **window is worse** — and 0.829
  against 0.950 once the weekday and promo move into the key. So it is not that
  averages beat points; it is that **comparability belongs in the key**, and a lag
  is a key expressed as a distance. The machinery stays
  in the class so pre-0072 bundles still predict; the refusal is pinned as a
  signature)*

### What the agent can build with

The feature vocabulary itself — what a tool can express, and what it costs to add
one. See also 0032 under *Keeping features honest*.

- [0046 — The mean is a lossy summary of a group, and the loss was not sayable](0046-the-mean-is-a-lossy-summary-of-a-group.md)
  *(a category's target spread, middle and size alongside its average; why only
  the mean is smoothed, why min/max/sum are off the menu, and a leak test that
  passed while the leak was real)*
- [0069 — Feature selection pays on columns that mislead, not on columns that are idle](0069-feature-selection-pays-on-columns-that-mislead-not-on-columns-that-are-idle.md)
  *(the first tool that makes a table **narrower**, built from the Rossmann
  winner's report and measured at ~zero on three tables. Two of my designs were
  wrong and both were caught by measurement: a selection block cut as a *fraction*
  is six months where the run is graded over 45 days (0043/0060 again), and ranking
  subsets individually is an argmax whose margin is thinner than the block's own
  noise (0023 one level down). The finding is why: **a boosted tree loses almost
  nothing to a useless column, so removing one wins almost nothing back** — the
  winner's problem was columns that fit the training period and do not transfer.
  Built, tested, and deliberately **kept off the agent's tool surface**)*
- [0074 — A categorical declaration is a fitted step, not a flag](0074-a-categorical-declaration-is-a-fitted-step-not-a-flag.md)
  *(**proposed, the design before the code.** Ulf asked to let the agent declare a
  column categorical and choose native handling — the fourth option the literature
  treats as first-class for high cardinality, which this project has never had. It
  cannot be a flag: pandas `category` codes are positions into the observed level
  set, so a predict batch of 200 levels out of 13,553 remaps them silently — the
  same defect 0011 caught when `factorize` was tried. So the level set is **fitted
  state**, stored in the sidecar and replayed, and the capability is an encoder that
  emits a category dtype instead of a number. That framing also keeps 0035 intact:
  the agent changes the table's types, the model layer applies one fixed rule, no
  hyperparameter is exposed. Names the trap in advance — the three families disagree
  about NaN in categoricals, so a naive rule would fit the blend's members on
  different representations of the same table. And 0073 warns that the probe will
  have an opinion about these steps that should not be trusted at one budget)*
- [0076 — A fitted aggregate loses its coverage exactly as its key becomes worth having](0076-a-fitted-aggregate-loses-coverage-exactly-as-its-key-becomes-worth-having.md)
  *(what the IEEE-CIS winners had that this harness cannot express — and it is not
  vocabulary. Their whole solution is one key, `card1_addr1_D1n`, which a run could
  build today; what it cannot do is compute the aggregates over train **and** test
  the way they do. Measured on our own files: a `card1`-keyed aggregate covers
  **98.2%** of test rows, `card1_addr1` covers 81.7%, and their uid covers
  **21.0%** — so **79% of rows would receive `default_`, and coverage collapses in
  exactly the direction that makes the key worth having.** The unlooked-for
  finding: the sealed holdout reports that same key as **43.0%** covered, twice its
  real reach — 0060's each-window-flatters-the-next arriving in *coverage* rather
  than in score. The argument: **0011 refuses leakage and transduction under one
  rule and they are not the same thing** — a group mean over an unlabelled predict
  batch reads no target. **No fix is proposed**, on three load-bearing grounds
  (single-row serving genuinely breaks, the bundle contract is row-wise, and 0073
  says the probe cannot price this at one budget). One small thing is recommended
  without reservation: `fit_group_stats` should report its coverage, since a run
  today cannot see a number that moves from 98% to 21%)*

### Runs on the outside world

Whole-run records: what an agent did with a real dataset, measured against
predictions registered before it started. See also 0038, 0039, 0060, 0065 in the
sections above.

- [0075 — Every declared categorical it could see, it used; every one it could not, it ignored](0075-every-declared-categorical-it-could-see-it-used-and-every-one-it-could-not-it-ignored.md)
  *(the first IEEE-CIS run, at unchanged defaults, and **two of six registered
  predictions refuted**. The 49 columns the competition declares categorical split
  49-for-49 on **visibility, not dtype**: all 31 strings were encoded because
  `non_numeric_columns` is not truncated, the 6 numeric ones inside the profile cap
  were encoded, and the 12 numeric ones past column 100 — invisible in both
  channels — were touched by nothing. So a bare name in a refusal is a working
  channel, and 0074's premise holds only where no channel reaches. Replicates
  0071's conclusion on a second competition: the agent rebuilt four of the winners'
  feature families unaided. The ruler could not resolve its own steps — **18 of 20
  comparisons `sign_stable: False`, 9 of them accepted as improvements** — and the
  three model families disagreed by 0.025, more than the whole feature chain was
  worth. The seal came back **easier** than validation on all four candidates,
  against 0060, and the mechanism is left unexplained rather than guessed. The blend
  beat every member on both blocks (0044, fifth dataset, first on a
  higher-is-better metric). And a terminal fit — `elasticnet` — was attempted, ran
  at least 45 of the run's 93 minutes, was discarded, and **left no trace anywhere
  in the journal**)*

### What the winners had

- [0057 — The capabilities the agent does not have](0057-the-capabilities-the-agent-does-not-have.md)
  *(the design inventory, nothing implemented: lead-time-restricted aggregates —
  found independently by a second dataset and by a winner's writeup — signed
  distance to a named event, per-group trend extrapolation justified by GBDTs not
  extrapolating at all, and what each would demand of the harness; plus what is
  rejected and why, and why none of it should be built before 0056's ruler is
  fixed — **the last of which 0058 withdraws**)*
- [0070 — What the winner had that we cannot express](0070-what-the-winner-had-that-we-cannot-express.md)
  *(the distillation `HANDOFF.md` had carried as a todo for two weeks, done against
  the tool vocabulary. **One of his un-ensembled models scores ~0.105 private, rank
  ~3 of 3,300; his entire ensembling apparatus was worth 0.005.** Four capabilities
  missing — and the record deliberately **refuses to rank them**, because two
  attempts to rank a lever from a description failed the same day. His feature dict
  puts the trend features outside his main model, and puts **no store identity** in
  it either, which is in tension with 0054. Recommends measuring instead: rebuild
  his salesmodel by hand in `scripts/` and find out whether the gap is vocabulary at
  all. Also: we can already express a per-store aggregate of a train-only column and
  throw it away in every Rossmann run. Vindicates 0064's refusal; leaves 0001 and
  the 0.985 multiplier out)*

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
