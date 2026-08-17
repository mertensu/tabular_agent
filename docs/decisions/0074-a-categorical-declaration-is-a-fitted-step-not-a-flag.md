# 0074 — A categorical declaration is a fitted step, not a flag

- **Date:** 2026-08-17
- **Status:** proposed — **this is the design, written before the code** (0062's
  precedent). Nothing below is built.
- **Answers:** Ulf, after reading 0073's "what was rejected": *"i would like to let
  the agent decide if it chooses native categorial handling and let it declare a
  column as cateogirla if it wants to"*
- **Related:**
  [0073](0073-the-ruler-is-mispriced-on-one-dataset-and-three-mechanisms-failed-to-say-why.md)
  (which found that no model here is ever told a column is categorical, and why the
  probe cannot be the instrument that judges the fix),
  [0011](0011-stability-check.md) (the guard that refused `factorize`, and the
  reason this cannot be a flag),
  [0035](0035-the-search-is-over-features.md) (the search is over features and the
  model is held fixed — the objection this design has to survive),
  [0030](0030-a-model-that-cannot-be-served-is-not-trained.md) (servability, which
  the level set has to satisfy),
  [0018](0018-the-payload-is-a-budget.md) (every new tool is a cost on every run),
  [0062](0062-a-stream-is-a-second-input-and-its-observation-time-is-a-column.md)
  (the precedent for designing before building),
  `encoders.py:OrdinalEncoder` (the behavioural template).

## The request

The harness currently has three encodings — `frequency`, `ordinal`, `target` — and
all of them turn a category into a **number**. What it has never had is the fourth
option the literature treats as first-class for high cardinality: hand the column to
the booster **as a category** and let it find splits over partitions of the level
set. LightGBM takes `categorical_feature`, CatBoost takes `cat_features`, XGBoost
takes `enable_categorical`. This project passes none of them, anywhere.

0073 found this while investigating something else, and IEEE-CIS makes it concrete:
that competition *declares* 49 categorical columns, of which 31 are strings that
`_feature_frame` drops and 18 are numeric dtype that survive and get split as
ordered quantities.

## The correction that this record exists to record

The 18 are **not being handled wrongly.** An arbitrary integer code split as a
number is *label encoding*, a legitimate and widely used GBDT treatment — and
rsakata's comparison of encodings for GBDT reports that label encoding is notably
**robust** to large `num_leaves`, where native categorical handling overfits badly
at high cardinality. So the accidental treatment is the capacity-robust one.

> The defect is not that the treatment is bad. It is that **nobody chose it**, the
> agent cannot see it, and there is no way to say otherwise.

`ProductCD` is dropped because it happens to be a string; `card1` is label-encoded
because it happens to be an int. Two different fates decided by a dtype that came
out of a CSV parser.

## Why this cannot be a flag

The obvious implementation — mark the column, cast to pandas `category`, pass the
name to the booster — is refused by a rule this project already enforced once.

`probe_capacity.py` records it: an attempt to give Restaurant an integer store code
used `pd.factorize`, and **0011's stability check caught it**, because a positional
code depends on which rows are in the frame, so the same entity gets a different
number in a batch of one. Pandas `category` codes have the identical defect: the
codes are positions into `.cat.categories`, and a predict batch that contains 200 of
13,553 levels produces a different mapping under the same column name. Silent, and
wrong in the direction that looks fine.

> **So the level set is fitted state.** It is learned on the training table, stored
> in the lineage sidecar, and replayed at predict time — exactly as every encoder in
> `encoders.py` already does.

Which means the honest name for this capability is not "a flag" and not "a setting".
It is an **encoder that emits a category dtype instead of a number**.

## Why this does not reopen 0035

The objection Ulf's request has to survive: 0035 removed model choice from the agent
on purpose, so that the search is over *features* with the model held fixed. Letting
the agent turn on native categorical handling looks like handing back a model knob,
and would reintroduce the confound 0035 removed.

It does not, **provided the declaration is a property of the table rather than of the
fit.** The agent changes the table's types; the model layer applies one fixed rule —
*columns arriving as category dtype are declared to the booster* — identically to
every table it ever sees. No hyperparameter is exposed, no model is chosen, and the
probe and the terminal fit follow the same rule. The comparison "is t7 a better table
than t6" stays a comparison of tables.

This is the same argument 0063 used for the grain: the caller states a fact about the
data, and the library derives the behaviour from it.

## What gets built

1. **`CategoricalEncoder` in `encoders.py`.** Stores `{column: [levels]}` learned at
   fit, emits `pd.Categorical` with those levels in that order, `to_dict` /
   `from_dict` like its siblings. Behaviour copied from `OrdinalEncoder`, which is
   already the right template: **strict at fit, lenient at predict.** A level unseen
   at fit maps to a defined unseen bucket rather than to a fresh code.
2. **`fit_categorical` in `tools.py` + `agent.py`.** Adding a tool trips
   `test_harness_agent.py`'s pinned tool set on purpose (0018) — that failure is the
   guard working.
3. **`_feature_frame` keeps category dtype.** Today it keeps only
   `select_dtypes(include="number")`; it must keep `category` too, and must still
   drop bare strings, or the declaration means nothing.
4. **`_make_model` declares them per family.** One fixed rule, three
   implementations.
5. **`fit_probe` does the same**, or the ruler measures a different model than the
   one that ships, which is 0035's whole point and 0073's whole subject.

## The implementation trap, named in advance

**The three families do not agree about categoricals, and one of them will fail on
real data.** CatBoost's `cat_features` does not accept float NaN; LightGBM's
`categorical_feature` does; XGBoost needs `enable_categorical` together with a
histogram tree method. IEEE-CIS's declared categoricals are riddled with NaN —
`card2`, `addr1` and most of the `id_*` columns are float64 precisely because they
have missing values.

So a single "declare it and pass it on" rule will work for one family, throw for
another, and silently differ for the third. **The blend's members would then be
fitted on materially different representations of the same table**, which is a
confound inside the thing that ships. Whatever is built has to decide this
explicitly, and the decision belongs in a record rather than in a `try/except`.

## Open, and deliberately not decided here

- **What an unseen level becomes.** `OrdinalEncoder` maps to `-1`, below every
  declared level, so one split isolates it. The analogue for a category dtype is a
  reserved level. Cheap to get wrong at predict time and invisible when wrong.
- **Whether the agent can declare a column categorical that is currently numeric.**
  `card1` is an int; declaring it categorical is exactly the interesting case, and
  it is also the one where a mistaken declaration destroys a genuinely ordered
  column. Type inference cannot settle it, which is why 0051 forbids guessing.
- **Whether `frequency`/`ordinal`/`target` should be able to run *alongside* it** on
  the same column, which several strong Kaggle solutions do.

## What this will cost, and one warning that is already measured

0073 measured that this project's ruler overprices exactly the class of step that
supplies a model with structure it could otherwise reach by splitting. **Native
categorical handling is that class of step**, and rsakata's result — native handling
degrading as `num_leaves` grows while label encoding holds — says the interaction
with capacity is strong and runs the *other* way from a target encoding's.

> So the probe will have an opinion about `fit_categorical` steps, and 0073 says not
> to trust it. **Before this tool is put on the agent's surface, re-rule its steps at
> more than one budget** — `probe_ruler_budget_rank.py` already does this.

Adding a fourth way to handle a categorical also widens what the agent must choose
between from three to four, on every run, for every column (0018's argument), and
0053 measured that an existing free knob could not be searched usefully.

## Build order

1. `CategoricalEncoder` + tests.
2. `_feature_frame` keeps category dtype + tests.
3. `_make_model` per-family declaration, and the NaN decision — **its own record.**
4. `fit_probe`, then `fit_categorical` on the tool surface.
5. `predict_bundle` round-trip test with a batch containing a strict subset of the
   levels, and one containing an unseen level. **This is the test that matters**; it
   is the failure 0011 caught and the one a flag-shaped implementation would ship.
6. Re-rule the new step type at two budgets before trusting any score it produces.
7. Notebook + quiz.
