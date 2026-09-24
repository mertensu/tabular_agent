# 0076 — A fitted aggregate loses its coverage exactly as its key becomes worth having

- **Date:** 2026-08-17
- **Status:** accepted (a measurement). **No fix is proposed and none should be
  read into this**: the tension it names is real, the refusal it examines is
  probably still correct, and the record exists so that the cost of that refusal
  is written down instead of assumed to be zero.
- **Answers:** what the IEEE-CIS 1st-place solution does that this harness cannot,
  after the run showed the agent reaching the same feature *families* on the same
  columns (see 0075)
- **Related:**
  [0011](../../../docs/decisions/0011-stability-check.md) (the guard that refuses this, and the rule this
  record says is doing two jobs),
  [0032](0032-fitted-aggregates-and-a-step-that-is-not-replayed.md)
  (`GroupStatsEncoder`: fitted once, stored, replayed — the design under
  examination),
  [0030](../../../docs/decisions/0030-a-model-that-cannot-be-served-is-not-trained.md) (servability, which
  is why the refusal exists at all),
  [0003](0003-file-based-tools.md) (tools take a table, and a bundle predicts from
  raw rows — the contract a batch-dependent feature breaks),
  [0060](0060-the-seal-is-optimistic-at-a-horizon-it-cannot-reach.md) (each window
  flattering the next; this record finds the same shape in *coverage* rather than
  in score),
  [0073](0073-the-ruler-is-mispriced-on-one-dataset-and-three-mechanisms-failed-to-say-why.md)
  (why the probe must not be the instrument that prices this),
  `fraud_detection/winner.md`, `fraud_detection/xgb-fraud-with-magic-0-9600.ipynb`,
  `scripts/prep_ieee.py`.

## Where the question came from

The IEEE-CIS winners describe their solution as one idea, which they call the
magic feature. It is a key:

```python
uid = card1_addr1 + '_' + floor(TransactionDT/86400 - D1)
```

`D1` is documented as "days since this credit card began", so `day − D1` is the
day the card was issued, and the triple is a stable client identifier. They then
build ~47 group aggregates keyed on it — mean, std, nunique, count of other
columns within the group — and **delete the key**, because 68.2% of the clients
in the private test set never appear in training and a key used directly would be
useless for them.

Our first IEEE run reached the same aggregates on the same columns, keyed on
`card1` and on a `card1`/`addr1` combination it built itself (0075). So the
question is not vocabulary. It is: **why can this harness not have the better
key?**

## The key is not the problem

Every ingredient exists. `transform_features` does row-local arithmetic over the
table, so `day − D1` is legal and passes 0011's stability check — it reads only
the row's own values. String concatenation into a composite key is legal, and the
run proves it: the agent built `c1_addr` in `t10` without being asked.
`fit_group_stats` then takes any column as a key.

A run could build `uid` today. **Nothing in the tool vocabulary is missing.**

## What is missing is one line of the winner's code, and it is deliberate

Their aggregates are computed over the concatenation of both files:

```python
temp_df = pd.concat([train_df[[col, main_column]], test_df[[col, main_column]]])
temp_df = temp_df.groupby([col])[main_column].agg([agg_type])
```

Ours cannot be. `GroupStatsEncoder` learns the per-group value at fit and stores
it, and its own docstring states the reason in exactly these terms:

> *"a `groupby(...).transform("mean")` written in `transform_features` recomputes
> the mean from whatever rows are in the frame. That is batch-dependent by
> construction: one row alone gets its own value back, and 0011's stability check
> rejects it — correctly. What makes it legal here is that the group's number is
> **fitted once and stored**, so a single row maps to the same value a million
> rows would."*

And:

> *"A group absent from the fit gets `default_`."*

That is the whole difference. For an unseen client, their aggregate is computed
from that client's own test-set transactions; ours is a constant.

## The measurement

`scripts/` was not needed for this; it is a coverage count on the two files this
project already prepared. **What fraction of test rows fall into a group the
training half has actually seen?**

| key | groups in train | test rows covered | test rows given `default_` |
|---|---|---|---|
| `card1` | 13,553 | **98.2%** | 1.8% |
| `card1_addr1` | 37,531 | **81.7%** | 18.3% |
| `card1_addr1_D1n` (the winner's uid) | 199,070 | **21.0%** | **79.0%** |

> **Coverage collapses in exactly the direction that makes the key worth having.**
> A coarse key is well covered and says little about a row; a key precise enough
> to identify a client is 79% unanswerable. The precision is the value and the
> precision is the cost, and they are the same property.

This is the general statement, and it names no dataset: **a fitted group
aggregate is only as useful as the overlap between the groups at fit time and the
groups at serve time, and that overlap shrinks as the key sharpens.** Any panel
dataset whose entities turn over has it. Nothing in the harness reports it.

## The second finding, which nobody was looking for

The same count, run against the harness's *own* instruments rather than against
Kaggle's test file — the sealed holdout is the last 5% of training, and it is the
block this project trusts most (0041):

| key | sealed rows covered | test rows covered |
|---|---|---|
| `card1` | 98.9% | 98.2% |
| `card1_addr1` | 86.8% | 81.7% |
| `card1_addr1_D1n` | **43.0%** | **21.0%** |

> **The seal reports the winning key as twice as covered as it really is where the
> competition is graded.**

This is 0060's shape — each window in time flattering the next — arriving in
**coverage** rather than in score, which is a place nobody had looked for it. It
follows from the same geometry: the sealed block is adjacent to training, so it
shares more clients with it than a block thirty days further out does. A run that
built this feature would see a *plausible* coverage number and a *doubly
optimistic* one.

## What this says about 0011, and it is the point of the record

0011 refuses a column whose value depends on which other rows are in the frame.
That rule is doing **two jobs at once**, and this dataset separates them:

* **Leakage** — the column depends on the target. Refusing it is this project's
  foundation and is not in question anywhere.
* **Transduction** — the column depends on the *unlabelled* rows it is served
  alongside. A group mean over a predict batch reads no target, could be computed
  by anyone holding the batch, and is a standard, legitimate technique.

They are refused by the same check and they are not the same thing. On this
dataset the second refusal costs the feature the competition was won with; the
winners price their magic at **+0.011 local AUC** (0.9363 → 0.9472), which is
larger than every lever this project has ever measured except capacity on a bare
table.

**That is not an argument to remove the refusal**, and this record does not make
one. Three things stand in the way, and all three are load-bearing:

1. **Single-row serving genuinely breaks.** 0011's own example is the honest one:
   a batch of one row gets its own value back. A model that answers differently
   depending on who else is in the request is a real defect for a real deployment,
   and 0030 exists because this project takes serving seriously.
2. **The bundle contract is row-wise.** `predict_bundle` replays a lineage over
   raw rows (0003, 0010). A transductive step is not a lineage step; it is a
   property of the batch, and there is nowhere in the artifact for it to live.
3. **We cannot currently price it.** 0073 measured that this project's ruler
   overvalues exactly the class of step that hands a model structure it could
   otherwise reach by splitting, and a UID aggregate is the extreme case of that
   class. The probe would have a confident opinion about this feature and 0073
   says not to believe it at one budget.

## What would have to be true to build it

Recorded so that whoever picks this up does not start from scratch, and
deliberately **not chosen here**:

The harness would need the data's owner to declare a **serving mode** — that
predictions are issued for a batch rather than a row — the way `split_on`
declares ordering and `forecast_horizon` declares the gap. A step could then be
legal under batch serving and refused under row serving, and 0011's check would
test batch-independence *conditional on that declaration* rather than
unconditionally. That is a large change to the guarantee on the tin
("leakage-safe by construction"), and it should not be made to win a competition.

The cheaper thing, which is not the same thing and is worth separating: **report
the coverage**. `fit_group_stats` knows how many groups it fitted; it does not
know, and could compute, what share of the validation and sealed rows fall into
one of them. The table above is the argument that this number is not decoration —
it moves from 98% to 21% across three keys a run might plausibly try, and today a
run cannot see it. That is a small, honest change that names no dataset, and it is
the only thing here I would build without a second opinion.

## Limits

* **Coverage is not score.** This record measures how many rows would receive a
  constant, not what that costs in AUC. The +0.011 is the winners' number, on
  their features, with their model. Nothing here has been fitted.
* **`default_` is not nothing.** An uncovered row gets the population statistic
  and keeps every other column; the model is not blind, it is uninformed about the
  entity. "The feature is dead" would be too strong.
* **One dataset.** Client turnover this severe is a property of fraud data. The
  *law* — coverage falls as the key sharpens — is arithmetic and holds everywhere;
  how much it bites is a property of the data, and Rossmann's 1,115 stores are all
  present in both halves.
