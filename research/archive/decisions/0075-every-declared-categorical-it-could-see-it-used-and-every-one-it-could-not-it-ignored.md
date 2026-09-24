# 0075 — Every declared categorical it could see, it used; every one it could not, it ignored

- **Date:** 2026-08-17
- **Status:** accepted (a measurement — the first run on IEEE-CIS Fraud Detection,
  at unchanged defaults, scored externally). **Two of six registered predictions
  refuted and the refutation is the finding; the remaining four confirmed.**
  Private **0.899498**, public **0.928187**, rank ~4354/6355. **This board is now
  spent** (one submission, per the plan the run was built to).
- **Related:**
  [0073](0073-the-ruler-is-mispriced-on-one-dataset-and-three-mechanisms-failed-to-say-why.md)
  (which found that no model here is ever told a column is categorical, and warned
  that AUC differences on a 3.5%-positive target are noise),
  [0074](0074-a-categorical-declaration-is-a-fitted-step-not-a-flag.md) (whose
  premise this run weakens in one half and confirms in the other),
  [0071](0071-the-gap-was-not-the-vocabulary-it-was-a-column-nobody-asked-for.md)
  (whose central conclusion this replicates on a second dataset and a second
  competition),
  [0018](../../../docs/decisions/0018-the-payload-is-a-budget.md) (`MAX_REPORTED_COLUMNS`, the cap this run
  is the first to actually hit),
  [0059](0059-the-interval-belongs-to-the-difference-not-the-score.md) (the
  interval, firing on 18 of 20 comparisons here),
  [0076](0076-a-fitted-aggregate-loses-coverage-exactly-as-its-key-becomes-worth-having.md)
  (what the winners had that this run could not reach),
  `scripts/prep_ieee.py`, `scripts/run_ieee_v1.py`, `_runs/ieee_v1/`.

## Why a fifth dataset

Rossmann's private board is spent (0045) and Restaurant's is spent for the
question it was kept for (0066). A harness with no external board cannot check its
own answer, which is a worse position than being 0.008 above a median. IEEE-CIS is
finished, both leaderboards are on disk, and it is the first dataset here with
**high-cardinality identity at scale**: 590,540 rows, 434 columns, `card1` alone
holding 13,553 distinct values.

Nothing was changed for it. Model, thinking budget, split fractions, fit budget,
stop policy, prompts and tool surface are what every other run has used.

## What the competition declares, and what the harness does with it

The competition's data page names **49 columns as categorical**. Measured, not
assumed:

| | | what the harness does |
|---|---|---|
| **31 arrive as strings** | `ProductCD`, `card4`, `card6`, both email domains, `M1`-`M9`, `DeviceInfo`, `id_12`… | `_feature_frame` **drops** them, and reports the names |
| **18 arrive as numbers** | `card1` (13,553 levels), `card2` (500), `addr1` (332), `id_19` (522)… | **kept, and split as ordered quantities** — no report of anything |

0074 argued the second row is the dangerous one because it is silent. This run was
built to find out whether that silence changes behaviour.

## The registered predictions, and the two that failed

Registered in `run_ieee_v1.py`'s docstring before the run started, with the raw
work table measured through the probe beforehand at **0.89604**.

**1. The identity block will be invisible and the agent will encode none of it.**
The per-column profile is capped at `MAX_REPORTED_COLUMNS = 100`, and column 100
of this table is `V45`, so all 41 identity columns fall off the end of
`inspect_data`. **Refuted.** It frequency-encoded `DeviceType`, `DeviceInfo`,
`id_12`, `id_15`, `id_16`, `id_23`, `id_27`-`id_31`, `id_33`-`id_38` in `t2`, and
target-encoded `DeviceInfo` in `t12`.

The reason is worth keeping: `non_numeric_columns` is **not** truncated. The agent
saw those names in a list of what had been dropped — no dtype, no cardinality, no
description — and that was enough to act on. **A bare name in a refusal is a
working channel**, which is a cheap and useful thing to know about a payload this
project treats as a budget (0018).

**2. It will encode the string categoricals it can see.** **Confirmed**, and
completely: all 31 strings were encoded.

**3. Nothing will touch `card1`, `addr1`, `card2` or `id_19`.** The sharp one, and
**refuted**. `t3` target-encodes `card1`, `addr1`, `addr2`; `t5` builds `card1`
target-history; `t10` combines `card1` with `addr1` into a key; eight group-stat
tables are keyed on `card1`.

**4. `TransactionID` ships.** **Confirmed.** It is in all four terminal fits'
feature sets, alongside `TransactionDT`. Nothing refuses it and nothing should
(0029) — it is genuinely present at predict time, so 0030 has no objection. Third
instance of the trap, after `Id` on house-prices (0019) and `Customers` on
Rossmann.

## The finding, which is sharper than either prediction

Predictions 1 and 3 failed in opposite directions, and putting them together gives
a cleaner rule than either. Across all 49 declared categoricals, **with no
exceptions**:

| | touched by the run | untouched |
|---|---|---|
| **31 strings** (all listed in `non_numeric_columns`) | **all 31** | none |
| **6 numbers inside the first 100 columns** — `card1`, `card2`, `card3`, `card5`, `addr1`, `addr2` | **all 6** | none |
| **12 numbers past column 100** — `id_13`, `id_14`, `id_17`-`id_22`, `id_24`-`id_26`, `id_32` | none | **all 12** |

> **The line is not string-versus-number. It is visible-in-some-channel versus
> visible-in-neither.** Every declared categorical the run could see, it used.
> Every one it could not see, it ignored. Forty-nine for forty-nine.

The 12 it missed are invisible **twice over**: truncated out of the profile because
they sit past column 100, and absent from the dropped list because they are
numeric. There is no channel in which they appear, and so there is no decision to
make about them — the model simply splits them as ordered quantities and nothing
anywhere says so.

## What this does to 0074

Both halves of 0074's premise are tested and they come apart.

* **Weakened where the column is visible.** 0074's claim is that the defect is not
  that label encoding is bad but that *nobody chose it*. For `card1` that is now
  false: the agent read the declaration next to the profile and chose — target
  encoding, target history, and eight aggregate keys. It made a choice, from the
  three encodings that exist.
* **Confirmed where it is not.** For `id_19` and its eleven siblings nobody chose
  anything. They are label-encoded by default, invisibly, exactly as 0074 says.

So the sentence to carry forward is narrower and more useful than the one 0074
wrote: **the declaration channel works when it reaches the agent, and the harness
has a second defect that decides whether it reaches the agent at all.** What 0074
asks for — native categorical handling as a fourth option — is untouched by this
run either way; the agent never had it.

## The vocabulary was there again, and that is 0071 on a second dataset

The 1st-place solution's feature engineering, before its magic feature, against
what this run built unprompted:

| winners (`xgb-fraud-with-magic-0-9600.ipynb`) | this run |
|---|---|
| `encode_FE(['addr1','card1','card2','card3','P_emaildomain'])` | `t2`, `t3` — frequency and target encoding, same columns |
| `encode_CB('card1','addr1')` — combine into a key | `t10` — `c1_addr`, `c1_c2` |
| `encode_AG(['TransactionAmt',…], ['card1','card1_addr1'], ['mean','std'])` | `t6`/`t7`/`t8` `card1_TransactionAmt_max/mean/std`, `t9` `addr1_TransactionAmt_mean` |
| `encode_AG2([...], ['uid'])` — nunique within group | `t15`-`t20` `card1_P_emaildomain_nunique`, `card1_addr1_nunique/count`, `card1_card2_nunique` |

Four families, largely the same columns, unprompted, in one session.

> **0071 concluded that the distance to a winner is not the feature vocabulary.
> That was n=1 on Rossmann. It is now n=2, on a different competition, a different
> metric and a different winning team.**

Where the two solutions part is a single key — `card1_addr1_D1n` — and the reason
this harness cannot have it is not vocabulary either. That is 0076.

## The ruler could not resolve its own steps

**18 of 20 comparisons came back `sign_stable: False`, and 9 of those are positive
gains the run treated as improvements.** Block sd runs 0.003 to 0.014 against gains
of 0.001 to 0.006. Restaurant v1, the previous high-water mark, was 6 of 17 (0060).

This is the gotcha 0073 registered — AUC differences under ~0.005 on a
3.5%-positive target are noise — arriving on schedule. It does not invalidate the
chain, and the chain did improve: work table 0.89604, best table `t18` 0.91734.
What it means is that **no individual step in this run may be quoted as a
finding**, and that the harness selected a 24-table chain on differences smaller
than its own instrument's noise, with the instrument saying so nine times and
nothing listening. 0059 built that interval and nothing is gated on it; this is
the third dataset on which that is visible and the most extreme.

One incidental waste: `t15` and `t17` both produced `card1_addr1_nunique`. The run
built the same column twice.

## The terminal fits disagree by more than any feature step

| model | val AUC |
|---|---|
| lgbm | **0.91734** |
| catboost | 0.89584 |
| xgb | 0.89197 |

A **0.025 spread between families on the same table**, at library defaults and 200
trees — larger than the entire feature chain's 0.021. 0067's standing note that
every hyperparameter except `n_estimators` and `learning_rate` is a library default
in every run this project has ever made now has a second dataset behind it, and the
winners' XGB runs `max_depth=12, learning_rate=0.02, colsample_bytree=0.4,
n_estimators=5000` with early stopping on a held-out month.

## The seal came back the wrong way round, and no mechanism is established

| model | val | sealed | optimism |
|---|---|---|---|
| lgbm (the argmax) | 0.91734 | 0.93817 | **−0.02083** |
| xgb | 0.89197 | 0.92526 | −0.03329 |
| catboost | 0.89584 | 0.92128 | −0.02544 |
| **blend (lgbm+xgb+catboost)** | **0.91874** | **0.94176** | −0.02302 |

AUC is higher-is-better, so a negative optimism here means **the sealed block was
easier than validation** — by 0.021 to 0.033, on all four candidates. Every other
dataset in this repo has the seal *harder* than validation, and 0060's whole
finding is that each window in time flatters the next.

The mechanism is **not established, and the agent's own explanation is not
adopted.** It closed the run saying the entity features "shine most for cards that
already have fraud history in the full training span, which the later holdout rows
have more of". That is plausible and it is not what was measured; both blocks sit
outside the work half that the encoders were fitted on. What the composition
actually shows:

| block | rows | days | fraud rate | `card1` seen in work |
|---|---|---|---|---|
| work | 531,486 | 1.0–161.9 | 3.4714% | — |
| val | 29,527 | 161.9–172.0 | 3.5357% | 99.10% |
| sealed | 29,527 | 172.0–183.0 | **3.9591%** | 98.86% |

The sealed block carries a **12% higher fraud rate** and *slightly worse* entity
coverage. Neither obviously buys +0.021 AUC, and a class-balance shift moves AUC in
no fixed direction. Recorded as an open question rather than explained, because
0058 is the record of this project accepting a mechanism that turned out to be
about composition and not about what it claimed.

**It matters for how the sealed number is read.** Registered prediction 6 said the
sealed AUC would come back above the private one. It very likely will — but if the
seal is anomalously easy for a reason nobody has named, that gap is inflated and
must not be quoted as this dataset's horizon decay.

## The blend beat every member, on both blocks

0044 ships the equal-weight average rather than the argmax. Here it wins on
validation (0.91874 against lgbm's 0.91734) **and** on the sealed rows (0.94176
against 0.93817) — a fifth dataset, and the first on a higher-is-better metric.
0045's warning about the weakest member is untested here for the reason in the next
section.

## A terminal fit vanished without trace, and that is a defect

`TERMINAL_MODEL_TYPES` is `(lgbm, xgb, catboost, elasticnet)`. **Only three GBDTs
are in the journal**, the bundle holds three models, and `experiments/` holds
`e1`–`e4`. Searching the whole journal for the string `elasticnet` returns nothing.

It was attempted. `models.py:267` fired four times —

> *"the linear fit stopped at its 200-iteration ceiling rather than converging, so
> it may be under-fitted — and this model type is the yardstick other models are
> judged against"*

— and the process sat at 99% CPU emitting `sag` convergence warnings for **at least
45 minutes** of the run's 93 (observed by polling; not instrumented, which is part
of the complaint). Then it was discarded. `_terminal_fits` says so by design: *"A
model type that fails is skipped, not fatal."*

> **The skip is right and the silence is not.** A reader of this run's journal
> cannot tell that a fourth candidate was ever on the ballot, that the blend is
> missing its only non-tree member, or that roughly half the wall clock went into a
> fit that produced nothing.

This is 0015's subject — *attribution gaps are reported, not prevented* — with the
gap unreported, and it lands on the model that `models.py`'s own docstring calls
the yardstick the others are judged against. It also puts a number on 0033 rather
than a principle: the run's `fit_time_budget_s` was **20 minutes**, the check is
between folds by design (`models.py:550`), and the arm ran at least 45. The bound
Porto Seguro bought — `_SAGA_MAX_ITER = 200`, `PENALTY_SEARCH_MAX_ROWS = 20_000` —
is in place and working; what is unbounded is the deliberate part, the refit on
every row, and at 531,486 × 489 that is what a fold costs.

Neither is fixed here. Both want their own record: one line in the journal saying a
candidate was attempted and why it is absent would close the reporting half.

## Two things prep found that no run would have

Recorded because both would have been silent failures, and neither is about
modelling:

* **`test_identity.csv` spells its columns `id-01`; the training file uses
  `id_01`.** Left alone, a bundle's stored lineage asks for `id_01` at predict time
  and 0030 correctly refuses rows carrying `id-01` — 0055's failure exactly, after
  the run is over. Normalised in `prep_ieee.py`, on both sides.
* **A note written to nobody.** The first draft put the categorical declaration in
  a single `_declared_categorical` entry listing all 49 names. `profile_dataframe`
  attaches a note to its column and silently drops any key that is not one, so that
  entry would never have been shown to anyone. The dictionary was well-formed and
  the text was correct. 0050's shape in the reporting layer again.

## Predictions 5 and 6: both confirmed, and the board is now spent

One submission was made. **Private 0.899498, public 0.928187.**

**5 — confirmed.** Predicted "private between 0.88 and 0.91, below the median
0.91096". It landed at 0.899498, inside the interval and **0.01146 below the
median**, at rank ~4354/6355 (top 69%).

**6 — confirmed.** Predicted the sealed AUC would come back above the private one.
Sealed 0.941757 against private 0.899498: **the seal was optimistic by +0.04226.**

**The board is spent from here.** A finished competition returns a private score on
every late submission, so this seal is self-imposed discipline and nothing else —
0045 is the record of what happens without it.

## The gradient, and the seal is a local peak rather than a step on it

Four windows in time order, each the harness's own reading of itself:

| window | rows | AUC |
|---|---|---|
| val (days 161.9–172.0) | 29,527 | 0.91874 |
| **sealed (days 172.0–183.0)** | 29,527 | **0.94176** |
| *30-day gap, no rows released* | | |
| public (nearer half of the test block) | ~254k | 0.92819 |
| private (further half) | ~253k | 0.89950 |

0060's law is that each window flatters the next, and from `sealed` onward it holds
exactly — 0.94176 → 0.92819 → 0.89950, monotone in distance from training, which is
0066's gradient on a fourth dataset.

**But the sequence is not monotone at its start.** The sealed block scores *above*
the block immediately before it and above everything after it. It is a local
maximum, not a point on a decay curve, and that is the anomaly this record flagged
before the submission and could not explain.

The decomposition is worth writing down, because it says which number is quotable:

| | |
|---|---|
| val → private | **+0.01924** |
| sealed → private | +0.04226 |
| **difference, i.e. how much the seal's own easiness contributes** | **≈ +0.023** |

> **Roughly half of the seal's 0.042 optimism is the sealed block being unusually
> easy, not the horizon being far away.** So this dataset's horizon decay is
> +0.019 (val) or +0.029 (public → private) — **not** the +0.042 the sealed number
> invites. Quoting the sealed gap as horizon decay would have overstated it about
> twofold.

The caution registered here before the upload — *"if the seal is anomalously easy
for a reason nobody has named, that gap is inflated and must not be quoted as this
dataset's horizon decay"* — is the thing this measurement most clearly earns. The
mechanism is still not established.

## 0061's pattern holds a third time, and is nearly gone

On Rossmann and Restaurant the field degraded more from public to private than we
did. Third instance:

| | public → private |
|---|---|
| **ours** | **+0.02869** |
| field median | +0.03031 |
| field top 10% | +0.02502 |
| field best | +0.02226 |

Directionally the same and **much weaker**: Rossmann was +0.0147 field against our
+0.0088 and Restaurant +0.0446 against our +0.0310 — decays 30–40% smaller than the
field's. Here it is 5%. We move from rank ~4507/6355 public (top 71%) to ~4354/6355
private (top 69%), which is 153 places.

Two things keep this from being evidence for much. The field's own decay **grows as
score falls** (best +0.0223, top 10% +0.0250, median +0.0303), so a decay figure is
not comparable across score levels without care. And a below-median team moving
slightly up is what regression to the mean predicts — which 0061 named as not ruled
out, and this instance does nothing to rule out.

## Limits

* **One run, one dataset, no repetition.** The 49-for-49 split is clean but it is a
  single agent trajectory; a second run might encode `id_19` by luck.
* **The visibility explanation is inferred, not instrumented.** What is measured is
  which columns were touched and which channels each column appears in. That the
  cap *caused* the omission is the obvious reading and it is not proved — the test
  would be to raise `MAX_REPORTED_COLUMNS` and re-run, which was deliberately not
  done before measuring (it would be the harness tuned against the dataset
  measuring it).
* **`0.89604` is the harness's own number for the raw table**; my pre-run
  measurement of the same thing was 0.89125, most likely tie ordering among the
  17,191 duplicate `TransactionDT` values. Unresolved, and it is the journal's
  number that every comparison here uses.
