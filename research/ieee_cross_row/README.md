# IEEE-CIS again: do features built from other rows earn their place?

Opened 2026-11-03. Status: **r1 run 2026-11-03.**

## Question

On independent rows with clean columns (credit default, freMTPL2, Otto) an
early-stopped GBM already builds most of what classic feature engineering would.
Where features clearly paid (Rossmann), they came from other rows: history per
store, statistics per group. IEEE-CIS is the dataset where that should matter
most: the label follows the card holder, and the winners' solution was aggregates
per reconstructed card holder (0071, 0076).

What do the agent's features add over a plain early-stopped model on the raw
columns, when the test is a later block of time with new clients in it?

What is different from the August run (0075, Kaggle private 0.8995): the run is
bounded by 60 tables / 30 minutes (0097), declared categoricals can be native
(c81144a), identifiers are proposed and confirmed, and the Kaggle board is spent
(0075), so this experiment scores its own test block instead.

## Design

- Data: `fraud_detection/train_joined.parquet` from `scripts/prep_ieee.py`
  (unchanged, except the `TransactionID` note cut back to what the competition says).
- The competition's geometry inside the labelled table: the last 30 days are the
  test block (85,431 rows, 3.5% fraud), the 30 days before them are dropped (the
  competition's gap), the agent gets the first 122 days (420,061 rows).
- `runner.py`: one run of `ta.ForecastingAgent` (full profile: it ships the blend),
  `split_on="TransactionDT"`, horizon 30 days, forecast length 30 days, metric AUC,
  agent model `gpt-6.1-terra@eu` at medium effort, 20-minute cap per fit.
  `TransactionID` is proposed as an identifier by name and accepted.
- `references.py`: early-stopped LightGBM on the raw columns, no LLM.

## Prediction (written before running)

- The raw early-stopped LightGBM scores 0.90–0.92 AUC on the test block.
- The agent's run beats it by 0.005 to 0.015, mostly from group statistics keyed on
  card and address columns; that is more than any feature step bought on credit
  default, freMTPL2 or Otto.
- Its card-level aggregates reach fewer test rows than training rows (0076):
  coverage on the test block is visibly below coverage on validation.

## Readings

**r1: test AUC 0.918, against 0.903 for the raw early-stopped LightGBM (+0.015).**
The top of the predicted range. 82 minutes: about 18 of search (35 tables), about 30
choosing the table, about 35 fitting the blend.

What the agent built, in order of the shipped table's lineage: time phases, native
categoricals, amount and D-column transforms (`origin_D*` = day minus D, the
competition's card-holder trick), frequency counts and then target encodings of
card, card × address, card × email and device keys (t15, t19, t35). Validation AUC
under the probe went 0.887 (work table) → 0.902 (frequencies) → 0.907 (first target
encodings) → 0.909. One `fit_group_stats` call, nine `fit_history` calls, none of
which reached the shipped table.

- **The +0.015 is not all features.** The reference is one early-stopped LightGBM;
  what shipped is a six-member blend. The run has no test-block score of the raw
  table under the same blend.
- **The blend is under capacity.** Every member is a library default at 200 rounds
  (lgbm 0.902, xgb 0.892, catboost 0.892, three more configurations 0.889–0.901 on
  validation); the blend scores 0.9088. The early-stopped fit on the same table t35
  scored 0.9125 (697 rounds). 0098 fixed this for the minimal profile only.
- **The fitted ruler chose a table the probe ranked 13th.** Probe: t28 at 0.9089;
  fitted ruler: t35 at 0.9125, then t22 (probe rank 10). A shortlist of the probe's
  top 8 plus their lineage was built for speed and dropped on this reading: on this
  run it would have shipped t19 at 0.9083. On seven earlier runs the fitted winner
  had sat within the probe's top 9; on 420k rows with target encodings it does not.
  The full refit costs about 30 minutes here.
- **Coverage, prediction wrong.** Share of rows whose key was seen in earlier rows:
  card1+addr1 0.932 on validation vs 0.950 on test; card1–5+addr1 0.907 vs 0.940.
  The test block has more history behind it (all 420k rows rather than the first
  230k), and the 30-day gap costs less than that gains.
- **Exploration was rare.** 2 `explore_table` and 2 `explain_table` calls against
  about 37 building calls.

**Kaggle, 2026-11-04 (`submit.py`, one submission each).** Ranks against the board
saved on 2026-08-17 (6,355 teams; private median 0.911, top 10% 0.928, first
0.946).

| file | private | rank | public |
|---|---|---|---|
| r1's features and blend, refitted on all 590k labelled rows | **0.9159** | top 41% | 0.9383 |
| early-stopped LightGBM, raw columns, same rows | 0.9023 | top 65% | 0.9273 |
| r1's bundle as shipped (never saw the last 60 labelled days) | 0.8948 | top 72% | 0.9159 |
| August run (0075) | 0.8995 | top 69% | 0.9282 |

- The features are worth **+0.0136 on the private board**, against +0.015 on our own
  test block: the local reading held.
- The first run of this project above a Kaggle median on IEEE-CIS, by 0.005. Top 10%
  is 0.012 further.
- Staleness costs 0.021: the bundle as shipped is fitted on rows that end 90 days
  before Kaggle's test block. That is a property of scoring a finished competition,
  not of the product, which refits on every row it was given (0092).
- All three files predate 0099: the blend in them still runs at 200 rounds.

**k1, 2026-11-04: the competition's own split, one early-stopped LightGBM (0100).**
A fresh run on all 590,540 labelled rows (420,061 training, 170,479 validation), 17
tables, stopped during its blend fits and finalized by `finish.py` without them
(`runner.py k1`). Table t7 (time phases, categoricals, rolling target means per
card/address/email key at 60 and 90 days, amount and email transforms), 564 rounds,
validation AUC 0.9130. **Kaggle private 0.8997, public 0.9302.**

- Below the raw early-stopped LightGBM's private 0.9023, and 0.016 below r1's refitted
  blend, while 0.003 above the raw model on public. The drop from public to private
  is 0.031, against 0.022 for r1's blend and 0.025 for the raw model.
- **It does not measure 0099 or the blend.** Features and model both differ from r1:
  r1's table has frequency counts and target encodings and no rolling means.
- Whether its rolling target means go stale past the last label: tested below; not
  the cause.

**r1's table under one early-stopped LightGBM (`submit.py r1 single`), 2026-11-04.**
The same features as r1's refitted blend, on the same 590,540 rows, 697 rounds at
learning rate 0.03 (the count r1's fitted ruler stopped at on 230k training rows).
**Kaggle private 0.9164, public 0.9382**; the 200-round blend scored 0.9159 and 0.9383.
Rank correlation between the two files 0.93.

- On one set of features the blend adds nothing over the single model. This is the
  200-round blend; a blend at capacity is still unmeasured.
- So k1's 0.8997 was its features, not its model: 0.0167 between r1's and k1's tables
  under the same kind of model. Two runs of the same agent on the same data differ by
  more than the features are worth against the raw table (+0.014). Their validation
  scores, t7 0.9130 and t35 0.9125, are on different splits and do not compare; on one
  split validation does separate them (below).

**Why k1's table lost (`staleness.py`, `staleness.json`), 2026-11-04.** Prediction: its
rolling target means go stale past the last label, and validation cannot see it.
**The mechanism is real; it is not the cause, and validation does see the difference.**

Rolling means missing on Kaggle's test rows, by days past the last label (about 52%
on the labelled rows, from rare keys):

| window | 30–60 | 60–90 | 90–120 | 120+ (284k of 506k rows) |
|---|---|---|---|---|
| 60 days | 56% | 100% | 100% | 100% |
| 90 days | 44% | 57% | 100% | 100% |
| 120 days | 41% | 45% | 58% | 100% |

AUC under one early-stopped LightGBM, same training rows and same later rows:

| fitted on | scored on | k1 table | k1 without rolling | r1 table |
|---|---|---|---|---|
| first 90 days | 30–50 days later | 0.8984 | 0.8959 | 0.9166 |
| | 50–70 | 0.8753 | 0.8731 | 0.8942 |
| | 70–92 | 0.8747 | 0.8744 | 0.8978 |
| | all | 0.8830 | 0.8814 | **0.9032** |
| k1's training rows | k1's validation, first 30 days | 0.9271 | 0.9224 | 0.9274 |
| | k1's validation, after that | 0.8981 | 0.8966 | 0.9163 |
| | k1's validation, all | 0.9130 | 0.9097 | **0.9220** |

- The rolling columns never hurt in reach of the labels; their gain fades with
  distance (+0.0025 to +0.0003). The private board's distance is out of reach here.
- r1's table beats k1's by 0.020 on the first cut, about the 0.0167 on Kaggle, and
  k1's table is about the same without its rolling columns. What differs is what k1
  never built: r1 has 14 target encodings, 7 frequency counts and 8 `origin_D` columns;
  k1 has none of them.
- **Validation would have preferred r1's table**, 0.9220 against 0.9130, on rows r1
  never saw and k1's search did. So the agent stopped short, on a table validation
  could have beaten. Not established: which of the three families carries the gap.
- The gap opens with distance: level on the first 30 validation days, 0.018 after.
