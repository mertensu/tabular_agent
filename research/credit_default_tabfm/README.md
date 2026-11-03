# The default product on UCI credit default, on a TabICL post's design

Opened 2026-11-03. Status: **run 2026-11-03; readings below.**

## Question

A LinkedIn post compared TabICL v2 (no tuning) with XGBoost tuned by 30 Optuna
trials on the UCI "default of credit card clients" data (Yeh & Lien 2009; 30,000
clients, 22% default). Its test AUC, mean of 3 seeds:

| training rows | TabICL | XGBoost + 30 Optuna |
|---|---|---|
| 1,000 | 0.757 | 0.744 |
| 5,000 | 0.775 | 0.768 |
| 24,000 | 0.784 | 0.780 |

Where does `ta.Agent()` land on the same design? It is a new dataset for the
harness (credit risk, binary, small, independent rows), so it is also a probe:
does anything in it hit a gap?

## Design

- Data: Kaggle copy `UCI_Credit_Card.csv` into `credit_default/` (gitignored);
  notes in `scripts/prep_credit.py`, transcribed from the UCI description.
- One test set of 6,000 rows (stratified, seed 0), never seen by a run. Training
  rows: a stratified sample of 1k / 5k / all 24k of the rest, seeds 0, 1, 2.
- `runner.py`: one run of `ta.Agent()` per cell, minimal profile, its own
  holdout off (the 6,000 rows are the holdout), agent model
  `gpt-6.1-terra@eu` at medium reasoning effort, as for the ForecastingAgent
  repeats (Rossmann r12–r14, Restaurant r3–r11). Score: the
  shipped bundle's AUC on the test rows.
- `anchors.py`: XGBoost + 30 Optuna trials on the same 9 cells, because the
  post's split is not published and AUC on a different 6,000 rows moves by about
  ±0.005. The search space is ours, not the post's. TabICL is not rerun.

## Prediction (written before running)

- The agent lands within 0.005 AUC of our XGBoost anchor at every size; neither
  side is clearly ahead. The dataset is 23 clean columns and is known to plateau
  near 0.78 AUC.
- At 1k rows the agent's 200-round probe at the default learning rate overfits
  more than a tuned model and is below the anchor, by up to 0.01.
- Agent features (delay counts, bill/limit ratios) buy at most 0.003 over the raw
  table.

## Readings

Test AUC on the 6,000 test rows, mean of seeds 0–2 (`result_*.json`,
`anchors.json`, `raw_arms.json`; `raw_arms.py` fits the run's own model on the
raw 23 columns, no LLM):

| training rows | agent (shipped) | raw table, probe | raw table, LightGBM early-stopped | XGBoost + 30 Optuna | post: XGBoost | post: TabICL |
|---|---|---|---|---|---|---|
| 1,000 | 0.732 | 0.727 | 0.742 | **0.760** | 0.744 | 0.757 |
| 5,000 | 0.746 | 0.743 | 0.766 | **0.773** | 0.768 | 0.775 |
| 24,000 | 0.777 | 0.776 | **0.784** | **0.784** | 0.780 | 0.784 |

- **The agent lost all 9 cells to tuned XGBoost**, by 0.028, 0.027 and 0.007.
  The prediction (within 0.005) was wrong at every size.
- **The features are worth +0.002 to +0.005 over the raw table at the same
  model** (agent minus raw probe, per cell: 0.000 to +0.010, one of 9 negative).
  They are sensible (delinquency counts and trends, utilization, paid/bill) and
  small, as predicted.
- **Capacity is worth 4–7× that.** The same LightGBM, early-stopped at learning
  rate 0.03 on the raw table, gains +0.015 / +0.022 / +0.008 over the probe, and
  at 24k rows it equals tuned XGBoost. At 1k and 5k the remaining 0.007–0.018 to
  XGBoost is tuning (depth, regularization), not rounds.
- **The 200-round probe is what ships here**, and on this dataset it is the
  bottleneck at every size, most at 5k. Same reading as `capacity_ship_minimal/`.
- **Our tuned XGBoost beats the post's** at every size (0.760 vs 0.744 at 1k), and
  matches or beats its TabICL. The difference is the test split and the search
  space, so the post's headline (+0.013 for TabICL at 1k) is within what a
  different split and search move.
- **Runs are short**: 8–25 tables in about a minute each; every run finalized by
  itself well inside 60 tables / 30 minutes. At 1k rows the agent ranks on 200
  validation rows; its validation scores ranged 0.71–0.81 across seeds.
- **Spread**: at 5k the agent's test AUC spans 0.733–0.756 across seeds, XGBoost
  0.772–0.774. The run-to-run variation is the probe ranking on few rows, not the
  data.
- No harness gap surfaced: the dataset needed nothing the minimal profile lacks.

### The features at capacity (`features_at_capacity.py`, added after the readings above)

Expected, after the above: the features' gain shrinks to about zero once the
model is early-stopped (the Rossmann v7 pattern). It did not. The same
early-stopped LightGBM, fitted on each run's selected table instead of the raw
table, test AUC, mean of 3 seeds:

| training rows | selected table, early-stopped | raw table, early-stopped | features' gain (per seed) | XGBoost + 30 Optuna |
|---|---|---|---|---|
| 1,000 | 0.751 | 0.742 | +0.009 (+0.011, +0.009, +0.009) | 0.760 |
| 5,000 | 0.769 | 0.766 | +0.004 (+0.011, +0.001, −0.001) | 0.773 |
| 24,000 | **0.788** | 0.784 | +0.004 (+0.005, +0.006, +0.002) | 0.784 |

- **At capacity the features are worth more, not less**: +0.004 to +0.009 against
  +0.002 to +0.005 at the probe; 8 of 9 cells positive, largest at 1k rows.
- **At 24k rows, agent features + an early-stopped fit beat tuned XGBoost on the
  raw table** in all 3 seeds (0.788 vs 0.784), and match the post's TabICL.
- At 1k and 5k it is still 0.004–0.009 behind XGBoost: early stopping on a 15%
  slice of 1,000 rows stops at 35–70 rounds, and depth/regularization are untuned.
- So on this dataset the shipped 200-round probe hides real feature value as well
  as costing capacity: what the agent builds is worth keeping, what it ships is not
  the right fit of it.

### What ships after 0098 (`shipped_replay.py`)

0098 makes the minimal profile ship the early-stopped fit on the probe's chosen
table. Replayed offline on the 9 runs (same tables, no agent rerun), test AUC:

| training rows | shipped before | shipped after 0098 | gain | XGBoost + 30 Optuna | post: TabICL |
|---|---|---|---|---|---|
| 1,000 | 0.732 | 0.750 | +0.018 | 0.760 | 0.757 |
| 5,000 | 0.746 | 0.768 | +0.022 | 0.773 | 0.775 |
| 24,000 | 0.777 | **0.789** | +0.012 | 0.784 | 0.784 |

All 9 cells improve. Against tuned XGBoost the agent now wins 3 of 9 (every 24k
seed), and is 0.010 behind at 1k and 0.005 behind at 5k. Stopped at 51–64 rounds on
1k rows, 69–152 on 5k, 126–138 on 24k.
