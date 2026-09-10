# Rossmann median gap — anatomy, and what our actual advantage is (2026-08-31)

Requested by Ulf: (1) which of {a) tuning/ensemble, b) features, c) weak agent
model} most explains why we're above the private median, and (2) given W&B's
Tabular Nomad wins where we don't — what exactly does that mean and what is *our*
advantage. Recorded here per his "halte deine recherche fest".

## Verified numbers
- Private board (0061): best 0.10021, top10% 0.11774, **median 0.12967**.
- Final run (Qwen/Chad `f0378d5` on `origin/chad/rossmann-final`, git-verified):
  lgbm sealed **0.13446** (best single), shipped blend (0044) sealed 0.14706.
  Gap of the best single to private median = **+0.00479**. Beats prior best 0.13813.
- 0061 horizon-robustness (public→private, joined by team id):
  field degrades **+0.01473** (Rossmann) / +0.04464 (Restaurant); **we degrade
  +0.00881 / +0.03097** — 40–44% less; our rank *improves* 62%→58% / 72%→64%.
  0061's own caveat: mid-table regression-to-mean is not ruled out.

## (1) Which cause — a/b/c
Ranked by measured impact (corroborates Qwen's anatomy):
- **a) ensemble/selection = the largest controllable lever.** Not "tuning" — the
  **unfiltered blend rule (0044)** ships the mean of all models including a
  catboost with +0.051 sealed optimism, costing ≈0.0126 (~4× the entire median
  gap). Shipping the best single model instead recovers most of it. Hyperparameter
  tuning proper is forbidden by design (0035) and is a much smaller lever.
- **b) features = real but secondary.** The winner already found the strong ones
  (t8 history: Store×6 honest windows, 65 features). The genuine miss is a
  *capability* the harness cannot express — trend extrapolation / forward calendar
  windows — not a feature the agent failed to try.
- **c) weak agent model = least likely as the cause.** The agent self-finalized
  and selected the correct winner table; a stronger agent yields marginally better
  feature proposals (feeds b), not a step change.

**But the load-bearing finding:** even stacking every measured lever (drop blend
+0.0126, budget +0.0023, winner features) reconstructs to ≈0.13024 — still ~+0.0006
over median. No remaining *measured* lever crosses it. The median was priced by
levers the project forbids (below).

## (2) What the Nomad comparison means, and our advantage
Nomad and this harness optimize different objectives:
- **Nomad = competition optimizer.** Iterates against the public leaderboard,
  lets the agent choose hyperparameters/models, stacks 677 features, uses external
  data + GPU budget. Those are exactly the rank-moving levers — and exactly what
  this harness forbids by decision (0001 no external data, 0035 no agent tuning, no
  leaderboard in the loop, six tools, one fixed model). On a frozen 2015 board,
  "score you can iterate against" *is* the graded objective, so they win the number.
- **The catch:** that objective does not exist in production. With no future
  leaderboard to iterate against, "optimize against the test board" is the
  definition of overfitting, not a strategy.

**Our advantage is a trustworthy model, not a higher score — and it's measured:**
1. **The reported number is the number you'd get in production.** Every guarantee
   is structural, not a post-hoc check: validation rows never reach a tool, feature
   code never sees the target, serving-unavailable columns (the `Customers` trap)
   cannot enter the shipped model, encoders serialize honest serving state, every
   table is replayable, the score is sealed. A leaderboard optimizer gives no such
   guarantee — it banks whatever leak the board rewards.
2. **We generalize better out-of-distribution (0061, above):** on the later
   private half we degrade 40–44% less than the field and our rank *improves*.
   Consistent with a harness that asks the forecasting question honestly (ordered
   split, forward-chaining folds). Honest caveat: regression-to-mean not fully
   excluded.

**One line:** we trade the score-winning levers for the one guarantee those levers
destroy — that the pipeline is leakage-free and behaves in deployment exactly as
measured. Worthless if the goal is "win Kaggle"; the whole game if the goal is
"automate ML for people with real data and no oracle."

## The fork (for Ulf)
Two different products:
- **Mission = competition score** → our constraints are a tax, Nomad is the bar,
  and beating median means unlocking the very things that define the project
  (0022 leaderboard-fitting warning). Then Qwen's read is right: harvest + writeup.
- **Mission = safe, deployable, auditable tabular ML** → the differentiated
  artifact already exists; the median was never the right yardstick. The right one
  is a user who ships our bundle and finds it behaves as promised. Rossmann's job
  (probe what the harness can't express) is done.
