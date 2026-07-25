# Two encoders, two jobs: OOF `city_te` vs. `final_enc`

Companion note to `fit_encoding` (`src/tabular_agent/encoders.py`) and
`notebooks/phase2_transforms_and_encoding.ipynb` (§4–5). Answers the question:
*"`final_enc` is target-encoding fit on all training rows — why isn't that leakage?"*

## The setup

`fit_encoding(method="target")` returns **two** things:

- **`train_feats`** — the *out-of-fold* `city_te` column, fit per fold. Goes into
  `train_cv` as the feature to **train + score** on.
- **`final_enc`** — a single `TargetEncoder` fit on **all** training rows. Goes into
  the `Pipeline` and is applied at **predict** time.

## Leakage is a property of the *evaluation*, not of "using all the data"

Leakage = **the model is scored on a row whose target it already peeked at.** The
encoder is used in two completely different situations, and only one is an
evaluation:

- **During CV (a row being *scored*):** if `city` is encoded using all rows
  including validation row A, then A's own `city_te` contains A's own churn value.
  The model sees A's label baked into A's feature, predicts A, and the recorded
  metric is a lie. → This is why the *metric* needs **OOF**: a scored row must be
  encoded by an encoder that never saw it.
- **At predict time (a row being *served*):** a brand-new customer's target is
  **genuinely unknown** and was **never part of building the encoder**. `final_enc`
  just says "historically, Munich churned at 0.70 in my training data." There is no
  target to peek at → no leakage possible. That is exactly the learned historical
  signal you *want* a model to use.

| | encoder used | served row in encoder's fit data? | leakage? |
|---|---|---|---|
| CV scoring row A | must exclude A → **OOF** (`train_feats`) | no (by construction) | avoided |
| new prediction | **`final_enc`** (all rows) | no — unseen future data | none possible |

## Why not reuse a fold-encoder for deployment?

Each fold-encoder saw only ~80% of the data. For the shipped model you want the
**most stable** estimate of each city's rate, so you fit on **100%**. There is no
held-out metric at predict time, so there is nothing left to be "honest" about —
you're past evaluation, you're shipping.

**One-liner: OOF exists so you don't fool yourself about performance; `final_enc`
exists to give the deployed model the strongest encoder. Different jobs → two fits.**

## The honest subtlety (scheme A — see `target_encoding_leakage_insight.md`)

`train_cv` fits its `final` *model* on the **OOF** `city_te` column, but at serve
time the pipeline feeds that model `final_enc`'s `city_te`. The two differ slightly:
OOF values are noisier (built from 80%, excluding the row), `final_enc` values are
cleaner (100%, including the row). That mild train/serve feature-distribution shift
is inherent to shared-fold OOF encoding — a trade-off, not a bug, and **not**
leakage: at no point is a row scored against its own leaked target.
