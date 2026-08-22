# Target encoding: who gets lied to?

Companion note to `research/archive/notebooks/target_encoding_step_by_step.ipynb` (sections 9–12).

## The core mechanism (scheme B — "fit encoder on the 4 training folds")

1. The encoder computes each city's mean target from the training rows — so every
   training row's **own target is one of the numbers inside its own feature value**.
2. The model then learns "feature → target" on those same rows. Rows with `y=1` got
   slightly higher encodings (their own 1 pulled the mean up), rows with `y=0`
   slightly lower — so the feature *looks* predictive **even when the category has
   zero real relationship to the target**. The correlation was manufactured in step 1.
3. The model trusts what correlates: in the notebook's experiment it puts **16× more
   weight** on a pure-noise feature under scheme B than under A/C.
4. At validation the self-boost is gone (held-out rows are encoded without their own
   targets), the feature is weaker than the model believes, and AUC drops
   (≈0.79 vs ≈0.81).

**One-liner: the metric isn't lied to — the model is.** The validation score honestly
measures a model trained on a doctored correlation.

## The mirror image (scheme A — shared-fold OOF, what the library does)

Training-row encodings exclude each row's own target (no fabricated correlation,
model learns honest trust) — but they *include the validation fold's* targets, so the
metric gets a tiny flattery (≈0.52 AUC on pure noise instead of 0.50) while the model
stays sound.

## Summary

| Scheme | who gets lied to | cost |
|---|---|---|
| B — fit on training folds (sklearn-`Pipeline` style) | the **model** (over-trusts a doctored feature) | real AUC lost |
| A — shared-fold OOF (**the library**) | the **metric** (slightly) | ≈ +0.02 optimism, worst case |
| C — nested CV | nobody | ~5× encoding compute |

B lies to the model; A whispers to the metric; C shuts both up.
