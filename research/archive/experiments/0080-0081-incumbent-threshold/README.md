# Incumbent block-threshold experiment (0080–0081)

Archived research experiment; **not part of the product runtime**.

## Question

Does the historical ordered-run rule `gain > 0.25 × incumbent_block_sd`
separate candidates that remain best-so-far under a higher-capacity reruling
better than `gain > 0`?

The rule and falsifier were fixed in `PREREGISTRATION.md` before reconstruction.
`FINDING.md` records the result.

## Result

Negative. On 18 capacity-labelled Restaurant steps, `k=0.25` accepted exactly
the same 11 steps as zero: 7 capacity-positive, precision 0.636, recall 1.0.
The `k=0.5/1.0` readings are exploratory only and cannot replace the failed
registered value. The exact candidate-vs-incumbent contrast remains useful as a
reporting instrument.

## Contents

- `PREREGISTRATION.md` — fixed geometry, primary rule, falsifier and limits
- `FINDING.md` — controls, result and interpretation
- `analyze.py` / `test_analysis.py` — reconstruction and focused tests
- `contrasts.csv` / `summary.json` — tracked deterministic readings
- `walkthrough.ipynb` — unexecuted, smoke-tested teaching walkthrough
- `artifacts.tar.gz` — journals, validation rows and prediction sidecars for the
  three covered runs; no fitted models or raw competition datasets

## Reproduce

From the repository root:

```bash
mkdir -p /tmp/tabular_incumbent_artifacts
tar -xzf research/archive/experiments/0080-0081-incumbent-threshold/artifacts.tar.gz \
  --exclude='*/._*' -C /tmp/tabular_incumbent_artifacts
uv run pytest research/archive/experiments/0080-0081-incumbent-threshold/test_analysis.py
uv run python research/archive/experiments/0080-0081-incumbent-threshold/analyze.py \
  --runs-root /tmp/tabular_incumbent_artifacts/_runs
```

The analysis defaults to writing its outputs back into this directory.

## Known limits

- 18 dependent capacity-labelled steps from two runs of one dataset family
- IEEE contrasts are descriptive because no high-capacity reruling exists
- the historical block implementation split tied dates on all four Restaurant
  block boundaries
- higher capacity is a second ruler, not external ground truth
