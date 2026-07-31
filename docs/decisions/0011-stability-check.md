# 0011 — Statelessness is tested as a property, not blacklisted

- **Date:** 2026-07-30
- **Status:** accepted
- **Commit:** `84dd8bb`
- **Found by:** reading the code an agent actually wrote in a live run
- **Corrects:** a claim made in `notebooks/phase3_harness_step3_transforms.ipynb`

## The problem

Decision 0010 stores agent-written code and re-runs it at prediction time. That
is only sound if the code is **stateless** — if a row's output depends on that
row's inputs and nothing else.

We believed this was already guaranteed. The sandbox blocks `sklearn`,
`lightgbm`, `xgboost` and `catboost`, so the code cannot fit an estimator. The
step-3 notebook said so in as many words: code that cannot import a fitting
library "cannot learn a parameter from the training set".

That claim was wrong, and a live run proved it.

## The bug

Inside an otherwise good sixty-line transform, the agent wrote:

```python
df["price_bin"] = pd.qcut(df["price"], 10, labels=False, duplicates="drop")
```

`qcut` splits a column at its deciles — **deciles computed from the frame it is
handed.** At training that is 800 rows. At prediction it is whatever arrives.

Loading the shipped bundle and predicting the same five rows two ways:

```
batched with 800 rows:  [0.0912 0.9953 0.9277 0.0303 0.1252]
sent as a batch of 5:   [0.0893 0.9954 0.9277 0.0297 0.1222]
```

Identical rows, different predictions, determined by who else was in the request.
That is train/serve skew, shipped.

And `qcut` is not special. `df["x"].rank(pct=True)`, `x / x.mean()`,
`(x - x.mean()) / x.std()`, `value_counts`, `factorize` — all pure pandas, all
reading the whole column. Blocking sklearn does nothing about any of them,
because the batch statistic *is* the learned parameter and no library was
involved in learning it.

## The decision

**Do not enumerate the guilty calls — test the property.**

Run the code a second time on a random 25% subset of the rows, then compare the
rows that appear in both runs. Row-local arithmetic cannot tell the difference
and produces identical values. Anything batch-dependent disagrees with itself
immediately. Columns that disagree are named in an error that says what went
wrong and what to use instead.

The error is treated as an argument problem, so the agent gets a turn to fix it
(0007) rather than losing the run.

## Alternatives considered

**Blacklist the dangerous pandas calls.** The first instinct, and unwinnable.
Any aggregation qualifies, `df.eval("x / x.mean()")` hides it in a string,
`getattr(df["x"], "mean")()` hides it further. A blacklist that is 90% complete
provides false confidence, which is worse than none.

**Static analysis of the code.** Same problem, more machinery.

**Give up on re-running and store the columns.** That is reversing 0010, and it
does not work for new rows.

**Accept it and warn.** Rejected: the failure is silent and lands in production.

## Consequences

**Good**

- Complete with respect to the property, regardless of how the code is written.
  It catches spellings nobody thought of, because it never looks at the spelling.
- Verified against the real offending transform: it flags exactly
  `['area_bin', 'price_bin']` and passes the other ~50 columns of that same
  function, including `pd.cut` with literal bin edges — which is stateless and
  correctly allowed.
- The rule it teaches is clean: *no numbers that come from the data without being
  written down.*

**Bad**

- One extra sandbox subprocess per transform at fit time. Nothing at prediction
  time — the replay path passes `check_stability=False`, since a stored transform
  was already checked when it was created.
- It is a **probabilistic** check. A transform whose batch-dependence happens to
  produce identical values on the sampled subset would pass. Unlikely, not
  impossible.
- It rejects some legitimate intent. An agent that genuinely wants
  quantile-binning is told no, and the honest alternative — a stateful binner that
  stores its edges — does not exist yet as an encoder.
- **A guarantee we had asserted in writing turned out to be false.** The
  notebook, the module docstring and the tool docstring all had to be corrected.
  Worth remembering when the next "cannot happen by construction" claim is made.

## Where it lives in the code

- `src/tabular_agent/transform.py` — `unstable_columns`, and
  `transform_features(..., check_stability=True)`.
- `src/tabular_agent/pipeline.py` — the replay path, which does not re-check.
- `tests/test_transform.py` — `qcut` and batch normalisation rejected, row-local
  code accepted.
- `notebooks/phase3_harness_step3_transforms.ipynb` — §3 demonstrates the skew.
