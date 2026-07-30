"""The stateless feature-transform tool ``transform_features`` (spec §5.1).

The persisting sibling of :func:`~tabular_agent.explore.explore_dataframe`. Both
let the agent run its own pandas/numpy code in the same sandbox
(:mod:`.sandbox`); the difference is what happens to the output:

- ``explore`` returns a **summary** that is shown to the agent and discarded —
  read-only, nothing it computes becomes a feature.
- ``transform_features`` (here) returns a **DataFrame** that *does* become
  features. So it is persisted as an ordered pipeline step
  (:class:`~tabular_agent.pipeline.Pipeline`) and simply **re-run at predict**.

Why re-running (rather than storing the produced columns) is leakage-safe: the
contract is ``df -> df`` with the **row index preserved** — row-local math with
no learned parameter (``dist = sqrt(x**2 + y**2)``, ratios, per-row counts, date
parts). Because it is stateless, replaying the exact code at predict time is
trivially reproducible and cannot skew train vs. serve.

Statelessness is enforced by **two** mechanisms, because the obvious one is not
enough. The sandbox blocks the model/sklearn libraries, so nothing here can fit
an estimator. But plain pandas is perfectly capable of learning from the batch —
``pd.qcut`` takes quantiles of the column it is handed, ``x / x.mean()`` divides
by the batch mean — and only the *code* is stored, so replay recomputes those
from whatever rows arrive at predict time. Hence :func:`unstable_columns`, which
re-runs the code over a subset and rejects any column that disagrees with itself.

Anything that genuinely must learn a parameter from the training set (a category's
target mean, a vocabulary) belongs in the structured, per-fold
:func:`~tabular_agent.encoders.fit_encoding` tool, which fits once and stores what
it learned.
"""

import numpy as np
import pandas as pd

from .sandbox import SandboxError, run_transform

__all__ = ["transform_features", "SandboxError"]


def _same_values(a: pd.Series, b: pd.Series) -> bool:
    """Equality that treats NaN as equal to NaN and tolerates float noise."""
    if a.dtype != b.dtype and not (
        pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)
    ):
        return False
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        return bool(
            np.allclose(
                a.to_numpy(dtype="float64"),
                b.to_numpy(dtype="float64"),
                equal_nan=True,
                rtol=1e-9,
                atol=1e-12,
            )
        )
    return bool(a.reset_index(drop=True).equals(b.reset_index(drop=True)))


def unstable_columns(df, code, out_full, *, timeout=30.0, seed=0):
    """Columns whose value depends on *which other rows* were in the frame.

    The blocked imports stop the code from fitting a model or an sklearn encoder,
    but they do nothing about plain pandas: ``pd.qcut`` takes quantiles of the
    column it is given, ``x / x.mean()`` divides by the batch mean, ``rank``
    ranks within the batch. Those are learned parameters in everything but name,
    and because only the *code* is stored, replaying it at predict time
    recomputes them from whatever rows happen to be in that request. The same
    row then scores differently depending on what it was batched with.

    Enumerating the guilty pandas calls is unwinnable — any aggregation is one.
    So this checks the property instead of the syntax: run the same code over a
    random subset and compare the rows that appear in both. Row-local arithmetic
    cannot notice the difference; anything batch-dependent will disagree.
    """
    n = len(df)
    if n < 4:
        return []
    subset = df.sample(n=max(2, n // 4), random_state=seed).sort_index()
    out_subset, _ = run_transform(subset, code, func_name="transform_features", timeout=timeout)

    bad = [c for c in out_full.columns if c not in out_subset.columns]
    bad += [c for c in out_subset.columns if c not in out_full.columns]
    for col in out_subset.columns:
        if col in out_full.columns and not _same_values(
            out_full.loc[subset.index, col], out_subset[col]
        ):
            bad.append(col)
    return sorted(set(bad))


def transform_features(df, code, *, timeout=30.0, check_stability=True):
    """Run stateless ``df -> df`` ``code`` against ``df``; return ``(df, summary)``.

    ``code`` must define ``transform_features(df)`` returning a DataFrame with the
    same row index as the input (add/replace columns; do not reorder or drop
    rows). Example::

        code = '''
        import numpy as np
        def transform_features(df):
            df = df.copy()
            df["dist"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
            df["ratio"] = df["a"] / (df["b"] + 1)
            return df
        '''
        out, summary = transform_features(raw_df, code)

    ``summary`` reports ``n_rows``, ``columns`` and ``new_columns``. Raises
    :class:`SandboxError` on blocked imports, timeout, contract violations, or
    user errors.

    ``check_stability`` re-runs the code over a subset and rejects columns whose
    values depend on the rest of the batch (see :func:`unstable_columns`) — the
    hole that blocking the model libraries does not cover. It costs one extra
    sandbox call at fit time and nothing at predict.
    """
    out, summary = run_transform(df, code, func_name="transform_features", timeout=timeout)
    if check_stability:
        unstable = unstable_columns(df, code, out, timeout=timeout)
        if unstable:
            raise SandboxError(
                f"these columns depend on which other rows are in the frame: {unstable}. "
                "The same rows produced different values when transformed as part of a "
                "smaller batch, so at predict time — where the batch is whatever arrives, "
                "sometimes a single row — they would not match what the model trained on. "
                "Causes are pandas operations that read the whole column: pd.qcut, rank, "
                "and anything dividing by df[...].mean()/std()/max(). Use fixed thresholds "
                "instead (pd.cut with explicit bin edges), or fit_encoding, which learns "
                "its parameters once and stores them."
            )
    return out, summary
