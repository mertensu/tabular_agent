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
trivially reproducible and cannot skew train vs. serve. Anything that must *learn*
a parameter from the training set (a category's target mean, a vocabulary) is
deliberately **not** expressible here — the sandbox blocks the model/sklearn libs
— and belongs in the structured, per-fold :func:`~tabular_agent.encoders.fit_encoding`
tool instead, which is what keeps target leakage out.
"""

from .sandbox import SandboxError, run_transform

__all__ = ["transform_features", "SandboxError"]


def transform_features(df, code, *, timeout=30.0):
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
    """
    return run_transform(df, code, func_name="transform_features", timeout=timeout)
