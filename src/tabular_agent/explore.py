"""The read-only exploration tool (``run_python`` in MCP terms).

A companion to ``transform_features`` (§5.1). Both let the agent run its own
pandas/numpy code in the same sandbox (:mod:`.sandbox`); the difference is what
happens to the output:

- ``transform_features`` returns ``df -> df`` and its result is **persisted as a
  pipeline step** and re-run at ``predict`` — it feeds the model.
- ``explore`` (here) returns a **summary that is shown to the agent and then
  discarded**. It is *read-only*: nothing it computes becomes a model feature.

That non-persistence is the structural guard the design cares about. It lets the
agent freely check hypotheses ("does churn rate differ by city?", "how skewed is
income?") — including target statistics per category — without reopening the
target-leakage backdoor that ``fit_encoding`` closes. A per-category target mean
looked at here cannot be pasted back in as a hand-built feature, because the
tool has no path to persist it (and the result is size-capped, so it cannot even
be smuggled back as a full-length column).

This is also why the distribution stats (skew, quantiles) and group-by tables
are *not* baked into :func:`~tabular_agent.inspect.profile_dataframe`: the agent
computes exactly the ones it needs, on demand, here.
"""

from .sandbox import SandboxError, run_readonly

__all__ = ["explore_dataframe", "SandboxError"]


def explore_dataframe(df, code, *, timeout=10.0, max_rows=200):
    """Run read-only exploration ``code`` against ``df`` and return its summary.

    ``code`` must define ``explore(df)`` returning a small, JSON-serializable
    result (scalar, dict, or a capped Series/DataFrame). Example::

        code = '''
        def explore(df):
            return df.groupby("city")["churn"].mean().to_dict()
        '''
        explore_dataframe(df, code)

    Raises :class:`SandboxError` on blocked imports, timeout, or user errors.
    """
    return run_readonly(df, code, func_name="explore", timeout=timeout,
                        max_rows=max_rows)
