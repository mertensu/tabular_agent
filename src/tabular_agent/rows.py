"""Row selection: the one pipeline step that is not replayed at predict time.

Every other step in a :class:`~tabular_agent.pipeline.Pipeline` answers "how do I
rebuild this column from raw input", and is therefore re-run whenever a
prediction is asked for. A row filter answers a different question — *which rows
should this model learn from* — and re-running it at predict would be wrong twice
over:

* **Scoring.** The validation half is scored by replaying the winning table's
  lineage over raw validation rows and lining the predictions up against the held
  back targets. A filter in that replay would drop some of those rows and
  misalign the two.
* **Serving.** Every row handed to a deployed model needs a prediction back.
  A model that silently declines to answer for 17% of requests is not a model,
  and nothing downstream would know which 17%.

So a filter is applied **once**, when the derived table is written, and marked
``train_only`` in the lineage so replay skips it. The rows are simply not in the
table the model trains on.

Two things follow that are easy to get backwards:

* **Batch dependence does not matter here.** ``transform_features`` has to reject
  ``x > x.mean()`` because the stored *code* is recomputed on whatever rows
  arrive at predict. A filter's code is never recomputed, so a batch-dependent
  predicate is evaluated exactly once against exactly one frame, and the notion
  of disagreeing with itself does not arise. This is the only agent-written code
  in the harness that needs no stability check.
* **The target is visible to it, and that is not leakage.** Dropping training
  rows by their target value — outlier removal — is an ordinary modelling choice.
  It cannot leak, because nothing the filter produces is ever evaluated on a row
  the model will be scored on: the validation half is untouched and the filter
  does not run there. Contrast ``transform_features``, where the target is
  withheld precisely because its output *is* replayed on rows that have no
  target.

What a filter does change is the meaning of ``cv_score``, which is now measured
over the kept rows only. ``val_score`` is measured on the full validation half
either way, which is what keeps a filtered and an unfiltered experiment
comparable at all.
"""

import pandas as pd

from .sandbox import SandboxError, run_filter

__all__ = ["select_rows", "SandboxError"]


def select_rows(df: pd.DataFrame, code: str, *, timeout: float = 30.0):
    """Run agent-written ``keep_rows(df) -> bool Series``; return ``(kept, summary)``.

    ``kept`` is ``df`` restricted to the rows the mask marked ``True``, with the
    original index preserved, so anything already aligned to those rows still
    lines up. ``summary`` reports ``n_rows`` (before), ``n_kept`` and
    ``n_dropped``.

    Raises :class:`SandboxError` on blocked imports, timeouts, a non-boolean or
    misindexed return, or an exception in the supplied code. Deciding whether the
    *number* of surviving rows is acceptable is the caller's — it depends on the
    fold count, which lives in the task spec rather than here.
    """
    mask, summary = run_filter(df, code, func_name="keep_rows", timeout=timeout)
    kept = df.loc[mask]
    return kept, {
        "n_rows": int(summary["n_rows"]),
        "n_kept": int(summary["n_kept"]),
        "n_dropped": int(summary["n_rows"] - summary["n_kept"]),
    }
