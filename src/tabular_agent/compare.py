"""The uncertainty of a *difference*, which is the quantity decisions are made on.

Every choice a run makes is a comparison: is this table better than the one it
came from. What the harness could put an interval around, until this module
existed, was a *score* — `val_se`, the standard error of one number, obtained by
resampling the rows it was computed on (see `harness/tools._bootstrap_se`).

Those are not the same quantity, and the gap is large. For two candidates scored
on the same rows,

    Var(A - B) = Var(A) + Var(B) - 2*Cov(A, B)

and the covariance is the dominant term, because the two predictors miss the same
rows for mostly the same reasons. Measured on one run (0036): a difference is
**3 to 5 times** better determined than either score it is computed from. Judging
a difference against a pooled `val_se` therefore uses a yardstick several times
too coarse — it calls real effects noise — while judging it against nothing at all
calls noise real. This project has done the second one for its whole history.

## Two modes, because there are two kinds of split

**A random split resamples rows.** Validation rows are exchangeable draws from
one population, so the bootstrap's own assumption holds: resample them with
replacement, rescore *both* candidates on the same resample, and the spread of
the differences is the difference's standard error. Pairing happens because both
candidates see the identical resample — that is the whole mechanism, and it is why
the two candidates' predictions have to be kept rather than their scores.

**An ordered split cannot.** Under `split_on` the validation half is a later
*period*, deliberately chosen to differ from the training rows. Resampling rows
inside it measures how much the difference wobbles within that period while saying
nothing about the thing that actually varies, which is *which* period it is — the
reason `val_se` is refused outright on ordered splits. A too-small yardstick is
worse than none.

So on ordered data the difference is recomputed on each of several **consecutive
blocks** of the validation half, and what is reported is how it varies across
them. That measures the period-level component directly instead of assuming it
away, and it is the instrument 0058 asked for after a feature was found to decay
with distance from its own freeze — a difference that was real in one block and
gone in the next, which no amount of row resampling inside one block could reveal.

## What the blockwise mode still cannot see

It samples the periods it has, and a run has one validation half. If an effect
changes sign beyond the far edge of that half — because the world moved, or
because a feature's usable range ran out — every block can agree and the effect can
still fail on the rows that matter. `sign_stable` says the difference held across
the blocks that were measured. It does not say it will hold at the next horizon,
and that limit is structural, not a matter of taking more blocks.

The honest summary of both modes: they measure how much of an observed difference
is attributable to the evaluation rows. Neither is a significance test, and
nothing here should acquire a threshold without evidence that the threshold
separates differences that survive from differences that do not — 0034's refusal
to fit a constant to three observations applies here with full force.
"""

from __future__ import annotations

import numpy as np

from .metrics import compute_metric, greater_is_better

__all__ = [
    "BLOCK_MIN_ROWS",
    "DEFAULT_BLOCKS",
    "PAIRED_RESAMPLES",
    "blockwise_difference",
    "paired_difference",
]

#: Resamples behind a paired standard error. Larger than `val_se`'s 200 because
#: the quantity is smaller — a difference of 0.003 against scores of 0.15 — and
#: the loop is still milliseconds against the seconds a fit costs.
PAIRED_RESAMPLES = 400

#: Consecutive blocks the validation half is cut into on an ordered split. Five
#: is a compromise with no data behind it and is stated as such: enough to see a
#: difference change sign partway through, few enough that each block still holds
#: a scorable number of rows. It is an argument of the function, not a constant to
#: tune against any dataset.
DEFAULT_BLOCKS = 5

#: A block smaller than this is not scored at all. Not a statistical threshold —
#: it is where `rmspe` and friends stop being computable at all on a handful of
#: rows, and a block that returns a wild number because it held nine rows would
#: dominate a spread it has no business influencing.
BLOCK_MIN_ROWS = 30


def _gain(score: float, reference: float, metric: str) -> float:
    """Positive is better, whatever the metric's own direction.

    Deliberately duplicated from `harness/tools._gain` rather than imported: this
    module is library-level and the harness imports *it*, so taking the dependency
    the other way would be a cycle. The two must agree, and a test asserts they do.
    """
    delta = score - reference
    return (delta if greater_is_better(metric) else -delta) + 0.0


def paired_difference(
    metric: str,
    y,
    preds_a,
    preds_b,
    labels: list | None = None,
    seed: int = 0,
    n_resamples: int = PAIRED_RESAMPLES,
) -> dict | None:
    """How well determined is "A is better than B", by resampling the rows.

    ``preds_a`` and ``preds_b`` must be predictions for the *same* rows in the
    same order — that is what makes the comparison paired, and misaligning them
    would silently produce an interval for a comparison nobody made.

    The gain is reported in `_gain`'s convention: positive means A is better.
    ``paired_se`` is the standard deviation of that gain across resamples, and
    ``independent_se`` is what it would have been had the two scores been
    unrelated. Their ratio is how much the pairing bought, which is the number
    0036 measured at 0.18-0.37 and the reason this module exists; reporting it
    keeps the claim checkable on every dataset instead of resting on one run.

    ``None`` when it cannot be computed — too few rows, or a metric that keeps
    failing on the resamples. A missing interval is handled by the caller; a
    wrong one is not detectable at all.
    """
    y = np.asarray(y)
    a = np.asarray(preds_a)
    b = np.asarray(preds_b)
    n = len(y)
    if not (len(a) == len(b) == n):
        raise ValueError(
            f"paired comparison needs predictions for the same rows: "
            f"got {len(a)} and {len(b)} for {n} targets"
        )
    if n < 10:
        return None

    rng = np.random.default_rng(seed)
    gains, scores_a, scores_b = [], [], []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        try:
            sa = compute_metric(metric, y[idx], a[idx], labels=labels)
            sb = compute_metric(metric, y[idx], b[idx], labels=labels)
        except Exception:
            # A degenerate resample (one class only, typically) is uninformative
            # rather than fatal; enough of them failing is what the check below
            # treats as "no estimate".
            continue
        gains.append(_gain(sa, sb, metric))
        scores_a.append(sa)
        scores_b.append(sb)

    if len(gains) < n_resamples // 2:
        return None

    gains = np.asarray(gains)
    paired_se = float(np.std(gains))
    independent_se = float(np.sqrt(np.var(scores_a) + np.var(scores_b)))
    return {
        "gain": round(float(_gain(
            compute_metric(metric, y, a, labels=labels),
            compute_metric(metric, y, b, labels=labels),
            metric,
        )), 6),
        "paired_se": round(paired_se, 6),
        "independent_se": round(independent_se, 6),
        # How much the pairing bought. Small is good: 0.3 means the difference is
        # determined three times better than two unrelated scores would be.
        "pairing_ratio": (
            None if independent_se == 0 else round(paired_se / independent_se, 3)
        ),
        "n_rows": int(n),
        "n_resamples": int(len(gains)),
        "mode": "paired_bootstrap",
    }


def blockwise_difference(
    metric: str,
    y,
    preds_a,
    preds_b,
    order_values,
    labels: list | None = None,
    n_blocks: int = DEFAULT_BLOCKS,
    min_rows: int = BLOCK_MIN_ROWS,
) -> dict | None:
    """The same difference, recomputed on each consecutive block of the rows.

    For ordered splits, where resampling rows is the wrong operation entirely —
    see the module docstring.

    ``order_values`` is the ordering column itself, and it is **required rather
    than assumed**. The validation half is not stored in its order: `RunSession`
    selects the validation positions in time order and then writes them with
    ``np.sort(val_pos)``, which sorts the *positions* — so the file comes back in
    the input's original row order, whatever that was. On a dataset that happened
    to arrive sorted by date the two coincide, and every block would be correct by
    luck; on one that did not, blocking on file order would cut five arbitrary
    slices, report a reassuring spread, and mean nothing. Sorting here, from the
    column, is the only version that cannot be wrong by accident.

    ``block_gains`` is the difference per block, oldest first, which is the part
    worth reading directly: a feature that decays will show a gain falling across
    the blocks rather than a wide spread around a stable mean, and those two are
    different findings that a single standard deviation cannot tell apart.

    ``sign_stable`` is whether every block agreed on who won. That is the weakest
    useful claim and deliberately so — it is a statement about the blocks that
    were measured, never about the next horizon.
    """
    y = np.asarray(y)
    a = np.asarray(preds_a)
    b = np.asarray(preds_b)
    n = len(y)
    if not (len(a) == len(b) == n):
        raise ValueError(
            f"paired comparison needs predictions for the same rows: "
            f"got {len(a)} and {len(b)} for {n} targets"
        )
    order_values = np.asarray(order_values)
    if len(order_values) != n:
        raise ValueError(
            f"blocking needs one ordering value per row: got {len(order_values)} "
            f"for {n} rows"
        )
    if n_blocks < 2:
        raise ValueError("a difference across blocks needs at least 2 blocks")
    if n // n_blocks < min_rows:
        return None

    # Stable, so rows sharing an ordering value keep their relative order and a
    # block boundary falling inside one date is at least reproducible.
    order = np.argsort(order_values, kind="stable")
    y, a, b = y[order], a[order], b[order]

    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    gains = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi - lo < min_rows:
            continue
        try:
            sa = compute_metric(metric, y[lo:hi], a[lo:hi], labels=labels)
            sb = compute_metric(metric, y[lo:hi], b[lo:hi], labels=labels)
        except Exception:
            continue
        gains.append(_gain(sa, sb, metric))

    if len(gains) < 2:
        return None

    gains = np.asarray(gains)
    overall = _gain(
        compute_metric(metric, y, a, labels=labels),
        compute_metric(metric, y, b, labels=labels),
        metric,
    )
    positive = int(np.sum(gains > 0))
    return {
        "gain": round(float(overall), 6),
        "block_gains": [round(float(g), 6) for g in gains],
        "block_sd": round(float(np.std(gains)), 6),
        # Whether the blocks agreed on a winner at all. An overall gain whose
        # blocks disagree is the shape 0058 found: real in one period, absent in
        # another, and indistinguishable from a solid effect on the pooled number.
        "sign_stable": bool(positive == len(gains) or positive == 0),
        "blocks_favouring_a": positive,
        "n_blocks": int(len(gains)),
        "n_rows": int(n),
        "mode": "ordered_blocks",
    }


def compare_predictions(
    metric: str,
    y,
    preds_a,
    preds_b,
    *,
    order_values=None,
    labels: list | None = None,
    seed: int = 0,
    n_blocks: int = DEFAULT_BLOCKS,
) -> dict | None:
    """Whichever of the two instruments the split calls for.

    ``order_values`` is what selects the mode, rather than a boolean flag: an
    ordered run *has* an ordering column and a random one does not, so passing the
    column is both the evidence and the data the blockwise mode needs. A flag
    would let a caller say "ordered" without supplying the order, which is the one
    combination that cannot be served and would otherwise have to fail at runtime.
    """
    if order_values is not None:
        return blockwise_difference(metric, y, preds_a, preds_b, order_values,
                                    labels=labels, n_blocks=n_blocks)
    return paired_difference(metric, y, preds_a, preds_b, labels=labels, seed=seed)
