"""Deterministic fold generation shared by encoding and training.

Two strategies, and which one is right is a fact about the data rather than a
preference. Both are kept because both are correct for their own case.

**Random k-fold** (the default) assumes rows are exchangeable: any row could have
landed in any fold without changing what the model is being asked to do. True for
a table of independent observations, and it is the lower-variance choice — every
row gets an out-of-fold prediction and every fold trains on the same amount of
data.

**Forward chaining** (``order=``) assumes the opposite: that the rows are ordered
and the task is to predict later ones from earlier ones. Each fold trains on a
prefix and validates on the block that follows, so no model ever sees past its own
prediction horizon.

Getting this wrong is not a small error. On a store-day panel, random folds let a
model predict Wednesday having trained on Tuesday and Thursday *of the same store*
— it interpolates instead of forecasting, scores far too well, and the gap only
shows up against a held-out future. Conversely, forward chaining on exchangeable
rows throws away data and measures nothing extra.
"""

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


def _change_positions(order_sorted):
    """Where the sorted ordering value changes — the only places a cut is honest.

    Splitting a Rossmann panel at an exact row count puts 603 of one day's stores
    on one side and 512 of the *same day* on the other, and rows sharing a day
    share everything day-level — the promo calendar, the weather, the holiday. The
    model learns that day's level from one group and applies it to the other,
    which is the leak an ordered split exists to prevent.
    """
    return np.flatnonzero(order_sorted[1:] != order_sorted[:-1]) + 1


def boundary_cut(order_sorted, target: int, exclude: tuple[int, ...] = ()) -> int:
    """The honest cut position nearest ``target``, for a single split.

    Used by the work/validation split, which wants one boundary rather than a set.
    The realised fraction moves to wherever the nearest value boundary is, which
    is a rounding error on a panel with a thousand rows per day and a coarse jump
    on a column with few distinct values.

    ``exclude`` is for the second cut of a three-way ordered split (0023): a
    boundary already spent on the work/validation edge cannot also be the
    validation/holdout edge, or one of the two slices is empty.
    """
    changes = _change_positions(order_sorted)
    if len(changes) == 0:
        raise ValueError(
            "cannot split on this column: every row shares one value, so any cut "
            "would put that value on both sides"
        )
    available = changes[~np.isin(changes, exclude)] if exclude else changes
    if len(available) == 0:
        raise ValueError(
            f"cannot split on this column: it has {len(changes)} value boundary/ies "
            "and they are already spent on earlier cuts"
        )
    return int(available[np.argmin(np.abs(available - target))])


def _boundary_cuts(order_sorted, n_blocks: int) -> list[int]:
    """Positions splitting sorted ``order_sorted`` into ``n_blocks``, on value ties."""
    n = len(order_sorted)
    changes = _change_positions(order_sorted)
    if len(changes) < n_blocks - 1:
        raise ValueError(
            f"cannot make {n_blocks} ordered blocks from {len(changes) + 1} distinct "
            "values: a block boundary would have to fall inside a group of rows "
            "sharing an ordering value, which would put the same value on both "
            "sides. Use fewer splits, or order on a finer column."
        )

    cuts: list[int] = []
    for i in range(1, n_blocks):
        target = round(n * i / n_blocks)
        # Nearest honest boundary to where an equal-sized cut would have gone,
        # and never one already used — two cuts at the same place is an empty
        # block, which is a silent way to lose a fold.
        available = changes[~np.isin(changes, cuts)]
        cuts.append(int(available[np.argmin(np.abs(available - target))]))
    return sorted(cuts)


def make_folds(y, task_type: str, n_splits: int = 5, seed: int = 42, order=None):
    """Train/validation index pairs.

    ``order`` switches strategy. Left ``None``, folds are random (stratified for
    classification) and every row is validated exactly once. Given an array of
    sortable values — dates, usually — folds are built by forward chaining
    instead, and three things change that callers have to know about:

    * There are ``n_splits`` folds over ``n_splits + 1`` blocks, so **the first
      block is never validated**. Nothing precedes it to train on.
    * Training sets grow: the first fold trains on one block, the last on
      ``n_splits``. Later folds score better partly because they saw more data, so
      the spread across folds mixes noise with the value of more history.
    * Consequently **not every row gets an out-of-fold prediction**. Callers that
      assume one per row — ``oof.parquet``'s row alignment, out-of-fold target
      encoding — have to carry NaN through rather than treat a missing prediction
      as a zero.
    """
    y = np.asarray(y)
    if order is not None:
        order = np.asarray(order)
        if len(order) != len(y):
            raise ValueError(
                f"order has {len(order)} values for {len(y)} rows; it must be the "
                "ordering column of the same table"
            )
        # Stable, so rows sharing a value keep their original relative order and
        # the folds are reproducible rather than dependent on sort internals.
        pos = np.argsort(order, kind="stable")
        cuts = _boundary_cuts(order[pos], n_splits + 1)
        blocks = np.split(pos, cuts)
        return [
            (np.concatenate(blocks[: i + 1]), blocks[i + 1])
            for i in range(n_splits)
        ]

    if task_type in ("binary", "multiclass"):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [(tr, va) for tr, va in splitter.split(np.zeros(len(y)), y)]
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in splitter.split(np.zeros(len(y)))]
