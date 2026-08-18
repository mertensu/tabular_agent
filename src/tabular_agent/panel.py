"""Several disagreeing evaluation geometries, because one ruler has been wrong.

## What this is for

Every table decision a run makes rests on one number: the probe's score on the
run's single validation half. That number has now been measured wrong four times
(0056, 0065, 0066, 0073), and the failures share a shape — the ruler was not
undecided, it was **confident and pointing the wrong way**.

`compare.py` was the first response: put an interval around the *difference*
rather than around the score. It works, and its own docstring names what it still
cannot see:

    It samples the periods it has, and a run has one validation half. If an
    effect changes sign beyond the far edge of that half [...] every block can
    agree and the effect can still fail on the rows that matter.

That is exactly what happened on Restaurant (0065/0066: `sign_stable: True` at 2.9
block sd, and the step lost privately) and on IEEE-CIS, where 18 of 20 comparisons
came back unstable and nine were accepted as improvements anyway.

This module varies something `compare.py` holds fixed. Blocking re-measures **the
same fit** on different evaluation rows. A panel refits on **different training
periods, at different distances, with different gaps** — so the thing that varies
is the forecasting problem itself, which is the thing the run is actually being
graded on.

## Why several rather than one better one

The instinct after four failures is to build a more trustworthy ruler. The
IEEE-CIS winners, facing the same problem on the same kind of data, did the
opposite and said so plainly:

    We never trusted a single validation strategy so we used lots of validation
    strategies. Train on first 4 months of train, skip a month, predict last
    month. We also did train 2, skip 2, predict 2. We did train 1 skip 4 predict
    1. [...] We did a CV GroupKFold using month as the group.

The four ordered schemes below are that set. What they buy is not precision — any
one of them is noisier than the run's own validation score, because each trains on
a fraction of the data. What they buy is **disagreement**: a gain that survives
being measured four different ways is a different kind of object from one that
does not, and no single ruler, however well built, can tell them apart.

## What a scheme measures, and the one thing it cannot

A scheme is a pair of row masks over one table: fit here, score there. Both come
from the table's own rows, so no lineage replay is involved and the panel is cheap
(measured on this repo's widest table, 531,486 x 489: four schemes cost **2.4x**
one full-data fit, because each trains on a fraction of the rows).

**The known limitation, stated rather than hidden.** The table's encoders were
fitted on the whole work half before this module sees it. A scheme that evaluates
on the last block is therefore scoring rows whose *features* were built with
knowledge of that block's targets — out-of-fold per row (0032), but in-sample for
the block. Every scheme shares this, and so does the parent table it is compared
against, so it does not obviously bias the *difference*. But "every window shares
the defect" is precisely how 0066's leaky column passed three instruments, and
this module is not exempt from that. A panel is a wider net, not a proof.

## Ordered and unordered runs measure different disagreements

With `split_on`, the schemes are time geometries and the disagreement is about
**horizon**: does this gain survive being asked to forecast further, from less
history, across a gap.

Without one there is no horizon, and a time geometry would be a fiction cut on row
order that means nothing. The schemes instead vary **which rows were trained on**
— repeated subsamples, each scored on the rows it did not see. The question
becomes "does this gain survive a different fit", which is the part of the idea
that transfers to exchangeable data. It is a weaker instrument and it is named
differently (`subsample` rather than `horizon`) so that a reader of the journal is
never in doubt about which question was answered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import compute_metric, greater_is_better
from .probe import fit_probe, probe_predict

__all__ = [
    "MIN_SCHEME_ROWS",
    "N_BLOCKS",
    "Scheme",
    "ordered_schemes",
    "panel_agreement",
    "scheme_scores",
    "subsample_schemes",
    "work_block_edges",
]

#: The work half is cut into six blocks, because the winners' schemes are stated
#: in sixths of a six-month training file ("train 4, skip 1, predict 1") and
#: because six is the smallest number that expresses all four of them. It is not
#: a tuned constant and nothing here should acquire one without evidence that it
#: separates gains that survive from gains that do not (0034).
N_BLOCKS = 6

#: A scheme with fewer rows than this on either side is not run at all. Same
#: reasoning as `compare.BLOCK_MIN_ROWS`: not a statistical threshold, just the
#: point below which a metric stops being computable and a wild number would
#: dominate an agreement count it has no business influencing.
MIN_SCHEME_ROWS = 30


@dataclass
class Scheme:
    """One evaluation geometry: fit on ``train``, score on ``evaluate``.

    Masks rather than indices, so a table with a different number of rows than the
    one the edges were computed from (a row filter ran) still selects the same
    *period* rather than the same positions.
    """

    name: str
    kind: str
    train: np.ndarray
    evaluate: np.ndarray

    @property
    def usable(self) -> bool:
        return (
            int(self.train.sum()) >= MIN_SCHEME_ROWS
            and int(self.evaluate.sum()) >= MIN_SCHEME_ROWS
        )

    def describe(self) -> dict:
        return {
            "scheme": self.name,
            "kind": self.kind,
            "n_train": int(self.train.sum()),
            "n_eval": int(self.evaluate.sum()),
        }


def work_block_edges(order_values, n_blocks: int = N_BLOCKS) -> list[float]:
    """The block boundaries, as values of the ordering column.

    **Computed once per run from the work table and reused for every table**, which
    is the whole reason this is a separate function. Recomputing quantiles per
    table would give a filtered table different boundaries from its parent, and the
    two would then be compared across different periods while appearing to be
    compared across the same ones — a confound that would look exactly like a
    feature effect.

    Returned as values rather than positions for the same reason: a value survives
    a row filter, a position does not.
    """
    values = pd.Series(np.asarray(order_values)).dropna()
    if values.empty:
        return []
    quantiles = np.linspace(0, 1, n_blocks + 1)[1:-1]
    # `method="lower"` so every boundary is a value that actually occurs. The
    # default interpolates, and on a column with many ties that invents
    # boundaries between two real values — 200 rows each of 1, 2 and 3 produce
    # edges at 1.997 *and* 2.0, and the block between them holds nothing. A
    # near-empty block is worse than a missing one: it is scored, it is wild, and
    # it votes.
    raw = np.quantile(values.to_numpy(), quantiles, method="lower")
    return [float(v) for v in np.unique(raw)]


def _block_index(order_values, edges: list[float]) -> np.ndarray:
    """Which block each row falls in, given boundaries measured elsewhere.

    ``side="left"`` so a row sitting exactly on a boundary joins the block that
    boundary *opens*. With `method="lower"` edges the boundary is itself an
    observed value, so the alternative would shave one row off every block and
    make the six unequal for no reason.
    """
    return np.searchsorted(np.asarray(edges, dtype=float),
                           np.asarray(order_values, dtype=float), side="left")


def ordered_schemes(order_values, edges: list[float]) -> list[Scheme]:
    """The four time geometries, in blocks of the work half.

    Transcribed from the IEEE-CIS winners' description and nothing more — the
    numbers are theirs, and the fourth is added because it is the geometry the
    run's *own* validation score most resembles, so the panel always contains one
    scheme that should agree with the headline number when nothing is wrong.

    | scheme | train | gap | evaluate |
    |---|---|---|---|
    | `train4_gap1_eval1` | blocks 1-4 | block 5 | block 6 |
    | `train2_gap2_eval2` | blocks 1-2 | blocks 3-4 | blocks 5-6 |
    | `train1_gap4_eval1` | block 1 | blocks 2-5 | block 6 |
    | `train5_gap0_eval1` | blocks 1-5 | — | block 6 |

    The gap is the part that matters and the part no other instrument here has.
    A run is graded across a real gap (IEEE-CIS's is thirty days), and a ruler that
    trains up to the moment it predicts is answering an easier question than the
    one being asked — measured on this repo's widest table, the no-gap scheme
    scores 0.921 where `train1_gap4_eval1` scores 0.834 on the identical columns.
    """
    if not edges:
        return []
    block = _block_index(order_values, edges)
    # Counted from the blocks that actually hold rows, never from `len(edges)`.
    # An edge equal to the largest value opens a block nothing can fall into, and
    # a geometry that evaluated on it would score zero rows while looking valid.
    populated = np.unique(block)

    def rng(lo: int, hi: int) -> np.ndarray:
        """Blocks ``lo``..``hi`` inclusive, one-based, as a mask."""
        return (block >= lo - 1) & (block <= hi - 1)

    if len(populated) < N_BLOCKS:
        # The ordering column is too coarse to cut six ways — many ties, or few
        # distinct values. The four geometries below would silently overlap or
        # evaluate on nothing, so one honest scheme is reported instead of four
        # dishonest ones, and `panel_agreement` then returns None because a single
        # scheme is not a panel.
        return [Scheme(
            "train_early_eval_late", "horizon",
            np.isin(block, populated[:-1]), block == populated[-1],
        )]

    return [
        Scheme("train4_gap1_eval1", "horizon", rng(1, 4), rng(6, 6)),
        Scheme("train2_gap2_eval2", "horizon", rng(1, 2), rng(5, 6)),
        Scheme("train1_gap4_eval1", "horizon", rng(1, 1), rng(6, 6)),
        Scheme("train5_gap0_eval1", "horizon", rng(1, 5), rng(6, 6)),
    ]


def subsample_schemes(n_rows: int, seed: int, n_schemes: int = 4,
                      train_fraction: float = 0.6) -> list[Scheme]:
    """Repeated fits on different halves, for runs with no ordering column.

    Each scheme trains on a random ``train_fraction`` of the rows and scores on the
    rest, so the two sides never overlap and the question is "does this gain
    survive a different fit". The seeds are derived from the run's seed, so a
    panel is reproducible and two tables in the same run are compared on the
    *identical* subsamples — which is what makes the per-scheme difference paired
    (`compare.py`'s argument, applied to fits rather than to resampled rows).
    """
    schemes = []
    for i in range(n_schemes):
        rng = np.random.default_rng(seed + 1000 * (i + 1))
        train = np.zeros(n_rows, dtype=bool)
        train[rng.choice(n_rows, size=int(n_rows * train_fraction), replace=False)] = True
        schemes.append(Scheme(f"subsample{i + 1}", "subsample", train, ~train))
    return schemes


def scheme_scores(
    X: pd.DataFrame,
    y,
    schemes: list[Scheme],
    task_type: str,
    metric: str,
    target_transform: str | None = None,
) -> dict[str, float]:
    """Fit and score one table under each scheme. Missing keys mean "not run".

    A scheme that raises is dropped rather than recorded as a failure: the panel is
    an agreement count, and a scheme that could not be computed has no opinion. It
    must not be allowed to read as a disagreement, which would turn a fit that
    crashed into evidence against a feature.
    """
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    X = X.reset_index(drop=True)
    out: dict[str, float] = {}
    for scheme in schemes:
        if not scheme.usable:
            continue
        try:
            fit = fit_probe(
                X.loc[scheme.train], y.loc[scheme.train], task_type, metric,
                target_transform,
            )
            preds = probe_predict(fit, X.loc[scheme.evaluate], task_type)
            labels = (
                sorted(pd.unique(y)) if task_type in ("binary", "multiclass") else None
            )
            kwargs = {"labels": labels} if labels is not None else {}
            score = compute_metric(
                metric, y.loc[scheme.evaluate].to_numpy(), preds, **kwargs
            )
        except Exception:  # noqa: BLE001 - see the docstring: no opinion, not a vote
            continue
        if score is not None and np.isfinite(score):
            out[scheme.name] = float(score)
    return out


def panel_agreement(
    child: dict[str, float],
    parent: dict[str, float],
    metric: str,
    headline_gain: float | None = None,
) -> dict | None:
    """Does the headline gain survive being measured several other ways?

    ``agrees`` is **all or nothing** by design. With four schemes a majority rule
    would let a step through on 3-1, and the whole reason this exists is that the
    project has repeatedly accepted steps that most of its instruments liked. The
    cost of being strict is bounded: under the soft gate this only decides whether
    a step counts as *progress*, never whether it may win (0035's argmax is
    untouched), so a false negative spends patience rather than losing a table.

    ``None`` when fewer than two schemes ran on both tables — a single scheme is
    not a panel, and reporting one as though it were would be worse than silence.
    """
    shared = sorted(set(child) & set(parent))
    if len(shared) < 2:
        return None
    higher_better = greater_is_better(metric)
    gains = {}
    for name in shared:
        delta = child[name] - parent[name]
        gains[name] = float(delta if higher_better else -delta)

    reference = headline_gain if headline_gain is not None else float(
        np.mean(list(gains.values()))
    )
    # A headline of exactly zero has no sign to agree with; nothing is claimed, so
    # nothing has to survive.
    if reference == 0:
        return {
            "n_schemes": len(shared),
            "n_agree": 0,
            "agrees": False,
            "gains": {k: round(v, 6) for k, v in gains.items()},
            "note": "the headline gain is zero, so there is no sign to confirm",
        }
    agreeing = [n for n, g in gains.items() if g != 0 and (g > 0) == (reference > 0)]
    return {
        "n_schemes": len(shared),
        "n_agree": len(agreeing),
        "agrees": len(agreeing) == len(shared),
        "disagreeing": [n for n in shared if n not in agreeing],
        "gains": {k: round(v, 6) for k, v in gains.items()},
        "worst": round(min(gains.values()) if reference > 0 else max(gains.values()), 6),
    }
