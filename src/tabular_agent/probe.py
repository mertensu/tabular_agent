"""The probe: one fixed estimator used as a ruler, so tables can be compared.

A run's leaderboard used to rank *experiments* — a model, its parameters and a
table, all varying at once — which meant that when one row beat another there was
no way to say which of the three was responsible. This module holds the other
half of that fix (0035): a single estimator, with fixed parameters and a fixed
seed, applied unchanged to every table a run produces.

It is a **measuring instrument, not a candidate.** Nothing here is meant to be
shipped. Its only job is to make the statement "t7 is a better table than t6"
mean something, and it does that by holding everything except the table still.

Two properties are load-bearing, and both are easy to break by accident:

**The seed is a module constant, not the run's seed.** A ruler that changes
between runs cannot compare across runs, and comparing runs is most of what this
project does with its datasets.

**The parameters are not tuned.** The probe being a *good* model is not the
point and would be a trap: tuning it against any dataset would make the ruler a
function of that dataset, which is the failure mode `docs/decisions/0022`
records. It only has to rank.

## Why one fit, and no cross-validation

The number tables are ranked by is the score on the run's validation half, which
is the same rows for every table — that constancy is the whole basis of the
comparison. A row filter changes how many rows a *table* has (see `rows.py`), so
a per-table cross-validation is not measured over a common row set and cannot be
compared across filtered and unfiltered tables; `val_score` can.

That score comes from a model fitted on all of the table's rows, so the number of
folds never enters it. The probe therefore does **one** fit per table rather than
``k + 1``, which is what makes it cheap enough to run on every table a run
creates rather than only the ones an agent chooses to spend an experiment on.

Cross-validation has not gone away — the terminal fit still needs out-of-fold
predictions, and ablation credit (0017) is defined in terms of them. It is simply
not on the ranking path.

## Why a noisy estimator is acceptable here

Measured against Kaggle, `val_score` is the worse of the two numbers a run
computes: error spread 0.0124 against cross-validation's 0.0043 (0034). That is a
statement about a score's distance from the truth *in absolute terms*, and it is
not the quantity this module is used for.

Fix the estimator and fix the evaluation rows, and two candidates' errors become
highly correlated — the same rows, missed for mostly the same reasons. What is
wanted is the *difference*, and

    Var(A - B) = Var(A) + Var(B) - 2*Cov(A, B)

so a large covariance makes the difference far better determined than either
score. Measured by paired bootstrap on one run (0036): a difference is **3 to 5
times** better determined than the scores it is computed from. The shared
component of the noise — "this validation half is a little easier than the world"
— is exactly what cancels, and it is also the component that makes the absolute
number untrustworthy.

**That benefit is not unique to comparing tables.** 0035 assumed a fixed
estimator was what bought the pairing, and 0036 measured otherwise: swapping
lgbm for catboost leaves the comparison about as precise as swapping one table
for another, because what governs the covariance is how similar the two
*predictors* are, not which knob was turned. Our own residual correlations
between boosted-tree families, 0.86-0.88, said so before this module was written.

So the probe is not here to make comparisons paired. It is here for three duller
reasons:

* **Coverage.** A run scores the tables it chose to spend an experiment on and
  leaves the rest blank — 8 of v8's 16 tables. One cheap ruler applied to every
  table fills that in, which is what makes recombined and intermediate tables
  comparable to the branches they came from at all.
* **One confound removed.** Without a fixed estimator, "is t16 better than t10"
  is answered by whichever model each happened to be trained with. 0017 already
  requires that a comparison differ by exactly one thing; this applies that rule
  to selection rather than to attribution.
* **Cost.** One fit per table, per the section above, which is what makes
  scoring everything affordable rather than a luxury.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import SUPPORTED_METRICS
from .models import _make_model, check_objective, poisson_fit_rows, predict_model
from .target import apply_target, check_target_domain

__all__ = [
    "PROBE_MODEL_TYPE",
    "PROBE_PARAMS",
    "PROBE_SEED",
    "ProbeFit",
    "default_target_transform",
    "describe_probe",
    "fit_probe",
]

#: The ruler. A gradient-boosted tree because that is the family the terminal fit
#: is most likely to choose from, and a probe from a different family would rank
#: tables by what suits an estimator nobody ships (0035 names this cost).
PROBE_MODEL_TYPE = "lgbm"

#: Untuned on purpose — see the module docstring. These are the library's own
#: defaults, which means historical experiments that ran `lgbm` at defaults are
#: directly comparable to a probe score on the same table.
PROBE_PARAMS: dict = {"n_estimators": 200}

#: Fixed, and deliberately not ``TaskSpec.seed``: the ruler must not vary between
#: runs, or cross-run comparison measures the seed as much as the change.
PROBE_SEED = 0


def default_target_transform(metric: str, task_type: str) -> str:
    """The scale the target is fitted on, derived from the metric (0035).

    0021 made this a per-experiment choice so that nobody assumed, at the top of a
    run, that logging helps. The assumption was then measured and it held; and
    0022 supplies the general reason, which is that a model fitted under one
    objective while the run is ranked under another is optimising something nobody
    is scoring.

    Under `rmsle` the derivation is exact: the metric *is* squared error on
    ``log1p(y)``, so fitting on ``log1p`` makes the model's own loss the run's
    metric. `rmspe` is the approximate case — squared error on a log scale is a
    first-order approximation of squared *relative* error, which is what `rmspe`
    measures — and it is included because the alternative, raw squared loss, is
    not an approximation of it at all.

    Everything else gets ``none``, because nothing derives it. That includes
    `mae`, where a log fit is often the right call and is exactly the kind of
    knowledge that belongs to whoever owns the data: `TaskSpec` overrides this,
    and the library does not guess.
    """
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"unknown metric: {metric}")
    if task_type != "regression":
        return "none"
    return "log1p" if metric in ("rmsle", "rmspe") else "none"


@dataclass
class ProbeFit:
    """A fitted probe, plus what is needed to score and replay it."""

    model: object
    feature_names: list[str]
    target_transform: str
    #: Sorted class labels for classification, so predictions keep a stable
    #: column order even when a fit saw fewer classes than the run has.
    classes: np.ndarray | None = None
    fit_seconds: float = 0.0
    spec: dict = field(default_factory=dict)
    #: Rows the fit skipped: negative counts under a Poisson objective.
    unfitted_rows: int = 0


def describe_probe() -> dict:
    """What the ruler is, for the journal — so a score can be interpreted later.

    Recorded per run rather than assumed: if these constants ever change, scores
    from before and after are not comparable, and the journal is the only place
    that fact can survive.
    """
    return {
        "model_type": PROBE_MODEL_TYPE,
        "params": dict(PROBE_PARAMS),
        "seed": PROBE_SEED,
    }


def fit_probe(
    X: pd.DataFrame,
    y,
    task_type: str,
    metric: str,
    target_transform: str | None = None,
    objective: str | None = None,
) -> ProbeFit:
    """Fit the ruler on one table. One fit, no folds — see the module docstring.

    ``target_transform`` defaults to :func:`default_target_transform`; pass it
    explicitly to honour a `TaskSpec` override. It changes what the model is
    fitted on and never what is reported: callers invert before scoring, exactly
    as ``train_cv`` does.

    ``objective`` is the run's regression loss. The ruler fits with it, so a
    Poisson run ranks its tables by what a Poisson model can make of them.
    """
    import time

    if target_transform is None:
        target_transform = default_target_transform(metric, task_type)

    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    check_target_domain(target_transform, y, task_type)
    y_fit = pd.Series(apply_target(target_transform, y)) if target_transform != "none" else y
    check_objective(objective, y_fit, target_transform)
    fittable = poisson_fit_rows(objective, y_fit)

    classes = np.unique(y.to_numpy()) if task_type in ("binary", "multiclass") else None

    started = time.monotonic()
    model = _make_model(
        PROBE_MODEL_TYPE, task_type, dict(PROBE_PARAMS), PROBE_SEED, metric, objective
    )
    model.fit(X[fittable], y_fit[fittable])
    elapsed = round(time.monotonic() - started, 2)

    return ProbeFit(
        model=model,
        feature_names=list(X.columns),
        target_transform=target_transform,
        classes=classes,
        fit_seconds=elapsed,
        spec=describe_probe(),
        unfitted_rows=int((~fittable).sum()),
    )


def probe_predict(fit: ProbeFit, X: pd.DataFrame, task_type: str) -> np.ndarray:
    """Predictions on the original target scale, whatever the model was fitted on."""
    from .target import invert_target

    preds = predict_model(fit.model, X, task_type, classes=fit.classes)
    if fit.target_transform != "none":
        preds = invert_target(fit.target_transform, preds)
    return preds
