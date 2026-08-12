"""Model wrappers over LightGBM/XGBoost/CatBoost, a regularised linear model and a
trivial baseline, plus cross-validated training that emits OOF predictions and fold
metrics.

The linear entry is not here for accuracy. It is the **diagnostic** a tabular run
otherwise cannot make: if a tuned GBDT cannot clearly beat a self-tuning
elasticnet on the same table, the signal is in the features and further work on
the model is wasted. On house-prices that is exactly what happened — a plain
elasticnet scored 0.11216 against a tuned catboost's 0.11171, a gap of 4% of one
standard error — and the run had no way to find out, because there was no linear
model to ask.

It is also the only family here whose predictions decorrelate from the trees
enough for averaging to do anything (GBDT-vs-GBDT residual correlation on that
table: 0.86-0.88; elasticnet-vs-catboost: 0.84, and the blend moved the score
where a three-GBDT blend did not).
"""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .metrics import compute_metric
from .target import apply_target, check_target_domain, invert_target


class _Baseline:
    """Predicts the training mean (regression) or class priors (classification)."""

    def __init__(self, task_type: str):
        self.task_type = task_type

    def fit(self, X, y):
        y = np.asarray(y)
        if self.task_type == "regression":
            self.value_ = float(np.mean(y))
        else:
            self.classes_ = np.unique(y)
            self.proba_ = np.array([float((y == c).mean()) for c in self.classes_])
        return self

    def predict(self, X):
        return np.full(len(X), self.value_)

    def predict_proba(self, X):
        return np.tile(self.proba_, (len(X), 1))


#: CatBoost accepts several spellings of the same setting and **rejects** being given
#: two of them, even with identical values. Our defaults use one spelling; a caller may
#: reasonably use another (`iterations` for `n_estimators`, `depth` for `max_depth`), so
#: a supplied alias has to *replace* our default rather than sit next to it.
_CATBOOST_ALIASES: tuple[tuple[str, ...], ...] = (
    ("n_estimators", "iterations", "num_boost_round", "num_trees"),
    ("random_state", "random_seed"),
    ("verbose", "logging_level", "silent", "verbose_eval"),
    ("learning_rate", "eta"),
    ("max_depth", "depth"),
    ("reg_lambda", "l2_leaf_reg"),
)


def _resolve_params(defaults: dict, params: dict | None, aliases=()) -> dict:
    """Defaults with caller params layered on top, one key per setting.

    Splatting `**params` next to keyword defaults raises `TypeError: got multiple
    values` the moment a caller re-specifies a default — and these params come from
    an LLM, which will. Merging into a single dict makes overriding the normal case
    instead of a crash.
    """
    out = dict(defaults)
    p = dict(params or {})
    for group in aliases:
        if any(name in p for name in group):
            for name in group:
                out.pop(name, None)
    out.update(p)
    return out


#: The L1/L2 mixes an elasticnet searches. `1.0` is lasso, `0.01` is ridge in all
#: but name, so **one model type covers the whole family** and the agent never has
#: to pick an endpoint. Measured on house-prices: ridge 0.11420, lasso 0.11215,
#: self-tuned elasticnet 0.11216 — it finds the better end unaided, and the
#: difference between the ends was 18% of one standard error anyway. Offering
#: `ridge` and `lasso` separately would be offering two ways to spend an
#: experiment on noise.
#:
#: Eight points is affordable *here* because coordinate descent walks the alpha
#: path with warm starts: the next penalty starts from the last one's solution, so
#: a denser grid is close to free. That is a property of the solver, not of the
#: model, which is why the classification grid below is a different length.
_L1_RATIOS = (0.01, 0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0)

#: The same mixes for `saga`, and shorter — three points rather than eight.
#:
#: `LogisticRegressionCV` has no path: it fits from scratch at every
#: (mix, strength, inner fold), so the grid multiplies the cost directly where the
#: regression grid above does not. Eight mixes x ten strengths x five inner folds
#: is 400 independent fits, each a full pass structure over the data.
#:
#: Three points keep what the eight were for — finding which *end* of the family
#: the data wants — and drop resolution that was already measured as noise: on
#: house-prices the two endpoints differed by 18% of one standard error.
_SAGA_L1_RATIOS = (0.1, 0.5, 0.9)

#: Coordinate descent needs headroom on the weakly-regularised end of the path,
#: where it otherwise emits a ConvergenceWarning per fold per alpha. One
#: "iteration" here is a sweep over the *coordinates*, which is cheap.
_COORDINATE_DESCENT_MAX_ITER = 20000

#: The ceiling for `saga`, and it is a different number because it counts a
#: different thing: one saga iteration is a full **epoch over every row**.
#:
#: The two were one constant until a 476,170-row classification run spent 63
#: minutes inside the first fold of a single experiment and was killed before it
#: could finish. Measured on that table: 0.23s per epoch, so the shared 20000
#: meant ~77 minutes for *one* fit and ~510 hours for the 400 a search then made —
#: while the cross-validated wall-clock bound, which only checks between folds,
#: could not fire because no fold ever ended.
#:
#: A ceiling in the wrong unit is not a conservative choice. It is an unbounded
#: one, and it read as reasonable because the constant's neighbour justified it
#: for a solver that does not share its arithmetic.
_SAGA_MAX_ITER = 200

#: Rows the penalty **search** may see. The refit that follows always sees every
#: row it was given, so this bounds the search and not the model.
#:
#: The distinction is the point. Selecting a penalty is choosing between a handful
#: of candidates, and that verdict stabilises in far fewer rows than the fitted
#: coefficients need — measured on a 476k-row table, a linear model scored gini
#: 0.2367 on 25k rows and 0.2501 on all 376k, the same verdict against a GBDT's
#: 0.2848 in a fortieth of the time. Cost of the search therefore stops growing
#: with the table while the model it selects does not get worse.
#:
#: Below this the sample is the whole input and nothing changes, so every run at
#: the sizes this library was built on behaves exactly as it did.
PENALTY_SEARCH_MAX_ROWS = 20_000


def _finite(X):
    """Infinities to NaN, so the imputer downstream can deal with them.

    Module-level (not a lambda) because the fitted pipeline is pickled into the
    bundle and has to survive the round trip.

    This exists because agents write ratio features constantly — ``area / rooms``,
    ``finished / total`` — and any of them divides by zero somewhere. A GBDT splits
    happily on ``inf``; a linear model raises ``Input contains infinity``. Treating
    a non-finite value as missing is the honest reading: the ratio is undefined for
    that row, which is what NaN means.
    """
    return np.where(np.isfinite(X), X, np.nan)


#: How the run's metric names its sklearn equivalent, for the inner selection
#: `LogisticRegressionCV` does. Following the run's own metric is the whole point:
#: sklearn's default is accuracy, and selecting a penalty by accuracy while the
#: leaderboard ranks on log loss tunes for an objective nobody is scoring. `auc` on
#: more than two classes needs the one-vs-rest spelling.
_SKLEARN_SCORERS = {
    "logloss": "neg_log_loss",
    "accuracy": "accuracy",
    "auc": {"binary": "roc_auc", "multiclass": "roc_auc_ovr"},
}


def _classifier_scoring(metric: str | None, task_type: str) -> str:
    """The sklearn scorer for this run's metric, or log loss when it has no name.

    Falling back to `neg_log_loss` rather than to sklearn's accuracy default: a
    probability-calibrated choice is the better guess for a metric we cannot map,
    and every classification metric here except `accuracy` reads probabilities.
    """
    scorer = _SKLEARN_SCORERS.get(metric or "", "neg_log_loss")
    if isinstance(scorer, dict):
        return scorer.get(task_type, "roc_auc")
    return scorer


def _scalar(value) -> float:
    """A selected hyper-parameter as a number, whatever shape sklearn returned it in.

    `LogisticRegressionCV` reports `C_` as a scalar under the post-1.10 attribute
    layout and as a per-class array under the legacy one, and a caller may switch
    that back on through `params`.
    """
    return float(np.ravel(value)[0])


class _PenaltySearchThenRefit(BaseEstimator):
    """Choose the penalty on a bounded sample of rows, then fit on all of them.

    A self-tuning linear model does two jobs that scale differently. **Selecting**
    a penalty compares a handful of candidates and needs only enough rows to rank
    them; **fitting** at the chosen penalty wants every row there is. sklearn's
    `*CV` estimators do both on whatever they are handed, so the search inherits
    the size of the table and a model whose entire purpose is one cheap verdict
    becomes the most expensive thing in the run.

    Splitting them costs one extra fit and bounds the part that had no bound. When
    the input is already at or below ``max_search_rows`` the sample *is* the input,
    and the search estimator — which refits at its own choice anyway — is used as
    it stands, so nothing about small-table behaviour changes.

    The sample is drawn inside :func:`train_cv`'s fold, from that fold's training
    rows only, so it crosses no fold boundary. It is seeded from the run's seed for
    the reason every other estimate here is: a model that changes between identical
    runs makes the leaderboard non-reproducible.
    """

    def __init__(self, search=None, final=None, selected=None,
                 max_search_rows: int | None = PENALTY_SEARCH_MAX_ROWS,
                 seed: int = 42, stratify: bool = False):
        self.search = search
        self.final = final
        #: ``{our name: the fitted search estimator's attribute}`` — how the choice
        #: is read back out. `ElasticNetCV` names it `alpha_`, `LogisticRegressionCV`
        #: names the inverse of it `C_`, and neither is guessable.
        self.selected = selected
        self.max_search_rows = max_search_rows
        self.seed = seed
        self.stratify = stratify

    def _sample(self, X, y):
        n = len(X)
        if not self.max_search_rows or n <= self.max_search_rows:
            return None
        from sklearn.model_selection import train_test_split

        idx = np.arange(n)
        keep, _ = train_test_split(
            idx, train_size=int(self.max_search_rows), random_state=self.seed,
            # Stratified for classification, and it matters most exactly where
            # this cap bites: a heavily imbalanced target sampled uniformly can
            # hand the inner cross-validation a fold with almost no positives.
            stratify=np.asarray(y) if self.stratify else None,
        )
        return np.sort(keep)

    def fit(self, X, y):
        from sklearn.base import clone

        X = np.asarray(X)
        y = np.asarray(y)
        keep = self._sample(X, y)
        search = clone(self.search)
        search.fit(X, y) if keep is None else search.fit(X[keep], y[keep])

        self.search_rows_ = len(X) if keep is None else len(keep)
        self.selected_ = {
            name: _scalar(getattr(search, attr))
            for name, attr in (self.selected or {}).items()
        }
        if keep is None:
            # No sampling happened, so the search already fitted on everything it
            # would be refitted on. Refitting would spend a fit to arrive back
            # where we are.
            self.estimator_ = search
        else:
            self.estimator_ = clone(self.final).set_params(**self.selected_).fit(X, y)
        self.classes_ = getattr(self.estimator_, "classes_", None)
        self._warn_if_unconverged()
        return self

    def _warn_if_unconverged(self) -> None:
        """Say so when the fit stopped on its iteration ceiling rather than on tol.

        This model is a *diagnostic*: a GBDT that cannot beat it means the signal
        is in the features. An under-fitted linear model biases that verdict in the
        one direction it must never be biased — towards flattering the tree — and
        an iteration ceiling is exactly how it would happen silently.

        Not an error, because a stopped fit is still a usable model and refusing
        would cost the run an experiment over a warning. Measured on a 476k-row
        table, the ceiling is not normally reached: at the strength the search
        selects, saga converged in 118 epochs, while an unregularised fit on the
        same data had not converged in 1000. Convergence is a property of the
        penalty, and the search picks penalties that converge.
        """
        import warnings

        limit = getattr(self.estimator_, "max_iter", None)
        n_iter = getattr(self.estimator_, "n_iter_", None)
        if limit is None or n_iter is None:
            return
        if int(np.max(np.asarray(n_iter))) >= int(limit):
            warnings.warn(
                f"the linear fit stopped at its {limit}-iteration ceiling rather "
                "than converging, so it may be under-fitted — and this model type "
                "is the yardstick other models are judged against",
                stacklevel=2,
            )

    def predict(self, X):
        return self.estimator_.predict(np.asarray(X))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(np.asarray(X))


def _make_elasticnet(task_type: str, params: dict | None, seed: int,
                     metric: str | None = None):
    """L1/L2-regularised linear model, self-tuning, with the preprocessing it needs.

    The preprocessing is the reason this cannot be one line. GBDTs take the raw
    numeric matrix — NaNs, wild scales, infinities and all — and a linear model
    takes none of those, so imputation and standardisation have to happen
    *somewhere*. They happen here, inside the estimator, which puts them **inside
    the fold**: `train_cv` builds a fresh model per fold and fits it on that fold's
    training rows only, so the median and the scale are learned from training rows
    and applied to held-out ones. Doing it in a feature table instead would learn
    them from every row and leak across the fold boundary — the batch-dependence
    0011 exists to catch, in the one place the stability check cannot see.

    Self-tuning over both the penalty strength and the L1/L2 mix, by an inner
    cross-validation. That is deliberate: this model type exists to answer "do my
    features carry the signal, or does my model" in a single experiment, and a
    version needing three tuning runs first would not answer it.

    That same argument bounds what the tuning is allowed to cost. A diagnostic
    nobody can afford to run is not a diagnostic, so the inner search is capped at
    `PENALTY_SEARCH_MAX_ROWS` rows by `_PenaltySearchThenRefit` while the model it
    selects is still fitted on all of them.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    if task_type == "regression":
        from sklearn.linear_model import ElasticNet, ElasticNetCV

        # `alphas` as a count, not a grid: sklearn then derives the path from the
        # data. The useful range of penalty strengths depends on the scale and the
        # number of features, so a hardcoded grid would be wrong for both.
        #
        # NOTE, and it is a real limitation: `ElasticNetCV` has no `scoring`
        # parameter. It always selects alpha by squared error, whatever the run's
        # metric is. Usually that is harmless or exactly right — under
        # `target_transform="log1p"` the model is fitted on `log1p(y)`, so squared
        # error on that scale *is* `rmsle`, and the selection matches the
        # leaderboard by construction. It is wrong for `mae`, where the honest fix
        # is a manual path search rather than a keyword, and is not built.
        kw = _resolve_params(
            {"l1_ratio": list(_L1_RATIOS), "alphas": 60, "cv": 5,
             "max_iter": _COORDINATE_DESCENT_MAX_ITER, "random_state": seed},
            params,
        )
        estimator = _PenaltySearchThenRefit(
            search=ElasticNetCV(**kw),
            final=ElasticNet(max_iter=kw["max_iter"], random_state=seed),
            selected={"alpha": "alpha_", "l1_ratio": "l1_ratio_"},
            seed=seed,
        )
    else:
        from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

        # Same family, different parameterisation: `Cs` is inverse strength and the
        # mixes are `l1_ratios`. Fractional `l1_ratios` are what selects the
        # elasticnet penalty now — passing `penalty=` as well is deprecated — and
        # only `saga` implements it.
        #
        # Every default here is smaller than the regression branch's, and all three
        # for one reason: saga pays for the whole grid because it cannot warm-start
        # along it. 3 x 5 x 3 is 45 fits where 8 x 10 x 5 was 400.
        kw = _resolve_params(
            {"l1_ratios": list(_SAGA_L1_RATIOS), "Cs": 5, "cv": 3,
             "solver": "saga", "max_iter": _SAGA_MAX_ITER, "random_state": seed,
             # Follows the *run's* metric, not a constant. sklearn's default is
             # accuracy; selecting a penalty by accuracy while the leaderboard ranks
             # on log loss or AUC tunes for an objective nobody is scoring.
             "scoring": _classifier_scoring(metric, task_type),
             # Opt into the post-1.10 attribute layout now. Nothing here reads the
             # deprecated paths (`predict_model` only needs `classes_`), and leaving
             # it unset emits a FutureWarning per fold into every run's log.
             "use_legacy_attributes": False},
            params,
        )
        estimator = _PenaltySearchThenRefit(
            search=LogisticRegressionCV(**kw),
            final=LogisticRegression(
                solver=kw["solver"], max_iter=kw["max_iter"], random_state=seed,
            ),
            selected={"C": "C_", "l1_ratio": "l1_ratio_"},
            seed=seed,
            stratify=True,
        )

    return make_pipeline(
        FunctionTransformer(_finite, feature_names_out="one-to-one"),
        # `keep_empty_features` so an all-NaN column in one fold does not silently
        # change the matrix width between folds.
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        estimator,
    )


def _make_model(model_type: str, task_type: str, params: dict | None, seed: int,
                metric: str | None = None):
    """Build an unfitted model.

    ``metric`` is only consulted by model types that run their own inner selection
    — currently just `elasticnet`, whose classification branch would otherwise
    choose a penalty by accuracy while the run is ranked on something else. The
    GBDTs ignore it: their loss is set by `task_type` and their rounds are fixed.
    """
    if model_type == "baseline":
        return _Baseline(task_type)
    if model_type == "elasticnet":
        return _make_elasticnet(task_type, params, seed, metric)
    if model_type == "lgbm":
        import lightgbm as lgb

        kw = _resolve_params({"random_state": seed, "n_estimators": 200, "verbose": -1}, params)
        cls = lgb.LGBMRegressor if task_type == "regression" else lgb.LGBMClassifier
        return cls(**kw)
    if model_type == "xgb":
        import xgboost as xgb

        kw = _resolve_params({"random_state": seed, "n_estimators": 200, "verbosity": 0}, params)
        cls = xgb.XGBRegressor if task_type == "regression" else xgb.XGBClassifier
        return cls(**kw)
    if model_type == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        kw = _resolve_params(
            {"random_state": seed, "n_estimators": 200, "verbose": False},
            params,
            _CATBOOST_ALIASES,
        )
        cls = CatBoostRegressor if task_type == "regression" else CatBoostClassifier
        return cls(**kw)
    raise ValueError(
        f"unknown model_type: {model_type!r} (use 'baseline', 'elasticnet', 'lgbm', "
        "'xgb' or 'catboost')"
    )


def predict_model(model, X: pd.DataFrame, task_type: str, classes=None) -> np.ndarray:
    if task_type == "regression":
        return np.asarray(model.predict(X), dtype=float)
    proba = np.asarray(model.predict_proba(X), dtype=float)
    if classes is None:
        if task_type == "binary":
            return proba[:, 1]
        return proba
    # Map the model's own (possibly class-incomplete) columns into the
    # global, sorted `classes` index space so callers always get a
    # consistently shaped matrix regardless of which classes this
    # particular model happened to see during fitting.
    classes = np.asarray(classes)
    model_classes = np.asarray(model.classes_)
    col_idx = np.searchsorted(classes, model_classes)
    full = np.zeros((len(X), len(classes)))
    full[:, col_idx] = proba
    if task_type == "binary":
        return full[:, 1]
    return full


class TrainingTooSlow(ValueError):
    """A cross-validated fit was abandoned for exceeding its wall-clock budget.

    A ``ValueError`` so it reaches the agent as a correction rather than ending
    the run (0007): the fit is gone either way, and "ask for something cheaper"
    is a message it can act on.
    """


@dataclass
class TrainResult:
    oof: np.ndarray
    fold_metrics: list[float]
    models: list
    final_model: object
    feature_names: list[str]
    model_type: str
    target_transform: str = "none"
    #: Wall-clock seconds spent fitting — every fold plus the final model. Recorded
    #: because nothing else in a run measures it, and the one time it mattered we
    #: had to reconstruct it from file timestamps after the fact (Rossmann v6: two
    #: experiments with identical parameters took 3.8 and 113.2 minutes).
    fit_seconds: float = 0.0


def train_cv(X, y, folds, task_type, metric, model_type="lgbm", params=None, seed=42,
             target_transform="none", time_budget_s=None) -> TrainResult:
    """Cross-validated fit. ``target_transform`` changes what the model is fitted
    on; it never changes what is *reported*.

    The models see ``f(y)``, but predictions are inverted before a metric touches
    them and before they are stored as out-of-fold values. So `fold_metrics` and
    `oof` are on the original scale for every experiment, transformed or not,
    which is what keeps a leaderboard mixing the two comparable — and what keeps
    a residual denominated in the units the agent asked about.

    ``time_budget_s`` bounds the whole fit, checked **between folds** — coarse on
    purpose. A hard interrupt would need a subprocess and would buy little: the
    case this exists for is a fit whose *every* fold is slow, so the first
    boundary catches it, and a genuinely cheap fit never reaches a check at all.

    It is deliberately not a parameter cap. Rossmann v6 produced two experiments
    with byte-identical parameters, 30 and 32 features and the same 799,654 rows
    that took 3.8 and 113.2 minutes — so the cost of a fit is not a function of
    what was asked for, and a rule about parameters would have caught the wrong
    thing. Wall clock does not care why.
    """
    check_target_domain(target_transform, y, task_type)
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    # What the models are fitted on. Stateless, so a fold's transform carries no
    # information about the rows held out of it.
    y_fit = pd.Series(apply_target(target_transform, y)) if target_transform != "none" else y
    n = len(X)
    classes = None
    if task_type in ("binary", "multiclass"):
        classes = np.unique(y.to_numpy())
    # NaN, not zero. Under forward-chaining folds the earliest block is never
    # validated — nothing precedes it to train on — so some rows genuinely have no
    # out-of-fold prediction. Zero-filling would hand those rows a *confident and
    # wrong* prediction: a residual of exactly `y`, indistinguishable from a real
    # catastrophic miss. With random folds every row is validated and nothing here
    # stays NaN, so this costs the common case nothing.
    if task_type == "multiclass":
        k = len(classes)
        oof = np.full((n, k), np.nan)
    else:
        oof = np.full(n, np.nan)

    models = []
    fold_metrics = []
    started = time.monotonic()
    for i, (tr, va) in enumerate(folds):
        model = _make_model(model_type, task_type, params, seed, metric)
        model.fit(X.iloc[tr], y_fit.iloc[tr])
        preds = predict_model(model, X.iloc[va], task_type, classes=classes)
        if target_transform != "none":
            preds = invert_target(target_transform, preds)
        oof[va] = preds
        # `labels` is only consumed by the logloss branch of compute_metric; it's a harmless no-op for other metrics.
        metric_kwargs = {"labels": list(classes)} if classes is not None else {}
        fold_metrics.append(compute_metric(metric, y.iloc[va].to_numpy(), preds, **metric_kwargs))
        models.append(model)

        elapsed = time.monotonic() - started
        # Checked after a fold rather than before: one completed fold is the only
        # honest estimate of what the remaining ones plus the final fit will cost,
        # and it lets the message quote a measurement instead of a prediction.
        if time_budget_s is not None and elapsed > time_budget_s:
            raise TrainingTooSlow(
                f"abandoned this fit after {elapsed / 60:.1f} min: {i + 1} of "
                f"{len(folds)} folds done, and the budget for a whole experiment is "
                f"{time_budget_s / 60:.1f} min. Nothing was recorded. Ask for something "
                f"cheaper — fewer boosting rounds, a shallower model, or a smaller "
                f"feature table — and spend the run on features rather than on one "
                f"expensive fit."
            )

    final = _make_model(model_type, task_type, params, seed, metric)
    final.fit(X, y_fit)
    return TrainResult(oof, fold_metrics, models, final, list(X.columns), model_type,
                       target_transform, round(time.monotonic() - started, 1))
