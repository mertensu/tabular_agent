"""Reference models on the run's split: what exposure-as-a-column costs against an offset.

    uv run python research/fremtpl2_exposure/references.py

Every arm predicts the claim count of each test policy and is scored by mean Poisson
deviance on the counts (`compute_metric`, as the run is). That is the same quantity as
the exposure-weighted deviance of the claim frequency, so offset and feature
formulations are scored on one scale. No LLM calls; IDpol is never a feature.

- `null` — the training rows' claim frequency × the policy's exposure.
- `glm_offset` — Poisson GLM on frequency weighted by exposure (scikit-learn
  tutorial's preprocessing: binned ages, log density, one-hot categories).
- `lgbm_feature` — LightGBM Poisson with Exposure as an ordinary column: the best
  the harness can express today, and what 0098 ships on the raw table.
- `lgbm_offset` — the same with log(Exposure) as an offset (init_score), Exposure
  not a column.

Both LightGBM arms: learning rate 0.03, rounds stopped early (200 patience) on a
stratified 15% of the training rows, refitted on all of them; category columns
native. Paired bootstrap (2,000 resamples) of each arm against `lgbm_feature`.
Writes `references.json`.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import split  # noqa: E402
import prep_fremtpl2 as prep  # noqa: E402

from tabular_agent.metrics import compute_metric  # noqa: E402

HERE = Path(__file__).resolve().parent
FEATURES = ["Area", "VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand",
            "VehGas", "Density", "Region"]
CATEGORICAL = ["Area", "VehBrand", "VehGas", "Region"]
LEARNING_RATE, MAX_ROUNDS, PATIENCE, STOP_FRACTION = 0.03, 5000, 200, 0.15
N_BOOT = 2000


def _frame(df, with_exposure):
    X = df[FEATURES + ([prep.EXPOSURE] if with_exposure else [])].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    return X


def lgbm(train, test, offset):
    import lightgbm as lgb

    X, Xt = _frame(train, not offset), _frame(test, not offset)
    for c in CATEGORICAL:  # one category set per column on both sides
        levels = sorted(set(train[c]) | set(test[c]))
        X[c], Xt[c] = X[c].cat.set_categories(levels), Xt[c].cat.set_categories(levels)
    y = train[prep.TARGET]
    base = np.log(train[prep.EXPOSURE].to_numpy()) if offset else None
    tr, va = train_test_split(np.arange(len(X)), test_size=STOP_FRACTION, random_state=0,
                              stratify=y > 0)

    def make(n):
        return lgb.LGBMRegressor(objective="poisson", n_estimators=n,
                                 learning_rate=LEARNING_RATE, random_state=0, verbose=-1)

    m = make(MAX_ROUNDS)
    m.fit(X.iloc[tr], y.iloc[tr], init_score=None if base is None else base[tr],
          eval_set=[(X.iloc[va], y.iloc[va])],
          eval_init_score=None if base is None else [base[va]],
          callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    rounds = max(1, int(m.best_iteration_))
    m = make(rounds).fit(X, y, init_score=base)
    if offset:
        raw = m.predict(Xt, raw_score=True)
        return np.exp(raw + np.log(test[prep.EXPOSURE].to_numpy())), rounds
    return m.predict(Xt), rounds


def glm(train, test):
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import PoissonRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import (
        FunctionTransformer, KBinsDiscretizer, OneHotEncoder, StandardScaler,
    )

    pre = ColumnTransformer([
        ("binned", KBinsDiscretizer(n_bins=10, subsample=None), ["VehAge", "DrivAge"]),
        ("onehot", OneHotEncoder(handle_unknown="ignore"),
         ["VehBrand", "VehPower", "VehGas", "Region", "Area"]),
        ("passthrough", "passthrough", ["BonusMalus"]),
        ("log", make_pipeline(FunctionTransformer(np.log), StandardScaler()), ["Density"]),
    ])
    model = make_pipeline(pre, PoissonRegressor(alpha=1e-4, solver="newton-cholesky"))
    freq = train[prep.TARGET] / train[prep.EXPOSURE]
    model.fit(train[FEATURES], freq, poissonregressor__sample_weight=train[prep.EXPOSURE])
    return model.predict(test[FEATURES]) * test[prep.EXPOSURE].to_numpy()


def paired(y, base, other):
    """Bootstrap base − other in mean deviance: positive means `other` is better."""
    from sklearn.metrics import mean_poisson_deviance

    rng = np.random.default_rng(0)
    n = len(y)
    diffs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        diffs.append(mean_poisson_deviance(y[i], base[i]) - mean_poisson_deviance(y[i], other[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"gain": round(float(np.mean(diffs)), 6), "interval": [round(lo, 6), round(hi, 6)]}


def main() -> int:
    train, test = split()
    y = test[prep.TARGET].to_numpy()
    freq = train[prep.TARGET].sum() / train[prep.EXPOSURE].sum()
    preds, rounds = {"null": freq * test[prep.EXPOSURE].to_numpy()}, {}
    preds["glm_offset"] = glm(train, test)
    preds["lgbm_feature"], rounds["lgbm_feature"] = lgbm(train, test, offset=False)
    preds["lgbm_offset"], rounds["lgbm_offset"] = lgbm(train, test, offset=True)
    out = {
        "n_train": len(train), "n_test": len(test),
        "test_poisson_deviance": {k: round(compute_metric("poisson_deviance", y, p), 6)
                                  for k, p in preds.items()},
        "rounds": rounds,
        "vs_lgbm_feature": {k: paired(y, preds["lgbm_feature"], p)
                            for k, p in preds.items() if k != "lgbm_feature"},
    }
    (HERE / "references.json").write_text(json.dumps(out, indent=2) + "\n")
    # Kept for pairing the agent's run against the references later.
    dest = HERE.parents[1] / "_runs" / "fremtpl2_exposure"
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(preds).to_parquet(dest / "reference_preds.parquet", index=False)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
