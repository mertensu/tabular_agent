"""The post's tuned baseline on our splits: XGBoost with 30 Optuna trials.

    uv run --with optuna --with xgboost python research/credit_default_tabfm/anchors.py

Same 9 cells and the same 6,000 test rows as runner.py, so the agent is compared
with XGBoost on identical rows rather than with numbers from a split we cannot
reproduce. The post does not give its search space or tuning protocol; this one
is a common one (3-fold stratified CV AUC, 30 TPE trials, then a refit on all
training rows). Writes `anchors.json`. No LLM calls.
"""

import json
from pathlib import Path
import sys
import time

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import SEEDS, SIZES, split  # noqa: E402
import prep_credit  # noqa: E402

HERE = Path(__file__).resolve().parent
TRIALS = 30


def xgb_optuna(X, y, seed):
    import optuna
    import xgboost as xgb

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)

    def params(t):
        return dict(
            n_estimators=t.suggest_int("n_estimators", 100, 1000),
            max_depth=t.suggest_int("max_depth", 2, 10),
            learning_rate=t.suggest_float("learning_rate", 0.005, 0.3, log=True),
            subsample=t.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=t.suggest_float("min_child_weight", 1, 20, log=True),
            reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True),
        )

    def objective(t):
        m = xgb.XGBClassifier(**params(t), tree_method="hist", random_state=seed, n_jobs=-1)
        return cross_val_score(m, X, y, cv=cv, scoring="roc_auc").mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=TRIALS)
    m = xgb.XGBClassifier(**study.best_params, tree_method="hist", random_state=seed, n_jobs=-1)
    return m.fit(X, y)


def main() -> int:
    rows = []
    for n in SIZES:
        for s in SEEDS:
            train, test = split(n, s)
            drop = [prep_credit.TARGET, "ID"]
            X, y = train.drop(columns=drop), train[prep_credit.TARGET]
            Xt, yt = test.drop(columns=drop), test[prep_credit.TARGET]
            t0 = time.time()
            p = xgb_optuna(X, y, s).predict_proba(Xt)[:, 1]
            rows.append({"arm": "xgb_optuna30", "n_train": n, "seed": s,
                         "test_auc": round(float(roc_auc_score(yt, p)), 6),
                         "seconds": round(time.time() - t0, 1)})
            print(rows[-1], flush=True)
            (HERE / "anchors.json").write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
