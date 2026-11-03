"""Reference model on the run's split, no LLM: a plain model on the raw columns.

    uv run python research/ieee_cross_row/references.py

`lgbm_es` — LightGBM at learning rate 0.03 on every raw column except TransactionID
and TransactionDT, string columns as native categories, rounds stopped early (200
patience) on the latest 15% of the agent's rows and refitted on all of them, as
0098 ships. Scored by AUC on the test block. Writes `references.json`.
"""

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import TARGET, split  # noqa: E402
import prep_ieee as prep  # noqa: E402

from tabular_agent.metrics import compute_metric  # noqa: E402

HERE = Path(__file__).resolve().parent


def frames(train, test):
    drop = [TARGET, "TransactionID", prep.SPLIT_ON]
    X, Xt = train.drop(columns=drop), test.drop(columns=drop)
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) in ("str", "string"):
            levels = sorted(set(X[c].dropna()) | set(Xt[c].dropna()))
            X[c] = pd.Categorical(X[c], categories=levels)
            Xt[c] = pd.Categorical(Xt[c], categories=levels)
    return X, Xt


def main() -> int:
    import lightgbm as lgb

    train, test = split()
    X, Xt = frames(train, test)
    y = train[TARGET]
    order = np.argsort(train[prep.SPLIT_ON].to_numpy(), kind="stable")
    n_stop = int(round(0.15 * len(X)))
    tr, va = order[:-n_stop], order[-n_stop:]

    def make(n):
        return lgb.LGBMClassifier(n_estimators=n, learning_rate=0.03, random_state=0,
                                  verbose=-1)

    started = time.time()
    m = make(5000).fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])],
                       callbacks=[lgb.early_stopping(200, verbose=False)])
    rounds = int(m.best_iteration_)
    p = make(rounds).fit(X, y).predict_proba(Xt)[:, 1]
    out = {"n_train": len(train), "n_test": len(test), "rounds": {"lgbm_es": rounds},
           "minutes": round((time.time() - started) / 60, 1),
           "test_auc": {"lgbm_es": round(compute_metric("auc", test[TARGET].to_numpy(), p), 6)}}
    (HERE / "references.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
