"""How far is the top 10% (0.43 log loss), and is the gap model or features? No LLM.

    uv run python research/otto_multiclass/gap_split.py

Same split as runner.py; `id` is never a feature. Every arm is LightGBM at learning
rate 0.03, rounds stopped early (200 patience) on a stratified 15% of the training
rows and refitted on all of them, as 0098 ships.

- `raw` — the 93 counts (references.py's `lgbm_es`).
- `raw_sub` — the same with row and column subsampling (0.8 / 0.5): the model side.
- `sub_tfidf` — `raw_sub` plus row totals and a tf-idf of the counts, its idf fitted
  on the training rows.
- `sub_tfidf_knn` — plus, per class, the mean distance to the 4 nearest products of
  that class (log1p counts, Euclidean). Out-of-fold on the training rows (5 folds),
  so no row finds itself; test rows look up all training rows.

Writes `gap_split.json`.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import split  # noqa: E402
import prep_otto as prep  # noqa: E402

from tabular_agent.metrics import compute_metric  # noqa: E402

HERE = Path(__file__).resolve().parent
K = 4


def fit_predict(X, y, Xt, **params):
    import lightgbm as lgb

    tr, va = train_test_split(np.arange(len(X)), test_size=0.15, random_state=0, stratify=y)

    def make(n):
        return lgb.LGBMClassifier(n_estimators=n, learning_rate=0.03, random_state=0,
                                  verbose=-1, **params)

    m = make(5000).fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])],
                       callbacks=[lgb.early_stopping(200, verbose=False)])
    rounds = int(m.best_iteration_)
    return make(rounds).fit(X, y).predict_proba(Xt), rounds


def row_features(X, tfidf):
    out = pd.DataFrame({"row_sum": X.sum(axis=1), "row_nonzero": (X > 0).sum(axis=1)},
                       index=X.index)
    t = tfidf.transform(X.to_numpy()).toarray()
    return pd.concat([out, pd.DataFrame(t, index=X.index,
                                        columns=[f"tfidf_{c}" for c in X.columns])], axis=1)


def knn_distances(ref_X, ref_y, query_X, classes):
    out = np.zeros((len(query_X), len(classes)))
    for j, c in enumerate(classes):
        nn = NearestNeighbors(n_neighbors=K).fit(ref_X[ref_y == c])
        out[:, j] = nn.kneighbors(query_X)[0].mean(axis=1)
    return out


def knn_features(X, y, Xt, classes):
    L, Lt, yy = np.log1p(X.to_numpy()), np.log1p(Xt.to_numpy()), y.to_numpy()
    oof = np.zeros((len(X), len(classes)))
    for fit_rows, held in StratifiedKFold(5, shuffle=True, random_state=0).split(L, yy):
        oof[held] = knn_distances(L[fit_rows], yy[fit_rows], L[held], classes)
    test = knn_distances(L, yy, Lt, classes)
    cols = [f"knn_{c}" for c in classes]
    return (pd.DataFrame(oof, index=X.index, columns=cols),
            pd.DataFrame(test, index=Xt.index, columns=cols))


def main() -> int:
    train, test = split()
    X, y = train.drop(columns=["id", prep.TARGET]), train[prep.TARGET]
    Xt, yt = test.drop(columns=["id", prep.TARGET]), test[prep.TARGET].to_numpy()
    classes = sorted(y.unique())
    sub = {"subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.5}

    tfidf = TfidfTransformer().fit(X.to_numpy())
    F, Ft = pd.concat([X, row_features(X, tfidf)], axis=1), pd.concat(
        [Xt, row_features(Xt, tfidf)], axis=1)
    kn, knt = knn_features(X, y, Xt, classes)

    arms = {
        "raw": (X, Xt, {}),
        "raw_sub": (X, Xt, sub),
        "sub_tfidf": (F, Ft, sub),
        "sub_tfidf_knn": (pd.concat([F, kn], axis=1), pd.concat([Ft, knt], axis=1), sub),
    }
    out = {"test_logloss": {}, "rounds": {}}
    for name, (A, At, params) in arms.items():
        p, rounds = fit_predict(A, y, At, **params)
        out["test_logloss"][name] = round(compute_metric("logloss", yt, p, labels=classes), 6)
        out["rounds"][name] = rounds
        print(name, out["test_logloss"][name], rounds, flush=True)
    (HERE / "gap_split.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
