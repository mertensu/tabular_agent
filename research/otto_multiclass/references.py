"""Reference models on the run's split, no LLM.

    uv run python research/otto_multiclass/references.py

- `probe` — the harness's probe on the raw 93 features (LightGBM, 200 rounds).
- `lgbm_es` — LightGBM at learning rate 0.03, rounds stopped early (200 patience)
  on a stratified 15% of the training rows, refitted on all: what 0098 ships on the
  raw table.

`id` is never a feature. Writes `references.json`.
"""

import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import split  # noqa: E402
import prep_otto as prep  # noqa: E402

from tabular_agent.metrics import compute_metric  # noqa: E402
from tabular_agent.probe import fit_probe, probe_predict  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> int:
    import lightgbm as lgb

    train, test = split()
    X, y = train.drop(columns=["id", prep.TARGET]), train[prep.TARGET]
    Xt, yt = test.drop(columns=["id", prep.TARGET]), test[prep.TARGET].to_numpy()
    labels = sorted(y.unique())
    preds, rounds = {}, {}

    preds["probe"] = np.asarray(probe_predict(fit_probe(X, y, "multiclass", "logloss"),
                                              Xt, "multiclass"))
    tr, va = train_test_split(np.arange(len(X)), test_size=0.15, random_state=0, stratify=y)
    make = lambda n: lgb.LGBMClassifier(n_estimators=n, learning_rate=0.03,  # noqa: E731
                                        random_state=0, verbose=-1)
    m = make(5000).fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])],
                       callbacks=[lgb.early_stopping(200, verbose=False)])
    rounds["lgbm_es"] = int(m.best_iteration_)
    preds["lgbm_es"] = make(rounds["lgbm_es"]).fit(X, y).predict_proba(Xt)

    out = {"n_train": len(train), "n_test": len(test), "rounds": rounds,
           "test_logloss": {k: round(compute_metric("logloss", yt, p, labels=labels), 6)
                            for k, p in preds.items()}}
    (HERE / "references.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
