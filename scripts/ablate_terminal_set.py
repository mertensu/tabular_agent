"""Ablation: fit the enumerated terminal set on v8's OWN winning table.

v9 improved on v8 and two things changed at once — the feature search, and the
fact that a terminal enumeration fits a family v8's search never tried. This
separates them. If elasticnet on v8's t11 lands near v9's 0.12037, the gain came
from the terminal fit rather than from the search.

Nothing here touches v8's journal: it reads the table, fits, writes a CSV.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from tabular_agent.cv import make_folds
from tabular_agent.harness import journal
from tabular_agent.io import load_table
from tabular_agent.metrics import compute_metric
from tabular_agent.models import predict_model, train_cv
from tabular_agent.pipeline import Pipeline

RUN = "_runs/house_prices_v8"
TABLE = "t11"          # v8's winning table (e9 catboost, val 0.11171)
TEST = "house_prices/test.csv"
TARGET = "SalePrice"

j = journal.load(RUN)
spec_metric = j["task"]["metric"]
task_type = j["task"]["task_type"]

tbl = load_table(os.path.join(RUN, "tables", f"{TABLE}.parquet"))
lineage = json.load(open(os.path.join(RUN, "tables", f"{TABLE}.pipeline.json")))
val = load_table(os.path.join(RUN, "data", "val.parquet"))

X = tbl.drop(columns=[TARGET]).select_dtypes(include="number")
y = tbl[TARGET]
print(f"{TABLE}: {len(tbl)} rows, {X.shape[1]} numeric features, metric {spec_metric}")

produced = Pipeline.from_dict(lineage).transform(val.drop(columns=[TARGET]))
missing = [c for c in X.columns if c not in produced.columns]
assert not missing, missing
y_val = val[TARGET].to_numpy()

test = pd.read_csv(TEST)
test_features = Pipeline.from_dict(lineage).transform(test.drop(columns=["Id"]))

# v9's own terminal set and target scale, so this is the same procedure applied
# to v8's table rather than a new one.
TERMINAL = ("xgb", "catboost", "elasticnet")
folds = make_folds(y, task_type, n_splits=5, seed=42)

rows = []
for model_type in TERMINAL:
    result = train_cv(X, y, folds, task_type, spec_metric, model_type=model_type,
                      seed=42, target_transform="log1p")
    preds_val = predict_model(result.final_model, produced[list(X.columns)], task_type)
    preds_val = np.expm1(preds_val)
    score = compute_metric(spec_metric, y_val, preds_val)
    preds_test = np.expm1(predict_model(result.final_model,
                                        test_features[list(X.columns)], task_type))
    out = f"{RUN}/submission_ablation_{model_type}.csv"
    pd.DataFrame({"Id": test["Id"], TARGET: preds_test}).to_csv(out, index=False)
    cv = float(np.mean(result.fold_metrics))
    rows.append((model_type, score, cv, out))
    print(f"  {model_type:<12s} val {score:.5f}  cv {cv:.5f}  -> {os.path.basename(out)}")

best = min(rows, key=lambda r: r[1])
print(f"\nterminal pick by val: {best[0]} at {best[1]:.5f}")
print("v8 actually shipped   : catboost at 0.11171 -> Kaggle 0.12298")
print("v9 shipped            : elasticnet at 0.11579 -> Kaggle 0.12037")
print(f"\nsubmit {os.path.basename(best[3])} to answer it.")
