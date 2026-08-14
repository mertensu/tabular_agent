"""Package an equal-weight blend of a finished run's terminal fits, for scoring.

0042 found that equal-weight averages of Rossmann v7's four terminal candidates
beat every one of them on the sealed rows — the 3-family average by 0.0164 of
rmspe over what actually shipped. That measurement has a defect it cannot fix
from inside: **it was made on v7's own holdout**, and choosing a rule by
consulting a holdout is what 0023 exists to prevent, one level up.

This script exists because there is a second held-out set that has *not* been
consulted: the competition's own test labels. Submitting the blend and the
shipped model together tests 0042's claim 1 on rows nothing here has ever
touched, today, rather than waiting for a fresh sealed run.

**It changes no library behaviour.** Like `rebuild_submission.py` and
`ablate_terminal_set.py`, it reads a finished run directory and writes CSVs
beside it.

It has since done its job: the three-GBDT blend scored **0.13813** on Rossmann's
private leaderboard against **0.14356** for the argmax winner, and on the
strength of that external set **0044 made the equal-weight blend what
`finalize_run` ships**. So this script no longer previews a hypothetical — it
enumerates the *subsets* the harness does not, which is still the only way to ask
whether some smaller combination would have done better than averaging
everything.

Usage:

    python scripts/blend_submission.py _runs/rossmann_v7

Every model is scored exactly as it was fitted — on the work half alone, no
refit. Predictions are averaged **after** each model's own target transform is
inverted, because the blend has to be an average of sales, not of log-sales.
"""
import itertools
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

from tabular_agent.harness import journal
from tabular_agent.harness.tools import _load_lineage
from tabular_agent.io import load_table
from tabular_agent.metrics import compute_metric
from tabular_agent.models import predict_model
from tabular_agent.target import invert_target


def load_predictors(session, run_dir: str):
    """Every non-baseline terminal fit, as (model_type, predict_fn)."""
    j = journal.load(run_dir)
    out = []
    for r in j.get("leaderboard") or []:
        if r["model_type"] == "baseline":
            continue
        exp_dir = os.path.join(run_dir, r["artifact_path"])
        with open(os.path.join(exp_dir, "meta.json")) as f:
            meta = json.load(f)
        model = joblib.load(os.path.join(exp_dir, "final_model.joblib"))
        pipeline = _load_lineage(session.resolve_table(meta["source_table"]),
                                 meta["task_type"])

        def predict(raw, _m=model, _meta=meta, _p=pipeline):
            feats = _p.transform(raw)
            preds = predict_model(_m, feats[list(_meta["feature_names"])],
                                  _meta["task_type"])
            # After the inversion, never before: averaging on the log scale is a
            # geometric mean of sales, which is a different estimator and not the
            # one 0042 measured.
            if _meta.get("target_transform", "none") != "none":
                preds = invert_target(_meta["target_transform"], preds)
            return preds

        out.append((r["model_type"], r["id"], predict))
    return out


def main(run_dir: str) -> int:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from prep_rossmann import TEST_OUT
    from run_rossmann_v7_sealed import make_session

    session = make_session()
    spec = session.resolved()
    preds = {}
    print("scoring each terminal fit on the sealed rows and the test set\n")

    holdout = load_table(session.holdout_path) if session.holdout_path else None
    test = pd.read_parquet(TEST_OUT)

    for model_type, exp_id, predict in load_predictors(session, run_dir):
        test_p = predict(test.drop(columns=["Id"]))
        line = f"  {model_type:<11s} ({exp_id})"
        if holdout is not None:
            sealed = compute_metric(spec.metric, holdout["Sales"],
                                    predict(holdout.drop(columns=["Sales"])))
            line += f"  sealed {sealed:.5f}"
        print(line)
        preds[model_type] = test_p

    gbdt = [m for m in ("lgbm", "xgb", "catboost") if m in preds]
    combos = [("all", list(preds))]
    combos += [("+".join(c), list(c))
               for k in (2, 3) for c in itertools.combinations(gbdt, k)]

    print("\nwriting submissions:")
    for name, members in combos:
        blended = np.mean([preds[m] for m in members], axis=0)
        negative = int((blended < 0).sum())
        path = os.path.join(run_dir, f"submission_blend_{name.replace('+', '_')}.csv")
        pd.DataFrame({"Id": test["Id"], "Sales": blended}).to_csv(path, index=False)
        print(f"  {name:<26s} -> {os.path.basename(path)}"
              + (f"  ({negative} negative predictions, left as they are)" if negative else ""))

    print("\nScore these against the run's own submission.csv (the argmax winner).")
    print("0042 claim 1 passes if a blend beats it on Kaggle — an independent set,")
    print("unlike the sealed rows the blend was discovered on.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "_runs/rossmann_v7"))
