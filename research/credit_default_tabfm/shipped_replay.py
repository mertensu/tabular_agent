"""What 0098's finalize would have shipped on each credit run, replayed offline.

    uv run python research/credit_default_tabfm/shipped_replay.py

Per run: `fit_fitted_ruler` on the work-only table the run chose (its rounds), then
LightGBM at those rounds refitted on the run's training + validation table, scored
on the test rows through the bundle's lineage. The path `_finalize_minimal_run`
takes since 0098, without rerunning the agent. Writes `shipped_replay.json`.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import SEEDS, SIZES, split  # noqa: E402
import prep_credit  # noqa: E402

from tabular_agent.api import Run  # noqa: E402
from tabular_agent.harness.tools import _feature_frame  # noqa: E402
from tabular_agent.models import fit_final_model  # noqa: E402
from tabular_agent.probe import PROBE_SEED, fit_fitted_ruler  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> int:
    rows = []
    for n in SIZES:
        for s in SEEDS:
            result = json.loads((HERE / f"result_{n}_s{s}.json").read_text())
            run = Run(result["run_dir"])
            _, test = split(n, s)
            refit = run.training_features()
            names = [c for c in refit.columns if c != prep_credit.TARGET]
            chosen = Path(run.run_dir) / "tables" / Path(run.bundle._bundle["training_table"]).name
            work = pd.read_parquet(chosen)
            X, _ = _feature_frame(work, prep_credit.TARGET)
            ruler = fit_fitted_ruler(X[names], work[prep_credit.TARGET], "binary", "auc", "none")
            params = {**ruler.spec["params"], "n_estimators": ruler.spec["rounds"]}
            model = fit_final_model(refit[names], refit[prep_credit.TARGET], "binary", "auc",
                                    model_type="lgbm", params=params, seed=PROBE_SEED)
            Xt = run.bundle.feature_table(test.drop(columns=[prep_credit.TARGET]))[names]
            p = np.asarray(model.predict_proba(Xt))[:, -1]
            rows.append({"n_train": n, "seed": s, "rounds": ruler.spec["rounds"],
                         "shipped_before": result["test_auc"],
                         "shipped_0098": round(float(roc_auc_score(test[prep_credit.TARGET], p)), 6)})
            print(rows[-1], flush=True)
    (HERE / "shipped_replay.json").write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
