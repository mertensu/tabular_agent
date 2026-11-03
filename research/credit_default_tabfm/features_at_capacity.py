"""Are the agent's features worth anything once the model is early-stopped?

    uv run python research/credit_default_tabfm/features_at_capacity.py

For each run: `raw_arms.py`'s early-stopped LightGBM (learning rate 0.03, stopped
on a stratified 15%, refitted on all rows), fitted on the run's selected table
(`run.training_features()`) and scored on the test rows through
`run.bundle.feature_table`. Compared per cell with the same fit on the raw table.
Writes `features_at_capacity.json`. No LLM calls.
"""

import json
from pathlib import Path
import sys

from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import SEEDS, SIZES, split  # noqa: E402
import prep_credit  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "capacity_ship_minimal"))
from evaluate import _lgbm, capacity_arm  # noqa: E402

from tabular_agent.api import Run  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> int:
    raw = {(r["n_train"], r["seed"]): r["test_auc"]
           for r in json.loads((HERE / "raw_arms.json").read_text()) if r["arm"] == "raw_lgbm_es"}
    rows = []
    for n in SIZES:
        for s in SEEDS:
            run = Run(json.loads((HERE / f"result_{n}_s{s}.json").read_text())["run_dir"])
            _, test = split(n, s)
            sel = run.training_features()
            X, y = sel.drop(columns=[prep_credit.TARGET]), sel[prep_credit.TARGET]
            Xt = run.bundle.feature_table(test.drop(columns=[prep_credit.TARGET]))[list(X.columns)]
            p, rounds, _ = capacity_arm(_lgbm(True), X, y, Xt, True, "none")
            auc = float(roc_auc_score(test[prep_credit.TARGET], p))
            rows.append({"n_train": n, "seed": s, "fitted_rows": len(X), "n_columns": X.shape[1],
                         "rounds": rounds, "selected_lgbm_es": round(auc, 6),
                         "raw_lgbm_es": raw[(n, s)], "features_gain": round(auc - raw[(n, s)], 6)})
            print(rows[-1], flush=True)
    (HERE / "features_at_capacity.json").write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
