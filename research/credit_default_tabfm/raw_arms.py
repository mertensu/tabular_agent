"""The run's own model on the raw 23 columns: is the gap to XGBoost features or capacity?

    uv run python research/credit_default_tabfm/raw_arms.py

Same 9 cells and test rows as runner.py, no agent features, two arms:

- `raw_probe` — the probe exactly as the product fits it (LightGBM, 200 rounds).
- `raw_lgbm_es` — LightGBM at learning rate 0.03, early-stopped on a stratified
  15% of the training rows, refitted on all of them (evaluate.py's protocol).

Writes `raw_arms.json`. No LLM calls.
"""

import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import SEEDS, SIZES, split  # noqa: E402
import prep_credit  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "capacity_ship_minimal"))
from evaluate import _lgbm, capacity_arm  # noqa: E402

from tabular_agent.probe import fit_probe, probe_predict  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> int:
    rows = []
    for n in SIZES:
        for s in SEEDS:
            train, test = split(n, s)
            drop = [prep_credit.TARGET, "ID"]
            X, y = train.drop(columns=drop), train[prep_credit.TARGET]
            Xt, yt = test.drop(columns=drop), test[prep_credit.TARGET].to_numpy()
            p = np.asarray(probe_predict(fit_probe(X, y, "binary", "auc"), Xt, "binary"))
            p = p[:, -1] if p.ndim == 2 else p
            es, _, _ = capacity_arm(_lgbm(True), X, y, Xt, True, "none")
            for arm, pred in (("raw_probe", p), ("raw_lgbm_es", es)):
                rows.append({"arm": arm, "n_train": n, "seed": s,
                             "test_auc": round(float(roc_auc_score(yt, pred)), 6)})
                print(rows[-1], flush=True)
    (HERE / "raw_arms.json").write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
