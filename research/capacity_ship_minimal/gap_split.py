"""Where is the gap to the top of the board: features or models? (no LLM)

    uv run python research/capacity_ship_minimal/gap_split.py

On r11's t20 (the 0094 pick, 0.1270 private on Kaggle), two tables × two ways
of combining the shipped models:

- tables: t20 as built, and t20 plus short calendar windows per store over
  `Open`, `SchoolHoliday` and `Promo` (closed / holiday / promo yesterday and
  tomorrow, counts over the last and next 7 days; `rossmann_open_windows.WINDOWS`).
  The windows are a real lineage step (`FutureWindowEncoder`, 0093), so they are
  refitted with everything else;
- models: r11's 7 shipped recipes. The blend uses the shipped weights; "best"
  is the single member with the best validation score.

Every member is fitted three times on each table:
  1. training rows → validation score (which member is best);
  2. training + validation rows → our 48 test days (`audit.parquet`);
  3. every row of the competition's training file → a Kaggle file per arm.

Writes `gap_split_rossmann_r11.json` and
`_runs/capacity_ship_minimal/rossmann/submission_gap_<arm>.csv`.
"""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rossmann_open_windows import WINDOWS  # noqa: E402
from runner import DATASETS, HERE, OUT  # noqa: E402

from tabular_agent.api import Run  # noqa: E402
from tabular_agent.blend import apply_blend  # noqa: E402
from tabular_agent.encoders import FutureWindowEncoder  # noqa: E402
from tabular_agent.harness.tools import _feature_frame, _fit_recipe, _load_lineage  # noqa: E402
from tabular_agent.metrics import compute_metric  # noqa: E402
from tabular_agent.models import fit_final_model, predict_model  # noqa: E402
from tabular_agent.pipeline import feature_table_bundle, input_dtypes  # noqa: E402
from tabular_agent.refit import refit_lineage  # noqa: E402
from tabular_agent.target import invert_target  # noqa: E402
from tabular_agent.taskspec import ResolvedSpec  # noqa: E402

TAG, TABLE = "r11", "t20"
TARGET, METRIC, GROUP, ORDER, BLOCK = "Sales", "rmspe", "Store", "Date", 48.0
CALENDAR = ("Open", "SchoolHoliday", "Promo")


def log(*a):
    print(*a, flush=True)


def main() -> int:
    cfg = DATASETS["rossmann"]
    root = Path(json.loads((HERE / f"result_rossmann_{TAG}.json").read_text())["run_dir"])
    run = Run(root)
    b, j = run.bundle._bundle, run.journal
    split = j["split"]
    metas = [json.loads((root / "experiments" / e / "meta.json").read_text())
             for e in b["fitted_on"]["refit_of"]]
    spec = ResolvedSpec(
        target=TARGET, task_type="regression", metric=METRIC, positive_class=None,
        notes=None, column_notes=None, n_splits=5, seed=split["seed"],
        target_transform=metas[0].get("target_transform", "none"),
        objective=j["task"].get("objective"), forecast_horizon=split["forecast_horizon"],
    )
    recipes = [_fit_recipe(SimpleNamespace(resolved=lambda: spec), m) for m in metas]
    weights = np.asarray(b["weights"], dtype=float)

    work = pd.read_parquet(root / "data" / "work.parquet")
    val = pd.read_parquet(root / "data" / "val.parquet")
    cols = list(work.columns)
    dtypes = input_dtypes(work, exclude=[TARGET])

    def prepared(frame: pd.DataFrame) -> pd.DataFrame:
        y = frame[TARGET].to_numpy() if TARGET in frame else None
        rows = run.bundle._prepare(frame.drop(columns=[TARGET], errors="ignore"))
        if y is not None:
            rows[TARGET] = y
        return rows[[c for c in cols if c in rows.columns]]

    audit = prepared(pd.read_parquet(OUT / "rossmann" / "audit.parquet"))
    full = prepared(pd.read_parquet(cfg["source"] / "train_joined.parquet"))
    kaggle = pd.read_parquet(cfg["source"] / "test_joined.parquet")
    kaggle_rows = run.bundle._prepare(kaggle.drop(columns=["Id"]))

    base = _load_lineage(str(root / "tables" / f"{TABLE}.parquet"), "regression")
    base_features = list(_feature_frame(pd.read_parquet(root / "tables" / f"{TABLE}.parquet"),
                                        TARGET)[0].columns)
    with_windows = _load_lineage(str(root / "tables" / f"{TABLE}.parquet"), "regression")
    window_cols = []
    for value in CALENDAR:
        enc = FutureWindowEncoder([GROUP], value, ORDER, BLOCK, WINDOWS).fit(work)
        with_windows.add_encoder(enc)
        window_cols += [c for c in enc.output_columns if c not in base_features]
    tables = {"t20": (base, base_features), "t20_windows": (with_windows, base_features + window_cols)}

    def fit_predict(lineage, features, fit_rows, targets: dict) -> dict:
        """Refit the lineage and every member on fit_rows; member predictions per target."""
        table, pipe = refit_lineage(lineage, fit_rows, spec, order=ORDER)
        bundle = {"pipeline": pipe, "feature_names": features, "input_dtypes": dtypes}
        X = {k: feature_table_bundle(bundle, rows.drop(columns=[TARGET], errors="ignore"))
             for k, rows in targets.items()}
        out = {k: [] for k in targets}
        for r in recipes:
            m = fit_final_model(table[features], table[TARGET], "regression", METRIC,
                                model_type=r["model_type"], params=r["params"] or None,
                                seed=r["seed"], target_transform=r["target_transform"],
                                reference_target=None, objective=spec.objective)
            for k in targets:
                p = predict_model(m, X[k], "regression")
                if r["target_transform"] != "none":
                    p = invert_target(r["target_transform"], p)
                out[k].append(np.asarray(p, dtype=float))
            log("   ", r["model_type"], "fitted")
        return out

    def score(y, p):
        return round(compute_metric(METRIC, y, np.clip(p, 0, None)), 6)

    result = {"tag": TAG, "table": TABLE, "window_columns": window_cols, "arms": {}}
    for name, (lineage, features) in tables.items():
        log(name, "1/3 training rows -> validation")
        on_val = fit_predict(lineage, features, work, {"val": val})["val"]
        val_scores = [score(val[TARGET].to_numpy(), p) for p in on_val]
        best = int(np.argmin(val_scores))

        log(name, "2/3 training + validation rows -> test days")
        on_test = fit_predict(lineage, features, pd.concat([work, val], ignore_index=True),
                              {"test": audit})["test"]
        y_test = audit[TARGET].to_numpy()

        log(name, "3/3 all rows -> Kaggle")
        on_kaggle = fit_predict(lineage, features, full, {"kaggle": kaggle_rows})["kaggle"]

        members = [{"model_type": r["model_type"], "params": r["params"],
                    "val": v, "test": score(y_test, p)}
                   for r, v, p in zip(recipes, val_scores, on_test)]
        for combo, test_p, kaggle_p in (
            ("blend", apply_blend(on_test, weights), apply_blend(on_kaggle, weights)),
            ("best", on_test[best], on_kaggle[best]),
        ):
            arm = f"{name}_{combo}"
            path = OUT / "rossmann" / f"submission_gap_{arm}.csv"
            pd.DataFrame({"Id": kaggle["Id"], TARGET: np.clip(kaggle_p, 0, None)}).to_csv(
                path, index=False)
            result["arms"][arm] = {"test": score(y_test, test_p), "file": str(path)}
            log(arm, result["arms"][arm])
        result[f"{name}_members"] = members
        result[f"{name}_best_member"] = best
        (HERE / "gap_split_rossmann_r11.json").write_text(
            json.dumps(result, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
