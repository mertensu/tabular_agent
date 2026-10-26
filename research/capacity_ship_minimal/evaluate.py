"""Score the shipped probe against capacity-fitted models on the audit slice.

    uv run python research/capacity_ship_minimal/evaluate.py house

Reads `result_<name>.json` written by runner.py and writes `arms_<name>.json`.
No LLM calls.
"""

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tabular_agent.api import Run
from tabular_agent.harness.tools import _feature_frame
from tabular_agent.metrics import compute_metric, greater_is_better
from tabular_agent.probe import default_target_transform, fit_probe, probe_predict
from tabular_agent.target import apply_target, invert_target

HERE = Path(__file__).resolve().parent
OUT = Path(__file__).resolve().parents[2] / "_runs" / "capacity_ship_minimal"
LEARNING_RATE = 0.03
MAX_ROUNDS = 5000
PATIENCE = 200
STOP_FRACTION = 0.15
SEED = 0
N_BOOT = 2000


def _stopped_rounds(make, X, y_fit, binary):
    """Rounds at which a fit on 85% of the work rows stopped improving on the other 15%."""
    tr, va = train_test_split(
        np.arange(len(X)), test_size=STOP_FRACTION, random_state=SEED,
        stratify=y_fit if binary else None,
    )
    model = make(MAX_ROUNDS)
    model.fit(X.iloc[tr], y_fit.iloc[tr], **_eval_kwargs(model, X.iloc[va], y_fit.iloc[va]))
    import lightgbm as lgb

    if isinstance(model, (lgb.LGBMRegressor, lgb.LGBMClassifier)):
        return max(1, int(model.best_iteration_))      # a count of rounds
    return int(model.get_best_iteration()) + 1          # CatBoost: index of the best tree


def _eval_kwargs(model, Xv, yv):
    import lightgbm as lgb

    if isinstance(model, (lgb.LGBMRegressor, lgb.LGBMClassifier)):
        return {"eval_set": [(Xv, yv)], "callbacks": [lgb.early_stopping(PATIENCE, verbose=False)]}
    return {"eval_set": (Xv, yv), "early_stopping_rounds": PATIENCE}


def _lgbm(binary):
    import lightgbm as lgb

    cls = lgb.LGBMClassifier if binary else lgb.LGBMRegressor
    return lambda n: cls(n_estimators=n, learning_rate=LEARNING_RATE,
                         random_state=SEED, verbose=-1)


def _catboost(binary):
    from catboost import CatBoostClassifier, CatBoostRegressor

    cls = CatBoostClassifier if binary else CatBoostRegressor
    return lambda n: cls(iterations=n, learning_rate=LEARNING_RATE, random_seed=SEED,
                         verbose=False, allow_writing_files=False)


def capacity_arm(make, X, y, Xa, binary, transform):
    """Early-stopped rounds on an inner slice, then a refit on every work row."""
    y_fit = pd.Series(apply_target(transform, y) if transform != "none" else y)
    started = time.monotonic()
    rounds = _stopped_rounds(make, X, y_fit, binary)
    model = make(rounds).fit(X, y_fit)
    preds = model.predict_proba(Xa)[:, 1] if binary else model.predict(Xa)
    if transform != "none":
        preds = invert_target(transform, preds)
    return np.asarray(preds, dtype=float), rounds, round(time.monotonic() - started, 1)


def score(metric, y, p):
    if metric == "rmsle":
        p = np.clip(p, 0, None)
    return compute_metric(metric, y, p)


def paired(metric, y, base, other):
    """Bootstrap the difference other - base, signed so positive is better."""
    rng = np.random.default_rng(SEED)
    sign = 1 if greater_is_better(metric) else -1
    y, base, other = map(np.asarray, (y, base, other))
    diffs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2 and metric == "auc":
            continue
        diffs.append(sign * (score(metric, y[i], other[i]) - score(metric, y[i], base[i])))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"gain": round(sign * (score(metric, y, other) - score(metric, y, base)), 6),
            "ci95": [round(float(lo), 6), round(float(hi), 6)]}


def main(name: str) -> int:
    result = json.loads((HERE / f"result_{name}.json").read_text())
    run = Run(result["run_dir"])
    task = run.journal["task"]
    target, metric, task_type = task["target"], task["metric"], task["task_type"]
    binary = task_type == "binary"
    transform = default_target_transform(metric, task_type)
    audit = pd.read_parquet(OUT / name / "audit.parquet")
    y_audit = audit[target].to_numpy()

    selected = run.training_features()
    X_sel, y_sel = selected.drop(columns=[target]), selected[target]
    Xa_sel = run.bundle.feature_table(audit.drop(columns=[target]))[list(X_sel.columns)]

    work = pd.read_parquet(Path(run.run_dir) / "data" / "work.parquet")
    X_work, _ = _feature_frame(work, target)
    y_work = work[target]
    Xa_work = audit[list(X_work.columns)]

    shipped = np.asarray(run.bundle.predict(audit.drop(columns=[target])), dtype=float)
    if shipped.ndim == 2:
        shipped = shipped[:, -1]

    preds, meta = {"probe": shipped}, {}
    # The probe refitted here must reproduce what shipped, or the comparison is off.
    refit = fit_probe(X_sel, y_sel, task_type, metric, transform)
    refit_preds = np.asarray(probe_predict(refit, Xa_sel, task_type), dtype=float)
    if refit_preds.ndim == 2:
        refit_preds = refit_preds[:, -1]
    meta["probe_refit_max_abs_diff"] = float(np.max(np.abs(refit_preds - shipped)))

    for arm, make in (("lgbm_es", _lgbm(binary)), ("catboost_es", _catboost(binary))):
        preds[arm], rounds, secs = capacity_arm(make, X_sel, y_sel, Xa_sel, binary, transform)
        meta[arm] = {"rounds": rounds, "seconds": secs}

    work_probe = fit_probe(X_work, y_work, task_type, metric, transform)
    wp = np.asarray(probe_predict(work_probe, Xa_work, task_type), dtype=float)
    preds["work_probe"] = wp[:, -1] if wp.ndim == 2 else wp
    for arm, make in (("work_lgbm_es", _lgbm(binary)), ("work_catboost_es", _catboost(binary))):
        preds[arm], rounds, secs = capacity_arm(make, X_work, y_work, Xa_work, binary, transform)
        meta[arm] = {"rounds": rounds, "seconds": secs}

    out = {
        "dataset": name,
        "metric": metric,
        "run_dir": result["run_dir"],
        "selected_table": run.bundle._bundle.get("training_table"),
        "n_selected_features": int(X_sel.shape[1]),
        "n_work_features": int(X_work.shape[1]),
        "n_work_rows": int(len(X_sel)),
        "n_audit_rows": int(len(audit)),
        "run_evaluation_score": (run.journal.get("best") or {}).get("val_score"),
        "audit_scores": {arm: round(score(metric, y_audit, p), 6) for arm, p in preds.items()},
        "vs_probe": {arm: paired(metric, y_audit, preds["probe"], p)
                     for arm, p in preds.items() if arm != "probe"},
        "feature_gain": {
            "at_probe": paired(metric, y_audit, preds["work_probe"], preds["probe"]),
            "at_lgbm_es": paired(metric, y_audit, preds["work_lgbm_es"], preds["lgbm_es"]),
            "at_catboost_es": paired(metric, y_audit, preds["work_catboost_es"],
                                     preds["catboost_es"]),
        },
        "fits": meta,
    }
    (HERE / f"arms_{name}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
