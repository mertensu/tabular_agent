"""Are the rulers' picks still apart once every table is refitted as it would ship?

    uv run python research/capacity_ship_minimal/rerank_refit.py restaurant r10

`rerank.py` scored every table on the test days from fits on the work rows only,
which end months before them on Restaurant, so stale history features sat in every
comparison. This keeps the picks exactly as `rerank_<name>_<tag>.json` made them
(on the validation rows, from work-only fits, as the run ranked) and changes only
how a table is scored on the test days: its lineage and each ruler's model are
refitted on training + validation rows, what ships since 0092. Writes
`rerank_refit_<name>_<tag>.json`. No LLM calls.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import OUT, _catboost, _lgbm, capacity_arm, paired, score  # noqa: E402

from tabular_agent.api import Run  # noqa: E402
from tabular_agent.harness.tools import _feature_frame, _load_lineage  # noqa: E402
from tabular_agent.metrics import greater_is_better  # noqa: E402
from tabular_agent.probe import default_target_transform, fit_probe, probe_predict  # noqa: E402
from tabular_agent.refit import refit_lineage  # noqa: E402
from tabular_agent.taskspec import ResolvedSpec  # noqa: E402

HERE = Path(__file__).resolve().parent
RULERS = ("probe", "lgbm_es", "catboost_es")


def _last_column(p):
    p = np.asarray(p, dtype=float)
    return p[:, -1] if p.ndim == 2 else p


def main(name: str, tag: str = "r10") -> int:
    from runner import DATASETS, audit_cut, streams_as_of

    cfg = DATASETS[name]
    ranked = json.loads((HERE / f"rerank_{name}_{tag}.json").read_text())
    run = Run(json.loads((HERE / f"result_{name}_{tag}.json").read_text())["run_dir"])
    b, j, plan = run.bundle._bundle, run.journal, run.bundle._plan
    metric, task_type = j["task"]["metric"], b["task_type"]
    target, split_on = cfg["target"], cfg["split_on"]
    spec = ResolvedSpec(
        target=target, task_type=task_type, metric=metric, positive_class=None,
        notes=None, column_notes=None, n_splits=plan.get("n_splits") or 5,
        seed=j["split"]["seed"], target_transform=b["target_transform"],
        objective=j["task"].get("objective"), forecast_horizon=j["split"]["forecast_horizon"],
        forecast_length=plan.get("forecast_length"),
    )
    binary = task_type == "binary"
    transform = default_target_transform(metric, task_type)
    higher = greater_is_better(metric)
    root = Path(run.run_dir)

    rows = pd.concat([pd.read_parquet(root / "data" / "work.parquet"),
                      pd.read_parquet(root / "data" / "val.parquet")], ignore_index=True)
    audit = pd.read_parquet(OUT / name / "audit.parquet")
    y = audit[target].to_numpy()
    groups = audit[cfg["group"]] if cfg.get("group") else None
    # The streams the run pinned (cut at the first test day): what the refit at
    # finalize reads, and what a forecast issued then could read.
    streams = streams_as_of(cfg, audit_cut(cfg)) if plan.get("streams") else {}
    serving = plan.get("serving_columns")

    paths = {"work": "data/work.parquet",
             **{t["id"]: t["path"] for t in j["tables"] if t.get("probe_score") is not None}}
    preds, scores = {}, []
    for entry in ranked["tables"]:
        tid = entry["table"]
        path = str(root / paths[tid])
        X_work, _ = _feature_frame(pd.read_parquet(path, columns=None).head(1), target)
        columns = list(X_work.columns)
        if tid == "work" and serving:
            columns = [c for c in columns if c in serving]
        table, lineage = refit_lineage(_load_lineage(path, task_type), rows, spec,
                                       order=split_on, streams=streams)
        X, yt = table[columns], table[target]
        # The refit's stored past ends where the test days begin: no known past.
        Xa = lineage.transform(audit.drop(columns=[target]), streams=streams)[columns]

        row = {"table": tid}
        fit = fit_probe(X, yt, task_type, metric, transform)
        preds[tid, "probe"] = _last_column(probe_predict(fit, Xa, task_type))
        for arm, make in (("lgbm_es", _lgbm(binary)), ("catboost_es", _catboost(binary))):
            preds[tid, arm], rounds, _ = capacity_arm(make, X, yt, Xa, binary, transform,
                                                      table[split_on])
            row[f"{arm}_rounds"] = rounds
        for m in RULERS:
            row[f"{m}_test"] = round(score(metric, y, preds[tid, m]), 6)
        scores.append(row)
        print(json.dumps(row), flush=True)

    frame = pd.DataFrame(scores).set_index("table")
    shipped = Path(b["training_table"]).stem
    picks = {r: ranked["picks"][r]["table"] for r in RULERS}
    out = {"dataset": name, "tag": tag, "metric": metric, "fitted_on": "work+validation",
           "picks_from": f"rerank_{name}_{tag}.json (validation, work-only fits)",
           "shipped_table": shipped, "tables": scores, "picks": {}}
    for r in RULERS:
        out["picks"][r] = {"table": picks[r],
                           **{f"test_{m}": frame.loc[picks[r], f"{m}_test"] for m in RULERS}}
        if r != "probe":
            # Under the ruler's own model, positive = its pick better than the probe's.
            out["picks"][r]["vs_probe_pick"] = paired(
                metric, y, preds[picks["probe"], r], preds[picks[r], r], groups)
            out["picks"][r]["vs_shipped_table"] = paired(
                metric, y, preds[shipped, r], preds[picks[r], r], groups)
    out["best_on_test_days"] = {
        m: {"table": (frame[f"{m}_test"].idxmax() if higher else frame[f"{m}_test"].idxmin()),
            "score": float(frame[f"{m}_test"].max() if higher else frame[f"{m}_test"].min())}
        for m in RULERS}
    (HERE / f"rerank_refit_{name}_{tag}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("picks", "best_on_test_days")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
