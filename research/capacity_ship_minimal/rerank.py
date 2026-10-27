"""Would a fitted model have picked the same table as the 200-round probe?

    uv run python research/capacity_ship_minimal/rerank.py house [tag]

Every table the run scored is refitted three ways (the probe, early-stopped
LightGBM, early-stopped CatBoost) and scored on the run's own evaluation rows,
which is what ranked tables during the run, and on the audit slice, which nothing
in the run read. Each ruler's pick is the table it scores best on the evaluation
rows; the audit slice then says what each pick is worth. Writes
`rerank_<name>_<tag>.json`. No LLM calls.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import OUT, _catboost, _lgbm, capacity_arm, score  # noqa: E402

from tabular_agent.api import Run  # noqa: E402
from tabular_agent.harness.tools import _feature_frame, _load_lineage  # noqa: E402
from tabular_agent.metrics import greater_is_better  # noqa: E402
from tabular_agent.probe import default_target_transform, fit_probe, probe_predict  # noqa: E402

HERE = Path(__file__).resolve().parent


def _last_column(p):
    p = np.asarray(p, dtype=float)
    return p[:, -1] if p.ndim == 2 else p


def main(name: str, tag: str = "r1") -> int:
    from runner import DATASETS

    cfg = DATASETS[name]
    split_on = cfg.get("split_on") if cfg.get("forecast") else None
    result = json.loads((HERE / f"result_{name}_{tag}.json").read_text())
    run = Run(result["run_dir"])
    j = run.journal
    target, metric, task_type = j["task"]["target"], j["task"]["metric"], j["task"]["task_type"]
    binary = task_type == "binary"
    transform = default_target_transform(metric, task_type)
    higher = greater_is_better(metric)
    root = Path(run.run_dir)

    val = pd.read_parquet(root / "data" / "val.parquet")
    audit = pd.read_parquet(OUT / name / "audit.parquet")
    tables = [{"id": "work", "path": "data/work.parquet",
               "probe_score": j["work_probe"]["probe_score"]}]
    tables += [t for t in j["tables"] if t.get("probe_score") is not None]

    rows = []
    for t in tables:
        path = str(root / t["path"])
        df = pd.read_parquet(path)
        X, _ = _feature_frame(df, target)
        y = df[target]
        lineage = _load_lineage(path, task_type)
        Xv = lineage.transform(val.drop(columns=[target]))[list(X.columns)]
        Xa = lineage.transform(audit.drop(columns=[target]))[list(X.columns)]

        row = {"table": t["id"], "n_features": int(X.shape[1]),
               "journal_probe_score": t["probe_score"]}
        fit = fit_probe(X, y, task_type, metric, transform)
        row["probe_val"] = score(metric, val[target], _last_column(probe_predict(fit, Xv, task_type)))
        row["probe_audit"] = score(metric, audit[target], _last_column(probe_predict(fit, Xa, task_type)))
        for arm, make in (("lgbm_es", _lgbm(binary)), ("catboost_es", _catboost(binary))):
            # One fit, predicting both row sets: stack them and split the output.
            both = pd.concat([Xv, Xa], ignore_index=True)
            preds, _, _ = capacity_arm(make, X, y, both, binary, transform,
                                       df[split_on] if split_on else None)
            row[f"{arm}_val"] = score(metric, val[target], preds[: len(Xv)])
            row[f"{arm}_audit"] = score(metric, audit[target], preds[len(Xv):])
        rows.append({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})
        print(json.dumps(rows[-1]), flush=True)

    frame = pd.DataFrame(rows)
    out = {"dataset": name, "tag": tag, "metric": metric, "tables": rows,
           "probe_reproduces_journal": bool(
               np.allclose(frame["probe_val"], frame["journal_probe_score"], atol=1e-6)),
           "picks": {}, "rank_agreement_spearman": {}}
    for ruler in ("probe", "lgbm_es", "catboost_es"):
        best = frame[f"{ruler}_val"].idxmax() if higher else frame[f"{ruler}_val"].idxmin()
        out["picks"][ruler] = {
            "table": frame.loc[best, "table"],
            # The pick's worth under each shipped model, on rows nothing read.
            **{f"audit_{m}": frame.loc[best, f"{m}_audit"]
               for m in ("probe", "lgbm_es", "catboost_es")},
        }
        if ruler != "probe":
            rho = spearmanr(frame["probe_val"], frame[f"{ruler}_val"]).statistic
            out["rank_agreement_spearman"][ruler] = round(float(rho), 3)
    for m in ("probe", "lgbm_es", "catboost_es"):
        col = frame[f"{m}_audit"]
        out.setdefault("best_possible_on_audit", {})[m] = {
            "table": frame.loc[col.idxmax() if higher else col.idxmin(), "table"],
            "score": float(col.max() if higher else col.min()),
        }
    (HERE / f"rerank_{name}_{tag}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("picks", "rank_agreement_spearman",
                                          "probe_reproduces_journal")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
