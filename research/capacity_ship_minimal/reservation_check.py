"""Would the collapsed reservation columns have helped a run that dropped them?

    uv run python research/capacity_ship_minimal/reservation_check.py restaurant r6

The run's agent dropped the six reservation columns (their training values count
same-day bookings, a forecast's cannot). This takes the shipped recipe (model
types, parameters, blend weights) and refits it on the run's training +
validation rows twice: on the shipped table as it is, and with the six columns
added. Both are scored on our test days, whose reservation columns hold only the
bookings made before the first test day, as a forecast issued then would see.

No LLM calls; nothing is re-chosen.
"""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import paired, score  # noqa: E402
from runner import DATASETS, HERE, OUT, audit_cut, streams_as_of  # noqa: E402

from tabular_agent.api import Run  # noqa: E402
from tabular_agent.harness.tools import _fit_recipe  # noqa: E402
from tabular_agent.models import fit_final_model  # noqa: E402
from tabular_agent.pipeline import predict_model  # noqa: E402
from tabular_agent.target import invert_reference_target, invert_target  # noqa: E402
from tabular_agent.taskspec import ResolvedSpec  # noqa: E402

RESERVATION = ["air_reserve_visitors", "air_reserve_count", "air_reserve_lead_days",
               "hpg_reserve_visitors", "hpg_reserve_count", "hpg_reserve_lead_days"]
KEY = ["air_store_id", "visit_date"]


def blend(run: Run, spec, table: pd.DataFrame, X_test: pd.DataFrame, features: list[str]):
    b = run.bundle._bundle
    preds = []
    for exp_id in b["fitted_on"]["refit_of"]:
        meta = json.loads((Path(run.run_dir) / "experiments" / exp_id / "meta.json").read_text())
        r = _fit_recipe(SimpleNamespace(resolved=lambda: spec), meta)
        model = fit_final_model(
            table[features], table[spec.target], spec.task_type, spec.metric,
            model_type=r["model_type"], params=r["params"] or None, seed=r["seed"],
            target_transform=r["target_transform"], reference_target=r["reference_target"],
            objective=spec.objective,
        )
        p = predict_model(model, X_test[features], spec.task_type)
        if b["target_transform"] != "none":
            p = invert_target(b["target_transform"], p)
        preds.append(invert_reference_target(p, X_test, b.get("reference_target")))
    return np.clip(np.asarray(b["weights"]) @ np.array(preds), 0, None)


def main(name: str, tag: str) -> int:
    cfg = DATASETS[name]
    run = Run(json.loads((HERE / f"result_{name}_{tag}.json").read_text())["run_dir"])
    b, j = run.bundle._bundle, run.journal
    spec = ResolvedSpec(
        target=cfg["target"], task_type=b["task_type"], metric=j["task"]["metric"],
        positive_class=None, notes=None, column_notes=None, n_splits=5,
        seed=j["split"]["seed"], target_transform=b["target_transform"],
        objective=j["task"].get("objective"), forecast_horizon=j["split"]["forecast_horizon"],
    )
    root = Path(run.run_dir)
    raw = pd.concat([pd.read_parquet(root / "data" / "work.parquet"),
                     pd.read_parquet(root / "data" / "val.parquet")], ignore_index=True)
    table = pd.read_parquet(root / b["training_table"])
    table = table.merge(raw[KEY + RESERVATION], on=KEY, how="left", validate="one_to_one")

    audit = pd.read_parquet(OUT / name / "audit.parquet")
    y = audit[cfg["target"]].to_numpy()
    streams = streams_as_of(cfg, audit_cut(cfg)) if run.bundle._plan.get("streams") else {}
    X_test = run.bundle.feature_table(audit.drop(columns=[cfg["target"]]), streams=streams)
    X_test = X_test.reset_index(drop=True)
    X_test[RESERVATION] = audit[RESERVATION].to_numpy()

    shipped = list(b["feature_names"])
    base = blend(run, spec, table, X_test, shipped)
    added = blend(run, spec, table, X_test, shipped + RESERVATION)
    groups = audit[cfg["group"]]
    day = (pd.to_datetime(audit[cfg["split_on"]]) - audit_cut(cfg)).dt.days.to_numpy()
    result = {
        "dataset": name, "tag": tag, "table": b["training_table"],
        "test_days": {"shipped_recipe": round(score(spec.metric, y, base), 6),
                      "with_reservation_columns": round(score(spec.metric, y, added), 6)},
        "gain": paired(spec.metric, y, base, added, groups),
        "by_week": {
            str(w): {"shipped_recipe": round(score(spec.metric, y[day // 7 == w], base[day // 7 == w]), 4),
                     "with_reservation_columns": round(score(spec.metric, y[day // 7 == w], added[day // 7 == w]), 4)}
            for w in sorted(set(day // 7))
        },
    }
    (HERE / f"reservation_check_{name}_{tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
