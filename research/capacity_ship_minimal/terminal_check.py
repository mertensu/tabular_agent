"""Would the table the agent trusted have done better than the one that shipped?

    uv run python research/capacity_ship_minimal/terminal_check.py restaurant r9 t3

r9 shipped t4 (t3 plus AirREGI bookings, whose gain on validation came from the
1-day lead). The agent built on t3 instead and flagged the bookings as optimistic.
This takes the shipped recipe (model types, parameters, blend weights) and refits
it on the run's training + validation rows twice: on the shipped table, and on the
named ancestor's columns only. Both are scored on our test days, whose streams
hold only records made before the first test day.

No LLM calls; nothing is re-chosen.
"""

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import paired, score  # noqa: E402
from reservation_check import blend  # noqa: E402
from runner import DATASETS, HERE, OUT, audit_cut, streams_as_of  # noqa: E402

from tabular_agent.api import Run  # noqa: E402
from tabular_agent.taskspec import ResolvedSpec  # noqa: E402


def main(name: str, tag: str, ancestor: str) -> int:
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
    table = pd.read_parquet(root / b["training_table"])
    shipped = list(b["feature_names"])
    # The ancestor's own columns, as far as the shipped table carries them: the
    # shipped table is the ancestor plus later steps' columns.
    ancestor_cols = set(pd.read_parquet(root / "tables" / f"{ancestor}.parquet").columns)
    kept = [c for c in shipped if c in ancestor_cols]

    audit = pd.read_parquet(OUT / name / "audit.parquet")
    y = audit[cfg["target"]].to_numpy()
    streams = streams_as_of(cfg, audit_cut(cfg)) if run.bundle._plan.get("streams") else {}
    X_test = run.bundle.feature_table(audit.drop(columns=[cfg["target"]]), streams=streams)
    X_test = X_test.reset_index(drop=True)

    base = blend(run, spec, table, X_test, shipped)
    other = blend(run, spec, table, X_test, kept)
    groups = audit[cfg["group"]]
    day = (pd.to_datetime(audit[cfg["split_on"]]) - audit_cut(cfg)).dt.days.to_numpy()
    result = {
        "dataset": name, "tag": tag, "shipped_table": b["training_table"],
        "ancestor": ancestor, "dropped_columns": [c for c in shipped if c not in ancestor_cols],
        "test_days": {"shipped": round(score(spec.metric, y, base), 6),
                      ancestor: round(score(spec.metric, y, other), 6)},
        # Positive: the ancestor did better than what shipped.
        "ancestor_gain": paired(spec.metric, y, base, other, groups),
        "by_week": {
            str(w): {"shipped": round(score(spec.metric, y[day // 7 == w], base[day // 7 == w]), 4),
                     ancestor: round(score(spec.metric, y[day // 7 == w], other[day // 7 == w]), 4)}
            for w in sorted(set(day // 7))
        },
    }
    (HERE / f"terminal_check_{name}_{tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:4]))
