"""Why Restaurant's audit score is so far from the run's own: skew or stale history?

    uv run python research/capacity_ship_minimal/restaurant_diag.py r1

1. **Reservation skew.** Score the shipped bundle on the audit days twice: with
   reservation columns as of the cut (as serving sees them) and with every booking
   counted (as training rows see them).
2. **Stale data.** The shipped models and their history encoders were fitted on the
   work rows only, which end about three months before the audit days. Replay the
   selected table's lineage, with the same parameters, on work + evaluation rows,
   finalize that the way the product does, and score it on the same audit rows.

Writes `diag_restaurant_<tag>.json`. No LLM calls.
"""

import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import OUT, _lgbm, capacity_arm, paired, score  # noqa: E402

from tabular_agent.api import Run  # noqa: E402
from tabular_agent.harness import tools  # noqa: E402
from tabular_agent.harness.session import RunSession  # noqa: E402
from tabular_agent.harness.tools import _feature_frame  # noqa: E402
from tabular_agent.taskspec import TaskSpec  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TARGET, METRIC, GROUP, ORDER = "visitors", "rmsle", "air_store_id", "visit_date"


def replay_on_all_rows(run: Run, dest: Path) -> Run:
    """The selected table's lineage, refitted step by step on work + evaluation rows."""
    root = Path(run.run_dir)
    manifest = json.loads((root / "bundle" / "api_plan.json").read_text())
    j = run.journal
    if dest.exists():
        shutil.rmtree(dest)
    session = RunSession(
        str(dest), str(root / "prepared" / "train.parquet"),
        TaskSpec(target=TARGET, metric=METRIC,
                 forecast_horizon=manifest["forecast_horizon"]),
        # The smallest evaluation slice the split allows: the point is to fit on
        # (nearly) every row, the way a model would be refreshed before serving.
        val_fraction=0.001, split_on=ORDER,
        serving_columns=manifest["serving_columns"],
        future_known=manifest["future_known"],
        profile="full", workflow="forecast",
    )
    tools.inspect_data(session)
    by_path = {t["path"]: t for t in j["tables"]}
    chain, path = [], run.bundle._bundle["training_table"]
    while path in by_path:
        chain.append(by_path[path])
        path = by_path[path]["source_table"]
    lineage = json.loads((root / f"{run.bundle._bundle['training_table'][:-len('.parquet')]}"
                          ".pipeline.json").read_text())
    steps = lineage.get("steps") or lineage["pipeline"]["steps"]
    table = None
    for step, rec in zip(steps, reversed(chain)):
        if step["type"] == "transform":
            out = tools.transform_features(session, step["code"], table=table)
        elif step["type"] == "filter":
            out = tools.filter_rows(session, step["code"], table=table)
        elif step["encoder"]["kind"] == "history":
            e = step["encoder"]
            out = tools.fit_history(session, by=e["by"], value=e["value"], keep=e.get("keep"),
                                    windows=e["windows"], table=table)
        else:
            e = step["encoder"]
            out = tools.fit_encoding(session, columns=e["columns"], method=e["kind"],
                                     smoothing=e.get("smoothing", 10.0),
                                     stats=e.get("stats"), table=table)
        table = out["table"]
        print(f"replayed {rec['id']} ({rec['method']}) -> {table}", flush=True)
    tools.finalize_run(session)
    shutil.copy2(root / "bundle" / "api_plan.json", dest / "bundle" / "api_plan.json")
    return Run(dest)


def main(tag: str = "r1") -> int:
    result = json.loads((HERE / f"result_restaurant_{tag}.json").read_text())
    run = Run(result["run_dir"])
    audit = pd.read_parquet(OUT / "restaurant" / "audit.parquet")
    y = audit[TARGET].to_numpy()
    groups = audit[GROUP]

    full = pd.read_parquet(REPO / "restaurant_visitor" / "train_joined.parquet")
    uncut = full[full[ORDER] >= audit[ORDER].min()].reset_index(drop=True)
    assert (uncut[[GROUP, ORDER]].values == audit[[GROUP, ORDER]].values).all()

    shipped_cut = np.asarray(run.bundle.predict(audit.drop(columns=[TARGET])), dtype=float)
    shipped_uncut = np.asarray(run.bundle.predict(uncut.drop(columns=[TARGET])), dtype=float)

    refit = replay_on_all_rows(run, OUT / "restaurant" / f"refit_{tag}")
    refit_cut = np.asarray(refit.bundle.predict(audit.drop(columns=[TARGET])), dtype=float)

    # The single fitted model that beat the blend, on the refitted table too.
    sel = refit.training_features()
    table = pd.read_parquet(Path(refit.run_dir) / refit.bundle._bundle["training_table"])
    Xa = refit.bundle.feature_table(audit.drop(columns=[TARGET]))[list(sel.columns[:-1])]
    lgbm_refit, rounds, _ = capacity_arm(_lgbm(False), sel.drop(columns=[TARGET]), sel[TARGET],
                                         Xa, False, "log1p", table[ORDER])

    preds = {"shipped": shipped_cut, "shipped_uncut_reservations": shipped_uncut,
             "refit_all_rows": refit_cut, "refit_all_rows_lgbm_es": lgbm_refit}
    out = {
        "tag": tag,
        "run_evaluation_score": (run.journal.get("best") or {}).get("val_score"),
        "refit_run_dir": refit.run_dir,
        "refit_rows": int(len(sel)),
        "refit_lgbm_es_rounds": rounds,
        "audit_scores": {k: round(score(METRIC, y, p), 6) for k, p in preds.items()},
        "vs_shipped": {k: paired(METRIC, y, shipped_cut, p, groups)
                       for k, p in preds.items() if k != "shipped"},
    }
    (HERE / f"diag_restaurant_{tag}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:2]))
