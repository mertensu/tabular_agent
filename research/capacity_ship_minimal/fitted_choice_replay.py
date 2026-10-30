"""Which table would finalize's fitted ruler have shipped on a finished run?

    uv run python research/capacity_ship_minimal/fitted_choice_replay.py restaurant r10

Copies the run directory, resumes it as a session and calls the product's own
choice (`tools._best_table_path`, which since the fitted ruler applies 0094 under
an early-stopped fit). The original run is not touched. Writes
`fitted_choice_<name>_<tag>.json`. No LLM calls.
"""

import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import DATASETS, HERE, OUT  # noqa: E402

from tabular_agent.api import Run  # noqa: E402
from tabular_agent.harness import journal, tools  # noqa: E402
from tabular_agent.harness.session import RunSession  # noqa: E402
from tabular_agent.taskspec import TaskSpec  # noqa: E402


def main(name: str, tag: str) -> int:
    cfg = DATASETS[name]
    run = Run(json.loads((HERE / f"result_{name}_{tag}.json").read_text())["run_dir"])
    copy = OUT / name / f"replay_{tag}"
    if copy.exists():
        shutil.rmtree(copy)
    shutil.copytree(run.run_dir, copy)

    j, plan = run.journal, run.bundle._plan
    split = j["split"]
    spec = TaskSpec(target=j["task"]["target"], metric=j["task"]["metric"],
                    seed=split["seed"], objective=j["task"].get("objective"),
                    forecast_horizon=split["forecast_horizon"])
    streams = {s: {**d, "path": str(OUT / name / "streams" / f"{s}.parquet")}
               for s, d in (plan.get("streams") or {}).items()}
    session = RunSession(
        str(copy), str(copy / "prepared" / Path(split["source_path"]).name), spec,
        val_fraction=split["val_fraction"], holdout_fraction=split["holdout_fraction"],
        split_on=split["split_on"], serving_columns=plan.get("serving_columns"),
        future_known=plan.get("future_known"), streams=streams or None,
        profile=split["profile"], workflow=split["workflow"],
    )
    rj = journal.load(session.run_dir)
    rj.pop("terminal_selection", None)
    started = time.monotonic()
    tools._best_table_path(session, rj)
    selection = journal.load(session.run_dir)["terminal_selection"]
    out = {"dataset": name, "tag": tag, "shipped": Path(run.bundle._bundle["training_table"]).stem,
           "minutes": round((time.monotonic() - started) / 60, 1), **selection}
    (HERE / f"fitted_choice_{name}_{tag}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
