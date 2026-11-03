"""The default product on UCI credit default, on the design of a TabICL-vs-XGBoost post.

    uv run python research/credit_default_tabfm/runner.py 1000 0
    uv run python research/credit_default_tabfm/runner.py all

One fixed test set of 6,000 clients (stratified, seed 0) that no run sees; the
training rows are a stratified sample of n from the other 24,000, drawn with the
run's seed (n = 24,000 is all of them, so the seed only reruns the agent).
Writes `result_<n>_s<seed>.json` with the shipped bundle's test AUC.
See README.md.
"""

import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = REPO / "_runs" / "credit_default_tabfm"
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "research" / "capacity_ship_minimal"))

import prep_credit  # noqa: E402

SIZES = (1000, 5000, 24000)
SEEDS = (0, 1, 2)
TEST_ROWS = 6000
PROMPT = (
    "Build the best model you can for this dataset of credit card clients. The "
    "target is default_next_month. Work until you stop making progress, then finalize."
)
#: The agent model of the ForecastingAgent repeats (Rossmann r12-r14, Restaurant r3-r11).
MODEL, EFFORT = "gpt-6.1-terra@eu", "medium"


def split(n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(training rows, test rows) for one cell of the grid."""
    df = prep_credit.load()
    y = df[prep_credit.TARGET]
    pool, test = train_test_split(df, test_size=TEST_ROWS, random_state=0, stratify=y)
    if n < len(pool):
        pool, _ = train_test_split(pool, train_size=n, random_state=seed,
                                   stratify=pool[prep_credit.TARGET])
    return pool.reset_index(drop=True), test.reset_index(drop=True)


def main(n: int, seed: int) -> int:
    import tabular_agent as ta
    from runner import agent_model
    from tabular_agent.metrics import compute_metric

    train, test = split(n, seed)
    files = OUT / f"n{n}_s{seed}" / "files"
    files.mkdir(parents=True, exist_ok=True)
    train.to_csv(files / "train.csv", index=False)
    # The serving schema only: five training rows without the target.
    train.drop(columns=[prep_credit.TARGET]).head(5).to_csv(files / "test.csv", index=False)

    agent = ta.Agent(model=agent_model(MODEL, EFFORT), run_root=OUT / f"n{n}_s{seed}" / "runs",
                     holdout_fraction=None)
    plan = agent.inspect(files)
    plan.target = prep_credit.TARGET
    plan.metric = "auc"
    plan.notes, plan.column_notes = prep_credit.ABOUT, prep_credit.COLUMN_NOTES
    train_cols = set(plan.tables[plan.training_table]["columns"])
    plan.serving_columns = [c for c in (plan.proposed_serving_columns or []) if c in train_cols]
    plan.accept_identifier_proposals()
    plan.split_on = False
    plan.reject_joins()
    plan.validate()

    started = time.time()
    run = agent.run(plan, prompt=PROMPT)
    elapsed = time.time() - started
    p = np.asarray(run.bundle.predict(test.drop(columns=[prep_credit.TARGET])), dtype=float)
    if p.ndim == 2:
        p = p[:, -1]
    result = {
        "n_train": n,
        "seed": seed,
        "model": MODEL,
        "reasoning_effort": EFFORT,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "run_dir": str(run.run_dir),
        "elapsed_min": round(elapsed / 60, 2),
        "identifier_columns": plan.identifier_columns,
        "test_auc": round(float(compute_metric("auc", test[prep_credit.TARGET].to_numpy(), p)), 6),
    }
    (HERE / f"result_{n}_s{seed}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "all":
        for n in SIZES:
            for s in SEEDS:
                if not (HERE / f"result_{n}_s{s}.json").exists():
                    subprocess.run([sys.executable, __file__, str(n), str(s)], check=False)
        raise SystemExit(0)
    raise SystemExit(main(int(sys.argv[1]), int(sys.argv[2])))
