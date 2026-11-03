"""One run of the default product on freMTPL2freq, with a test slice it never sees.

    uv run python research/fremtpl2_exposure/runner.py [tag]

20% of policies (seed 0, stratified on having a claim) are the test rows; the agent
gets the other 80%. Writes `result_<tag>.json` with the shipped bundle's mean Poisson
deviance on the test rows. See README.md.
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
OUT = REPO / "_runs" / "fremtpl2_exposure"
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "research" / "capacity_ship_minimal"))

import prep_fremtpl2 as prep  # noqa: E402

TEST_FRACTION = 0.2
PROMPT = (
    "Build the best model you can for this dataset of motor insurance policies. The "
    "target is ClaimNb. Work until you stop making progress, then finalize."
)
#: The agent model of the ForecastingAgent repeats and the credit default runs.
MODEL, EFFORT = "gpt-6.1-terra@eu", "medium"
#: Identifiers the owner declares per repeat. r1 accepted the harness's proposals,
#: which did not include IDpol (its name check misses a leading "ID"), and the agent
#: built features from it. r2 declares it, as an owner who knows the column would.
IDENTIFIERS = {"r2": ["IDpol"]}


def split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(training rows, test rows), the same for every run and every reference."""
    df = prep.load()
    train, test = train_test_split(df, test_size=TEST_FRACTION, random_state=0,
                                   stratify=df[prep.TARGET] > 0)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def main(tag: str = "r1") -> int:
    import tabular_agent as ta
    from runner import agent_model
    from tabular_agent.metrics import compute_metric

    train, test = split()
    files = OUT / tag / "files"
    files.mkdir(parents=True, exist_ok=True)
    train.to_csv(files / "train.csv", index=False)
    # The serving schema only: five training rows without the target.
    train.drop(columns=[prep.TARGET]).head(5).to_csv(files / "test.csv", index=False)

    agent = ta.Agent(model=agent_model(MODEL, EFFORT), run_root=OUT / tag / "runs",
                     holdout_fraction=None)
    plan = agent.inspect(files)
    plan.target = prep.TARGET
    plan.objective = "poisson"
    plan.metric = "poisson_deviance"
    plan.notes, plan.column_notes = prep.ABOUT, prep.COLUMN_NOTES
    train_cols = set(plan.tables[plan.training_table]["columns"])
    plan.serving_columns = [c for c in (plan.proposed_serving_columns or []) if c in train_cols]
    plan.accept_identifier_proposals()
    if tag in IDENTIFIERS:
        plan.identifier_columns = IDENTIFIERS[tag]
    plan.split_on = False
    plan.reject_joins()
    plan.validate()

    started = time.time()
    run = agent.run(plan, prompt=PROMPT)
    elapsed = time.time() - started
    p = np.asarray(run.bundle.predict(test.drop(columns=[prep.TARGET])), dtype=float)
    result = {
        "tag": tag,
        "model": MODEL,
        "reasoning_effort": EFFORT,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "run_dir": str(run.run_dir),
        "elapsed_min": round(elapsed / 60, 2),
        "identifier_columns": plan.identifier_columns,
        "n_train": len(train),
        "n_test": len(test),
        "test_poisson_deviance": round(float(compute_metric(
            "poisson_deviance", test[prep.TARGET].to_numpy(), p)), 6),
    }
    (HERE / f"result_{tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:2]))
