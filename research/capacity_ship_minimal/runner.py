"""One run of the default product per dataset, with an audit slice it never sees.

    uv run python research/capacity_ship_minimal/runner.py house
    uv run python research/capacity_ship_minimal/runner.py porto

See README.md for the question and the preregistered decision rule.
"""

import json
from pathlib import Path
import shutil
import sys
import time

import pandas as pd
from sklearn.model_selection import train_test_split

import tabular_agent as ta
from tabular_agent.harness.agent import (
    GATEWAY_MAX_OUTPUT_TOKENS, REQUESTY_BASE_URL, load_api_key,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "_runs" / "capacity_ship_minimal"
HERE = Path(__file__).resolve().parent
MODEL = "fireworks/deepseek-v4.1-flash-1017"
AUDIT_FRACTION = 0.2
AUDIT_SEED = 0

PORTO_ABOUT = """Auto insurance policy holders, one row per policy holder. The target
is whether that policy holder filed a claim: 1 if they did, 0 if they did not.
About 3.6% of rows are claims, so the classes are heavily imbalanced.

The column names carry the data owner's own grouping, and it is the only
documentation this dataset has; the columns are otherwise anonymised:

  * the tag after `ps_` groups related features: `ind` (individual), `reg`
    (region), `car` (car), `calc` (calculated).
  * a `_bin` suffix means the feature is binary.
  * a `_cat` suffix means the feature is categorical: the integers are labels,
    not quantities, and their order carries no meaning.
  * a column with neither suffix is continuous or ordinal.

A value of -1 means the feature was missing for that observation. It is a
sentinel, not a measurement.

Scoring is AUC: only the order the predictions put the rows in matters."""

DATASETS = {
    "house": {
        "source": REPO / "house_prices",
        "target": "SalePrice",
        "metric": "rmsle",
        "stratify": False,
        "description": "data_description.txt",
        "prompt": (
            "Build the best safe, replayable regressor you can for house sale "
            "prices. The target is SalePrice. Work until further ideas stop "
            "producing real progress, then finalize."
        ),
    },
    "porto": {
        "source": REPO / "seguro_driver_prediction",
        "target": "target",
        "metric": "auc",
        "stratify": True,
        "description": None,
        "prompt": (
            "Build the best model you can for this dataset of insurance policy "
            "holders. The target is target. Work until you stop making progress, "
            "then finalize."
        ),
    },
}


def agent_model():
    """The gateway model with thinking on, as the archived DeepSeek runs used."""
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
    from pydantic_ai.providers.openai import OpenAIProvider

    return OpenAIChatModel(
        MODEL,
        provider=OpenAIProvider(base_url=REQUESTY_BASE_URL, api_key=load_api_key()),
        settings=OpenAIChatModelSettings(
            max_tokens=GATEWAY_MAX_OUTPUT_TOKENS,
            extra_body={"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
        ),
    )


def carve(name: str) -> Path:
    """Write the agent's folder (80% of train + the serving schema) and the audit slice."""
    cfg = DATASETS[name]
    root = OUT / name
    files = root / "files"
    audit_path = root / "audit.parquet"
    if audit_path.exists():
        return files
    files.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(cfg["source"] / "train.csv", low_memory=False)
    work, audit = train_test_split(
        train, test_size=AUDIT_FRACTION, random_state=AUDIT_SEED,
        stratify=train[cfg["target"]] if cfg["stratify"] else None,
    )
    work.to_csv(files / "train.csv", index=False)
    # Only the header: the serving schema is all inspect() needs from it.
    pd.read_csv(cfg["source"] / "test.csv", nrows=5).to_csv(files / "test.csv", index=False)
    if cfg["description"]:
        shutil.copy2(cfg["source"] / cfg["description"], files / cfg["description"])
    audit.to_parquet(audit_path, index=False)
    return files


def main(name: str) -> int:
    cfg = DATASETS[name]
    files = carve(name)
    agent = ta.Agent(model=agent_model(), run_root=OUT / name / "runs")
    plan = agent.inspect(files)
    plan.target = cfg["target"]
    plan.metric = cfg["metric"]
    if name == "porto":
        plan.notes = PORTO_ABOUT
    plan.serving_columns = list(plan.proposed_serving_columns or [])
    plan.accept_identifier_proposals()
    plan.split_on = False
    plan.reject_joins()
    plan.validate()

    started = time.time()
    run = agent.run(plan, prompt=cfg["prompt"])
    result = {
        "dataset": name,
        "model": MODEL,
        "run_dir": str(run.run_dir),
        "elapsed_min": round((time.time() - started) / 60, 2),
        "identifier_columns": plan.identifier_columns,
        "n_serving_columns": len(plan.serving_columns),
    }
    (HERE / f"result_{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
