"""One run of the default product per dataset, with an audit slice it never sees.

    uv run python research/capacity_ship_minimal/runner.py house [tag]
    uv run python research/capacity_ship_minimal/runner.py porto [tag]

``tag`` (default r1) names the repeat; every repeat shares one audit slice.

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
    "restaurant": {
        # Forecasting: ForecastingAgent, full profile, the latest 39 days sealed.
        "source": REPO / "restaurant_visitor",
        "target": "visitors",
        "metric": "rmsle",
        "forecast": True,
        "split_on": "visit_date",
        "group": "air_store_id",
        "audit_days": 39,
        "future_known": ["holiday_flg"],
        "prompt": (
            "Build the best model you can for this dataset of daily restaurant "
            "visitor counts. The target is visitors. Work until you stop making "
            "progress, then finalize."
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


def _reservations_as_of(cut: pd.Timestamp) -> pd.DataFrame:
    """The prep script's reservation columns, counting only bookings made before ``cut``.

    The competition's forecast rows only ever see bookings made before the forecast
    was issued; training rows see every booking. Rebuilding the audit rows this way
    makes them look like serving rows rather than like more training rows.
    """
    src = REPO / "restaurant_visitor"
    parts = []
    for prefix in ("air", "hpg"):
        r = pd.read_parquet(src / f"{prefix}_reserve_stream.parquet")
        r = r[r["reserve_datetime"] < cut].copy()
        r["visit_date"] = r["visit_datetime"].dt.normalize()
        r["lead_days"] = (r["visit_date"] - r["reserve_datetime"].dt.normalize()).dt.days
        parts.append(r.groupby(["air_store_id", "visit_date"]).agg(**{
            f"{prefix}_reserve_visitors": ("reserve_visitors", "sum"),
            f"{prefix}_reserve_count": ("reserve_visitors", "size"),
            f"{prefix}_reserve_lead_days": ("lead_days", "mean"),
        }))
    return parts[0].join(parts[1], how="outer").reset_index()


def _carve_forecast(cfg: dict, files: Path, audit_path: Path) -> None:
    """Agent gets everything before the last ``audit_days``; the audit slice is those days."""
    train = pd.read_parquet(cfg["source"] / "train_joined.parquet")
    when = train[cfg["split_on"]]
    cut = when.max().normalize() - pd.Timedelta(days=cfg["audit_days"] - 1)
    work, audit = train[when < cut], train[when >= cut].copy()
    rebuilt = _reservations_as_of(cut)
    cols = [c for c in rebuilt.columns if c not in ("air_store_id", "visit_date")]
    audit = audit.drop(columns=cols).merge(
        rebuilt, on=["air_store_id", "visit_date"], how="left")[train.columns]
    work.to_parquet(files / "train.parquet", index=False)
    test = pd.read_parquet(cfg["source"] / "test_joined.parquet").head(5)
    test.to_parquet(files / "test.parquet", index=False)
    audit.to_parquet(audit_path, index=False)


def carve(name: str) -> Path:
    """Write the agent's folder (80% of train + the serving schema) and the audit slice."""
    cfg = DATASETS[name]
    root = OUT / name
    files = root / "files"
    audit_path = root / "audit.parquet"
    if audit_path.exists():
        return files
    files.mkdir(parents=True, exist_ok=True)
    if cfg.get("forecast"):
        _carve_forecast(cfg, files, audit_path)
        return files
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


def main(name: str, tag: str = "r1") -> int:
    cfg = DATASETS[name]
    files = carve(name)
    # holdout_fraction=None: the audit slice carved above is this experiment's
    # sealed rows, and r1 ran before the product sealed one of its own.
    forecast = cfg.get("forecast", False)
    cls = ta.ForecastingAgent if forecast else ta.Agent
    agent = cls(model=agent_model(), run_root=OUT / name / "runs", holdout_fraction=None)
    plan = agent.inspect(files)
    plan.target = cfg["target"]
    plan.metric = cfg["metric"]
    if name == "porto":
        plan.notes = PORTO_ABOUT
    # Only columns the training table has: a serving file's own row id is not one.
    train_cols = set(plan.tables[plan.training_table]["columns"])
    plan.serving_columns = [c for c in (plan.proposed_serving_columns or []) if c in train_cols]
    plan.accept_identifier_proposals()
    if forecast:
        sys.path.insert(0, str(REPO / "scripts"))
        from prep_restaurant import ABOUT, COLUMN_NOTES, FORECAST_HORIZON

        plan.notes, plan.column_notes = ABOUT, COLUMN_NOTES
        plan.split_on = cfg["split_on"]
        plan.forecast_horizon = FORECAST_HORIZON
        plan.future_known = cfg["future_known"]
    else:
        plan.split_on = False
    plan.reject_joins()
    plan.validate()

    started = time.time()
    run = agent.run(plan, prompt=cfg["prompt"])
    result = {
        "dataset": name,
        "tag": tag,
        "model": MODEL,
        "run_dir": str(run.run_dir),
        "elapsed_min": round((time.time() - started) / 60, 2),
        "identifier_columns": plan.identifier_columns,
        "n_serving_columns": len(plan.serving_columns),
    }
    (HERE / f"result_{name}_{tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
