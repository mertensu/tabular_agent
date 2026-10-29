"""One run of the default product per dataset, with an audit slice it never sees.

    uv run python research/capacity_ship_minimal/runner.py house [tag]
    uv run python research/capacity_ship_minimal/runner.py porto [tag]
    uv run python research/capacity_ship_minimal/runner.py restaurant [tag]
    uv run python research/capacity_ship_minimal/runner.py rossmann [tag]

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
        "prep": "prep_restaurant",
        "identifiers": "accept",
        "prompt": (
            "Build the best model you can for this dataset of daily restaurant "
            "visitor counts. The target is visitors. Work until you stop making "
            "progress, then finalize."
        ),
    },
    "rossmann": {
        # Forecasting like Restaurant: the latest 48 days sealed (the length of the
        # competition's test block). Agent settings as in the earlier Rossmann
        # ForecastingAgent repeats (research/rossmann_forecasting_agent/), so the
        # reading is comparable to them; no future_known declared, as there.
        "source": REPO / "rossmann_store_sales",
        "target": "Sales",
        "metric": "rmspe",
        "forecast": True,
        "split_on": "Date",
        "group": "Store",
        "audit_days": 48,
        "future_known": [],
        "prep": "prep_rossmann",
        "identifiers": "reject",
        # Training ends 31 July 2015 and the competition block is 1 August to
        # 17 September: the first forecast day is 1 day out, the block 48 days long.
        "forecast_horizon": 1.0,
        "forecast_length": 48.0,
        "agent": {"val_fraction": 0.05, "fit_time_budget_s": 20 * 60},
        "prompt": (
            "Build the best safe, replayable model you can for this daily store-sales "
            "dataset. The target is Sales. The unseen competition block spans 48 days "
            "and no Sales values are revealed while it is forecast. Work until further "
            "feature ideas stop producing real progress, then finalize the run."
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


#: Per-repeat changes to a dataset's setup, so one repeat differs from another in
#: exactly the named way. Rossmann r2: the calendar columns the competition's test
#: file gives for the whole forecast block, declared known in advance (r1 declared
#: none). StateHoliday is left out: it is text ('0', 'a', 'b', 'c'), and the
#: future-window tool reads values as numbers, so its holidays would read as missing.
TAG_OVERRIDES = {
    ("rossmann", "r2"): {"future_known": ["Open", "Promo", "SchoolHoliday"]},
    # r2 was stopped before finalizing. r3 repeats it after 0093, with trailing
    # windows of these columns reaching back across the block start.
    ("rossmann", "r3"): {"future_known": ["Open", "Promo", "SchoolHoliday"]},
    # r4 repeats r3 after future-covariate windows are refused on a table that
    # went through a training-only filter (windows first, then the filter).
    ("rossmann", "r4"): {"future_known": ["Open", "Promo", "SchoolHoliday"]},
    # r5 repeats r4 with a larger agent model: does it find the closure windows
    # over Open (short windows around a closed day) that r4's agent missed?
    ("rossmann", "r5"): {"future_known": ["Open", "Promo", "SchoolHoliday"],
                         "model": "deepseek-v4-pro-0813@eu"},
    # r5 was stopped. r6 repeats it after the agent's texts stopped telling it to
    # aggregate future-known columns forward only.
    ("rossmann", "r6"): {"future_known": ["Open", "Promo", "SchoolHoliday"],
                         "model": "deepseek-v4-pro-0813@eu"},
    # r6 was stopped: inspect_data still said "aggregate them forward". r7 repeats
    # it with that note pared back too.
    ("rossmann", "r7"): {"future_known": ["Open", "Promo", "SchoolHoliday"],
                         "model": "deepseek-v4-pro-0813@eu"},
    # r8 repeats r7 with rejected tool calls recorded in the journal, so a run that
    # tried windows after a filter and was refused can be told from one that did not try.
    ("rossmann", "r8"): {"future_known": ["Open", "Promo", "SchoolHoliday"],
                         "model": "deepseek-v4-pro-0813@eu"},
    # r9: validation spans whole forecast horizons (r8's 43-day validation put
    # whole-block windows on a different scale), the agent texts say windows can
    # look back as well as forward, and the flash model again.
    ("rossmann", "r9"): {"future_known": ["Open", "Promo", "SchoolHoliday"],
                         "model": "deepseek-v4.1-flash-1017@eu"},
    # r9 was cut off after 12 tables by a provider timeout; r10 repeats it unchanged.
    ("rossmann", "r10"): {"future_known": ["Open", "Promo", "SchoolHoliday"],
                          "model": "deepseek-v4.1-flash-1017@eu"},
    # r11 repeats r10 with validation spanning at least two forecast horizons (96
    # days, not 48), so a table's gain is judged over two forecasts, and with
    # derived key columns checked by replay.
    ("rossmann", "r11"): {"future_known": ["Open", "Promo", "SchoolHoliday"],
                          "model": "deepseek-v4.1-flash-1017@eu"},
    # r12: the current code (0094 selection, serving types restored) with a stronger
    # agent model, at medium reasoning effort. Does it build the short calendar
    # windows the flash model never did?
    ("rossmann", "r12"): {"future_known": ["Open", "Promo", "SchoolHoliday"],
                          "model": "gpt-6.1-terra@eu", "reasoning_effort": "medium"},
    # Restaurant r3: r2's setup on the current code (0094, 0095, serving types
    # restored, two-horizon validation) with r12's stronger agent model.
    ("restaurant", "r3"): {"model": "gpt-6.1-terra@eu", "reasoning_effort": "medium"},
    # r4 repeats r3 with the two reservation files declared as streams (one row per
    # booking, with when it was made), now that the public plan can carry them.
    ("restaurant", "r4"): {"model": "gpt-6.1-terra@eu", "reasoning_effort": "medium",
                           "streams": True},
    # r5 repeats r4 unchanged: how far apart are two runs of one setup?
    ("restaurant", "r5"): {"model": "gpt-6.1-terra@eu", "reasoning_effort": "medium",
                           "streams": True},
    # r6 repeats r5 after history windows shorter than a fold's block are refused.
    ("restaurant", "r6"): {"model": "gpt-6.1-terra@eu", "reasoning_effort": "medium",
                           "streams": True},
    # r7 and r8 repeat r6 with the agent's first finalize returning an inventory of
    # the run (what would ship, what passed and was never built on, tools unused).
    ("restaurant", "r7"): {"model": "gpt-6.1-terra@eu", "reasoning_effort": "medium",
                           "streams": True},
    ("restaurant", "r8"): {"model": "gpt-6.1-terra@eu", "reasoning_effort": "medium",
                           "streams": True},
}


def agent_model(name: str = MODEL, effort: str = "high"):
    """The gateway model with reasoning on; DeepSeek also needs thinking enabled."""
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
    from pydantic_ai.providers.openai import OpenAIProvider

    return OpenAIChatModel(
        name,
        provider=OpenAIProvider(base_url=REQUESTY_BASE_URL, api_key=load_api_key()),
        settings=OpenAIChatModelSettings(
            max_tokens=GATEWAY_MAX_OUTPUT_TOKENS,
            extra_body=(
                {"thinking": {"type": "enabled"}, "reasoning_effort": effort}
                if "deepseek" in name else {"reasoning_effort": effort}
            ),
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


def audit_cut(cfg: dict) -> pd.Timestamp:
    """The first of the test days: the agent's rows end before it."""
    when = pd.read_parquet(cfg["source"] / "train_joined.parquet", columns=[cfg["split_on"]])
    return when[cfg["split_on"]].max().normalize() - pd.Timedelta(days=cfg["audit_days"] - 1)


def streams_as_of(cfg: dict, cut: pd.Timestamp | None = None) -> dict[str, pd.DataFrame]:
    """The dataset's declared streams, holding only records known before ``cut``.

    ``None`` keeps every record: the files end where the competition's forecast is
    issued. Cut at the test days, they are what a forecast issued then could read.
    """
    import importlib

    sys.path.insert(0, str(REPO / "scripts"))
    declared = importlib.import_module(cfg["prep"]).STREAMS
    out = {}
    for name, spec in declared.items():
        df = pd.read_parquet(spec["path"])
        out[name] = df if cut is None else df[df[spec["observed_at"]] < cut].reset_index(drop=True)
    return out


def _run_streams(name: str, cfg: dict) -> dict[str, dict]:
    """The streams as the agent's plan declares them: cut where its rows end."""
    import importlib

    sys.path.insert(0, str(REPO / "scripts"))
    declared = importlib.import_module(cfg["prep"]).STREAMS
    cut = audit_cut(cfg)
    dest = OUT / name / "streams"
    dest.mkdir(parents=True, exist_ok=True)
    plan_streams = {}
    for stream, frame in streams_as_of(cfg, cut).items():
        path = dest / f"{stream}.parquet"
        frame.to_parquet(path, index=False)
        last = (cut - pd.Timedelta(days=1)).date().isoformat()
        notes = declared[stream].get("notes", "").replace("2017-04-22", last)
        plan_streams[stream] = {**declared[stream], "path": str(path), "notes": notes}
    return plan_streams


def _carve_forecast(cfg: dict, files: Path, audit_path: Path) -> None:
    """Agent gets everything before the last ``audit_days``; the audit slice is those days."""
    train = pd.read_parquet(cfg["source"] / "train_joined.parquet")
    when = train[cfg["split_on"]]
    cut = audit_cut(cfg)
    work, audit = train[when < cut], train[when >= cut].copy()
    if cfg["prep"] == "prep_restaurant":
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
    cfg = {**DATASETS[name], **TAG_OVERRIDES.get((name, tag), {})}
    files = carve(name)
    # holdout_fraction=None: the audit slice carved above is this experiment's
    # sealed rows, and r1 ran before the product sealed one of its own.
    forecast = cfg.get("forecast", False)
    cls = ta.ForecastingAgent if forecast else ta.Agent
    agent = cls(model=agent_model(cfg.get("model", MODEL), cfg.get("reasoning_effort", "high")), run_root=OUT / name / "runs",
                holdout_fraction=None,
                **cfg.get("agent", {}))
    plan = agent.inspect(files)
    plan.target = cfg["target"]
    plan.metric = cfg["metric"]
    if name == "porto":
        plan.notes = PORTO_ABOUT
    # Only columns the training table has: a serving file's own row id is not one.
    train_cols = set(plan.tables[plan.training_table]["columns"])
    plan.serving_columns = [c for c in (plan.proposed_serving_columns or []) if c in train_cols]
    if cfg.get("identifiers", "accept") == "accept":
        plan.accept_identifier_proposals()
    else:
        plan.reject_identifier_proposals()
    if forecast:
        import importlib

        sys.path.insert(0, str(REPO / "scripts"))
        prep = importlib.import_module(cfg["prep"])
        plan.notes, plan.column_notes = prep.ABOUT, prep.COLUMN_NOTES
        plan.split_on = cfg["split_on"]
        plan.forecast_horizon = cfg.get("forecast_horizon", getattr(prep, "FORECAST_HORIZON", None))
        plan.future_known = cfg["future_known"]
        if cfg.get("streams"):
            plan.streams = _run_streams(name, cfg)
    else:
        plan.split_on = False
    plan.reject_joins()
    plan.validate()

    started = time.time()
    run = agent.run(plan, prompt=cfg["prompt"])
    result = {
        "dataset": name,
        "tag": tag,
        "model": cfg.get("model", MODEL),
        "git_commit": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "run_dir": str(run.run_dir),
        "elapsed_min": round((time.time() - started) / 60, 2),
        "identifier_columns": plan.identifier_columns,
        "n_serving_columns": len(plan.serving_columns),
        "future_known": list(plan.future_known or []),
        plan.forecast_length = cfg.get("forecast_length", getattr(prep, "FORECAST_LENGTH", None))
        "streams": sorted(plan.streams),
    }
    (HERE / f"result_{name}_{tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
