"""IEEE-CIS Fraud Detection v1: the first run on the fifth dataset, at defaults.

**Nothing is changed for this dataset.** Model, thinking budget, both split
fractions, the fit budget, the stop policy, the prompts and the tool surface are
what every other run has used. That is the point: the dataset is the instrument
and the library is the subject (`CLAUDE.md`), so the run has to be the same run.

## Why this dataset exists in the repo at all

Rossmann's private board is spent (0045) and Restaurant's is spent for the
question it was kept for (0066). A harness with no external board has no way to
check its own answer, which is a worse position than being 0.008 above a median.
This competition is finished and **both boards are on disk**, so one run gets one
honest reading — and then this board is spent too, deliberately, after **one**
submission.

## What is different about this data, and none of it is a special case

* **590,540 rows x 434 columns**, an order of magnitude wider than anything this
  harness has been pointed at. The profile the agent reads is capped at 100
  columns (`MAX_REPORTED_COLUMNS`).
* **AUC**, already in `SUPPORTED_METRICS`, and the first *higher-is-better*
  metric to reach a competition run here.
* **3.50% positive.** Every difference under ~0.005 AUC on this target is noise,
  which is the gotcha Porto Seguro cost a session to learn (0073).
* **A real 30-day gap** between the end of training and the first scored row,
  declared as `forecast_horizon=2_592_000` — in the ordering column's units,
  because `TransactionDT` is in seconds (0051). This puts 0064 under test: it
  refused a `gap` argument on `fit_history` on the grounds that the asymmetry
  belongs to the evaluation geometry, and here the geometry has one.
* **49 columns the competition declares categorical**, of which 31 are strings
  that `_feature_frame` drops and 18 are numeric dtype that survive and are split
  as ordered quantities (0073, 0074). This run is the first sighting of that
  defect in the wild rather than in a probe script.

## Registered before the run

Measured beforehand, so the predictions have something to be wrong about: the raw
work table, put through the probe unchanged, scores **val AUC 0.89125**. The
private median is 0.91096 and the private best is 0.94588.

1. **The identity block will be invisible, and the agent will not encode any of
   it.** `inspect_data` truncates the per-column profile at 100 columns; column
   100 of this table is `V45`, so all 41 identity columns — including 29 of the 49
   declared categoricals, `DeviceInfo` among them — fall off the end. They are not
   *entirely* silent: `non_numeric_columns` is not truncated, so the agent sees
   `DeviceType`, `DeviceInfo` and `id_12`..`id_38` listed as dropped, with no
   dtype, no cardinality and no description beside them. **So this is a clean test
   of whether a bare name in a "dropped" list is enough to act on.** If the agent
   encodes an identity column anyway, the cap is survivable and the prediction is
   wrong.

2. **It will encode the string categoricals it *can* see** — `ProductCD`, `card4`,
   `card6`, both email domains, `M1`-`M9` — with `frequency` or `target`. They are
   profiled, described, and `inspect_data`'s closing note names `fit_encoding`.
   This is the prediction that should be easiest; if it fails, the surfacing
   failed and everything after it is uninterpretable.

3. **Nothing will touch `card1`, `addr1`, `card2` or `id_19`.** This is the sharp
   one. They are numeric dtype, so they are already in the model, and nothing
   anywhere reports a problem with them — the run's own instruments are silent by
   construction. Acting would mean reading *"the competition declares this column
   categorical"* against a column already being used, and choosing to re-express
   it. 0074 says the defect is not that label encoding is bad but that **nobody
   chose it**; if the agent chooses it, that record's premise weakens.

4. **`TransactionID` ships in the final feature set.** Third instance of the trap
   (`Id` in house-prices, 0019; `Customers` in Rossmann). It is numeric, it is
   genuinely present at predict time so 0030 has no objection, and its note says
   plainly that it tracks `TransactionDT` at r = 0.998 and that every test value is
   larger than every training value. Nothing refuses it and nothing should (0029).

5. **The run lands below the private median of 0.91096.** Registering a number:
   between 0.88 and 0.91 private. The raw table already probes at 0.891, so this
   is also a prediction that the feature chain will be worth little here — and
   0073 has just measured that this harness's ruler overprices feature steps by
   about a factor of three on a high-cardinality dataset, which is what this one is.

6. **Sealed will be optimistic about private.** 0060 measured +0.042 on
   Restaurant, 0061 +0.031, and both are metrics where higher is worse. Here it
   inverts: the sealed AUC should come back **higher** than the private AUC. The
   field decays ~0.030 public -> private on this board (median 0.94127 -> 0.91096),
   and we have decayed *less* than the field on both previous datasets — this is
   the third test of that (0060/0061).

## What is deliberately NOT done

No column is dropped, no feature is pre-built, no categorical is cast, nothing is
sampled and the profile cap is not raised. Prediction 1 exists because the cap is
in the way; raising it before measuring what it costs would be the harness tuned
against the dataset it is being measured on, which is the one thing `CLAUDE.md`
forbids twice.

**One submission, and then this board is spent.** A finished competition returns a
private score on every late submission, so the seal here is self-imposed
discipline and nothing else — which is exactly how 0045 happened.
"""
import json
import os
import sys
import time
import traceback
from collections import Counter

import pandas as pd
from pydantic_ai import capture_run_messages
from pydantic_ai.exceptions import ModelHTTPError, UsageLimitExceeded
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models.openai import OpenAIChatModelSettings
from pydantic_ai.usage import UsageLimits

from tabular_agent.harness import journal
from tabular_agent.harness.agent import Deps, build_agent, gateway_model
from tabular_agent.harness.session import RunSession
from tabular_agent.harness.stop import StopPolicy
from tabular_agent.harness.usage import context_profile
from tabular_agent.taskspec import TaskSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_ieee import (  # noqa: E402
    ABOUT,
    COLUMN_NOTES,
    DECLARED_CATEGORICAL,
    FORECAST_HORIZON,
    METRIC,
    OUT,
    SPLIT_ON,
    TEST_OUT,
    leaderboard,
    leaderboard_rank,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(REPO, "_runs", "ieee_v1")

MODEL = "fireworks/deepseek-v4-flash-0731"
REQUEST_TIMEOUT_S = 15 * 60
MODEL_SETTINGS = {
    "extra_body": {"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
    "timeout": REQUEST_TIMEOUT_S,
}

MAX_TABLES = 25
CALLS_PER_TABLE = 12
VAL_FRACTION = 0.05
HOLDOUT_FRACTION = 0.05
FIT_TIME_BUDGET_S = 20 * 60
MAX_PROVIDER_FAILURES = 5

#: Measured before the run by putting the raw work table through the probe
#: unchanged — the number every feature step is implicitly claiming to beat.
#: Recorded here so the claim in registered prediction 5 cannot be adjusted
#: afterwards.
RAW_TABLE_PROBE_AUC = 0.89125

FIRST = """Build the best model you can for this dataset of payment transactions.
The target is isFraud. Work until you stop making progress, then finalize."""

RESUME = """You are continuing work on this dataset that is already in progress.
Earlier attempts did things you have no memory of.

Carry on towards the best model you can. Call finalize_run only when the run is
genuinely finished."""


def serving_columns() -> list[str]:
    """What exists when this model is asked for a prediction.

    Every column of the test file, `TransactionID` included. That is not an
    oversight: the identifier genuinely is present at predict time, so declaring it
    unavailable would be a lie told to make the agent's decision for it. 0030 is
    about presence, and the column is present; whether it is *useful* is registered
    prediction 4 and is the agent's to get right or wrong.
    """
    return list(pd.read_parquet(TEST_OUT).columns)


def make_session() -> RunSession:
    return RunSession(
        RUN_DIR,
        OUT,
        TaskSpec(
            target="isFraud", metric=METRIC, notes=ABOUT, column_notes=COLUMN_NOTES,
            forecast_horizon=FORECAST_HORIZON,
        ),
        stop_policy=StopPolicy(max_tables=MAX_TABLES),
        val_fraction=VAL_FRACTION,
        holdout_fraction=HOLDOUT_FRACTION,
        split_on=SPLIT_ON,
        serving_columns=serving_columns(),
        fit_time_budget_s=FIT_TIME_BUDGET_S,
    )


def tool_calls(raw) -> list[str]:
    return [
        p["tool_name"]
        for m in raw
        for p in m.get("parts", [])
        if p.get("part_kind") == "tool-call"
    ]


def drive(agent, prompt, session, call_limit):
    """One agent run; returns (transcript, output, ended). The standard driver."""
    with capture_run_messages() as messages:
        try:
            result = agent.run_sync(
                prompt,
                deps=Deps(session=session),
                usage_limits=UsageLimits(tool_calls_limit=call_limit),
                model_settings=OpenAIChatModelSettings(**MODEL_SETTINGS),
            )
            return json.loads(result.all_messages_json()), result.output, "agent"
        except UsageLimitExceeded:
            raw = json.loads(ModelMessagesTypeAdapter.dump_json(messages))
            return raw, f"[cut at {call_limit} tool calls]", "cap"
        except ModelHTTPError as e:
            raw = json.loads(ModelMessagesTypeAdapter.dump_json(messages))
            return raw, f"[provider error {e.status_code}]", "provider"
        except Exception as e:
            raw = json.loads(ModelMessagesTypeAdapter.dump_json(messages))
            if "timeout" in type(e).__name__.lower() or "timeout" in str(e).lower():
                print(f"  request timed out after {REQUEST_TIMEOUT_S / 60:.0f} min")
                return raw, f"[timeout: {type(e).__name__}]", "provider"
            traceback.print_exc()
            return raw, f"[{type(e).__name__}: {e}]", "error"


def _shipped_features(j: dict) -> list[str]:
    best = j.get("best") or {}
    row = next(
        (r for r in (j.get("leaderboard") or []) if r["id"] == best.get("experiment_id")),
        {},
    )
    return list(row.get("features") or [])


def report_the_declared_categoricals(run_dir: str) -> dict:
    """Registered 1, 2 and 3: which of the 49 did the run reach, and how?

    Split by how the column *arrives*, because that is the whole subject. A string
    is dropped by the fit and its absence is reported; a numeric identifier is kept
    and split as an ordered quantity, and nothing reports anything. 0074's claim is
    that the second fate is invisible, so the interesting cell of this table is the
    one counting encodings of numeric-dtype declared categoricals.
    """
    j = journal.load(run_dir)
    raw = pd.read_parquet(OUT, columns=DECLARED_CATEGORICAL)
    numeric = {c for c in DECLARED_CATEGORICAL
               if pd.api.types.is_numeric_dtype(raw[c])}
    strings = set(DECLARED_CATEGORICAL) - numeric
    identity = {c for c in DECLARED_CATEGORICAL
                if c.startswith("id_") or c.startswith("Device")}

    encoded: dict[str, list[str]] = {}
    for t in j.get("tables") or []:
        for c in t.get("columns") or t.get("encoded_columns") or []:
            encoded.setdefault(c, []).append(f"{t.get('id')}:{t.get('method')}")
    # Fall back to matching produced column names against the declared list, since
    # not every tool records its input columns under the same key.
    produced = {c for t in (j.get("tables") or []) for c in (t.get("new_columns") or [])}
    for c in DECLARED_CATEGORICAL:
        if c in encoded:
            continue
        hits = sorted(p for p in produced if p.startswith(c + "_"))
        if hits:
            encoded[c] = [f"produced: {', '.join(hits[:3])}"]

    shipped = set(_shipped_features(j))
    return {
        "declared": len(DECLARED_CATEGORICAL),
        "arrive_as_strings": len(strings),
        "arrive_as_numbers": len(numeric),
        "REG_2_string_categoricals_encoded": sorted(c for c in encoded if c in strings),
        "REG_3_numeric_categoricals_encoded": sorted(c for c in encoded if c in numeric),
        "REG_1_identity_columns_encoded": sorted(c for c in encoded if c in identity),
        "numeric_categoricals_shipped_unencoded": sorted(
            c for c in numeric if c in shipped and c not in encoded
        ),
    }


def report_the_identifier(run_dir: str) -> dict:
    """Registered 4: did `TransactionID` survive into the shipped model?"""
    j = journal.load(run_dir)
    feats = _shipped_features(j)
    return {
        "shipped_n_features": len(feats),
        "REG_4_TransactionID_shipped": "TransactionID" in feats,
        "TransactionDT_shipped": "TransactionDT" in feats,
    }


def report_the_seal(run_dir: str) -> None:
    j = journal.load(run_dir)
    record = j.get("holdout")
    if not record:
        print("\nNO SEALED SCORE — the run did not finalize")
        return
    split = j.get("split") or {}
    print(f"\n{'=' * 70}")
    print(f"THE SEALED SPLIT — {record['n_rows']} rows, scored once, after the "
          "winner was picked")
    print(f"  work {split.get('n_work')} / val {split.get('n_val')} / "
          f"sealed {split.get('n_holdout')}, ordered on {split.get('split_on')}")
    print(f"\n  {'model':<12s} {'sel':<5s} {'val':>10s} {'sealed':>10s} {'optimism':>10s}")
    for e in record["scores"]:
        if e.get("holdout_score") is None:
            print(f"  {e.get('model_type', '?'):<12s} "
                  f"not scored: {e.get('holdout_unavailable')}")
            continue
        print(f"  {e['model_type']:<12s} {'yes' if e['selected'] else '':<5s} "
              f"{e['val_score']:>10.5f} {e['holdout_score']:>10.5f} "
              f"{e.get('optimism', float('nan')):>+10.5f}")
    print("\n  NOTE: auc is higher-is-better, so `optimism` reads the opposite way")
    print("  round from every other run in this repo. Registered 6 predicts the")
    print("  sealed AUC comes back ABOVE the private one.")


def write_submission(run_dir: str) -> str | None:
    """Predict the competition's test block.

    Binary classification, so the bundle returns P(isFraud), which is what AUC is
    computed on and what the sample submission asks for. No clipping and no
    rounding: a probability is already in range, and rounding one would destroy the
    ranking the metric is entirely made of.
    """
    from tabular_agent.pipeline import load_bundle, predict_bundle

    bundle_dir = os.path.join(run_dir, "bundle")
    if not os.path.isdir(bundle_dir):
        print("\nno bundle on disk — the run never called finalize_run")
        return None

    test = pd.read_parquet(TEST_OUT)
    bundle = load_bundle(bundle_dir)
    try:
        preds = predict_bundle(bundle, test)
    except Exception as e:
        print(f"\n!! cannot predict the test set: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None

    path = os.path.join(run_dir, "submission.csv")
    pd.DataFrame({"TransactionID": test["TransactionID"], "isFraud": preds}).to_csv(
        path, index=False
    )
    return path


def main() -> int:
    for p in (OUT, TEST_OUT):
        if not os.path.exists(p):
            print(f"missing {p} — run scripts/prep_ieee.py first", file=sys.stderr)
            return 1
    os.makedirs(RUN_DIR, exist_ok=True)

    session = make_session()
    print(f"model {MODEL} | thinking "
          f"{MODEL_SETTINGS['extra_body']['reasoning_effort']}")
    print(f"metric {METRIC} (higher is better) | split_on {SPLIT_ON} "
          f"| val {VAL_FRACTION} | SEALED {HOLDOUT_FRACTION} | {MAX_TABLES} tables")
    print(f"forecast_horizon {FORECAST_HORIZON} seconds "
          f"({FORECAST_HORIZON / 86400:.0f} days)")
    rec = session.split_record()
    print(f"split: {rec['n_work']} work / {rec['n_val']} val / {rec['n_holdout']} sealed")
    print(f"raw work table through the probe, measured beforehand: "
          f"{RAW_TABLE_PROBE_AUC:.5f}")

    episodes = []
    t_start = time.time()
    failures = 0
    attempt = 0
    while attempt < 1 + MAX_PROVIDER_FAILURES:
        attempt += 1
        j = journal.load(RUN_DIR)
        if j["stop"]["finalized"]:
            break
        resuming = bool(j.get("tables"))
        agent = build_agent(gateway_model(MODEL))
        t0 = time.time()
        raw, output, ended = drive(
            agent, RESUME if resuming else FIRST, session,
            MAX_TABLES * CALLS_PER_TABLE,
        )
        calls = tool_calls(raw)
        episodes.append({
            "attempt": attempt, "tool_calls": calls, "ended_by": ended,
            "output": output, "resumed": resuming,
            "elapsed_min": round((time.time() - t0) / 60, 2),
            **{k: context_profile(raw).get(k)
               for k in ("peak_input_tokens", "n_requests")},
        })
        print(f"\nattempt {attempt}: {len(calls)} tool calls, ended by {ended}"
              f"{' (resumed)' if resuming else ''}")
        print(f"  {dict(Counter(calls))}")
        with open(os.path.join(RUN_DIR, f"transcript_{attempt}.json"), "w") as f:
            json.dump(raw, f)

        if ended not in ("provider", "error"):
            print(f"\n{output}")
            break
        failures += 1
        print(f"  !! {ended} ({failures}/{MAX_PROVIDER_FAILURES}); "
              "restarting on the same run dir")
        time.sleep(10)

    with open(os.path.join(RUN_DIR, "episodes.json"), "w") as f:
        json.dump(episodes, f, indent=2)

    j = journal.load(RUN_DIR)
    if j.get("tables") and not j["stop"]["finalized"]:
        from tabular_agent.harness import tools
        print("\nthe run ended without finalize_run — packaging its best model")
        tools.finalize_run(session)
        j = journal.load(RUN_DIR)

    print(f"\n{'=' * 70}")
    rows = j.get("leaderboard") or []
    tables = [t for t in (j.get("tables") or []) if t.get("probe_score") is not None]
    print(f"{len(j.get('tables') or [])} tables ({len(tables)} scored) and "
          f"{len(rows)} terminal fits in {(time.time() - t_start) / 60:.1f} min")
    wp = j.get("work_probe") or {}
    if wp.get("probe_score") is not None:
        print(f"work table, same ruler: {wp['probe_score']:.5f} "
              f"— what the features had to beat (predicted {RAW_TABLE_PROBE_AUC:.5f})")
    # Descending: auc is higher-is-better, and every other runner in this repo
    # sorts ascending because every other metric is an error.
    for t in sorted(tables, key=lambda r: -r["probe_score"]):
        print(f"  {t['id']:>4s}  {t.get('method') or '':<18s} {t['probe_score']:.5f}")

    print("\nterminal fits:")
    for r in rows:
        val = r.get("val_score")
        print(f"  {r['id']:>3s}  {r['model_type']:<12s} val "
              f"{'n/a' if val is None else f'{val:.5f}'}  "
              f"({(r.get('fit_seconds') or 0) / 60:.1f} min)")

    if j.get("best"):
        best = j["best"]
        print(f"\nbest {best['experiment_id']}: val {best['val_score']:.5f}")
        print(f"  private board: {leaderboard(private=True)}")
        print(f"  public  board: {leaderboard(private=False)}")

    report_the_seal(RUN_DIR)

    print(f"\n{'=' * 70}")
    print("REGISTERED 1, 2, 3 — the 49 declared categoricals")
    print(json.dumps(report_the_declared_categoricals(RUN_DIR), indent=2))
    print("\nREGISTERED 4 — the identifier")
    print(json.dumps(report_the_identifier(RUN_DIR), indent=2))

    path = write_submission(RUN_DIR)
    if path:
        sub = pd.read_csv(path)
        print(f"\nsubmission: {path}")
        print(f"  {len(sub)} rows, isFraud {sub.isFraud.min():.5f} "
              f"to {sub.isFraud.max():.5f}, mean {sub.isFraud.mean():.5f} "
              f"(the data's own rate is 0.0350)")
        print("\n  REGISTERED 5: private between 0.88 and 0.91, below the median "
              f"0.91096 — {leaderboard_rank(0.91096)}")
        print("  ONE submission, and then this board is spent (0045).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
