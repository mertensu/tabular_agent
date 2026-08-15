"""Recruit Restaurant v1: the first agent run on the second forecasting dataset.

Everything about the harness here is Rossmann v8's, unchanged — model, route,
thinking budget, both split fractions, the fit budget, the stop policy, the
prompts. What changes is the data, and that is the point: every conclusion this
project has drawn about ordered splits, sealed holdouts and the probe rests on
one dataset, and this is the second reading.

## What this run is for

Three things, in descending order of how much they would teach us.

**1. The lead-time trap, live.** `air_reserve_*` and `hpg_reserve_*` summarise a
reservation stream that is **cut at the moment the forecast is issued**. 30% of
training reservations were booked with zero days' notice; **no row in the forecast
window can contain one**, and the floor grows to 39 days at the far edge. So the
column means something different in training than at serving time.

No guard in this harness sees it. No fold is crossed, no encoder is misfitted, the
column is genuinely present at predict time so `serving_columns` cannot refuse it,
and **validation actively confirms it** because the validation half shares the same
uncut stream. This is 0057's item 1 — the capability we do not have — and this run
is the natural experiment for it.

**2. Is the probe's ruler any good on a second dataset?** 0054 found that at 200
trees the probe scores Rossmann's features *backwards*. Restaurant is a twentieth
of the size, and 0054's own capacity table says it gains nothing from a bigger
budget once `air_store_id` is expressible as an integer — which the agent must do
for itself here, because `_feature_frame` drops strings. Whether the ruler ranks
sensibly at this scale is unmeasured.

**3. The first run to carry the difference interval (0059).** Every
`probe_vs_source_table` now arrives with the spread of that same difference beside
it, and on an ordered split that is the across-block version. This run is the first
time an agent has ever seen one.

## Registered before the run

1. **The agent will use the reservation columns, and validation will reward it.**
   They are numeric, present, described, and correlated with the target. Nothing in
   the harness discourages it and `column_notes` states the caveat in prose the
   agent may or may not act on.

2. **Kaggle will punish exactly that**, and by more at the far edge of the window
   than the near edge. This is the prediction the submission tests. If val and
   Kaggle agree instead, the trap is smaller than 0054 claimed and that is worth
   knowing before item 1 is built on its premise.

3. **`sign_stable` will be False for at least one reservation-derived table.** The
   skew grows with distance into the forecast window, so the block furthest out
   should disagree with the nearest. This is the first real test of whether 0059's
   block mode catches something *inside* a run — on Rossmann it did not, and the
   window there was 44 days against this one's ~25.

4. **The ragged spine will bite something.** A store has no row on days it did not
   report, so "the last 364 rows" is not "the last 364 days" (0054). Any window or
   offset feature the agent builds is measured in rows, and the mismatch is silent.

5. **Direction, honestly: I do not expect to beat the median.** Rossmann sits below
   it after sixteen runs. Recording the expectation so the result is a reading
   either way.

## What is deliberately NOT done

No feature is pre-built, no column is dropped for the agent, and the reservation
columns stay in. Removing them would be the harness doing the agent's work (0029)
and would throw away the one thing this dataset was chosen for.
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
from prep_restaurant import (  # noqa: E402
    ABOUT,
    COLUMN_NOTES,
    METRIC,
    OUT,
    SPLIT_ON,
    TEST_OUT,
    leaderboard_rank,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(REPO, "_runs", "restaurant_v1")

MODEL = "fireworks/deepseek-v4-flash-0731"
REQUEST_TIMEOUT_S = 15 * 60
MODEL_SETTINGS = {
    "extra_body": {"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
    "timeout": REQUEST_TIMEOUT_S,
}

MAX_TABLES = 25
CALLS_PER_TABLE = 12

#: Rossmann's fractions, kept so the two datasets are read the same way. Here 5%
#: of 252,108 ordered rows is ~25 days against the competition's 39-day forecast
#: window — closer than Rossmann's match, and the sealed slice again occupies the
#: most test-like rows in the file (0041).
VAL_FRACTION = 0.05
HOLDOUT_FRACTION = 0.05

#: A twentieth of Rossmann's rows, so this is a generous ceiling rather than a
#: real constraint. Kept identical anyway: changing two things at once between the
#: only two forecasting runs we have would waste the comparison.
FIT_TIME_BUDGET_S = 20 * 60

MAX_PROVIDER_FAILURES = 5

#: The public board, for reading the final number against something. `rmsle`,
#: lower is better, read off the downloaded CSV rather than typed from memory —
#: the first version of this constant was wrong by 0.046 on the median, which is
#: twice the entire distance from median to first place.
#:
#: **The field is extremely compressed**, and that is the single most important
#: fact for reading any score here: 2,150 teams span 0.46535 to 0.48921 between
#: first and median, and median to top 10% is **0.0112**. Differences this project
#: routinely calls "noise" on Rossmann are the whole leaderboard here.
LEADERBOARD = {"best": 0.46535, "top 10%": 0.47800, "median": 0.48921}

FIRST = """Build the best model you can for this dataset of daily restaurant
visitor counts. The target is visitors. Work until you stop making progress, then
finalize."""

RESUME = """You are continuing work on this dataset that is already in progress.
Earlier attempts did things you have no memory of.

Carry on towards the best model you can. Call finalize_run only when the run is
genuinely finished."""


def serving_columns() -> list[str]:
    """What exists when this model is asked for a prediction.

    Read from the competition's own test file so it cannot drift. Note what this
    does *not* refuse: every reservation column is here, because they genuinely
    exist at predict time. The trap in this dataset is not a missing column, it is
    a column that is *made differently* on the two sides — which no column-level
    declaration can express (0030 is about presence, not provenance).
    """
    return [c for c in pd.read_parquet(TEST_OUT).columns if c != "id"]


def make_session() -> RunSession:
    return RunSession(
        RUN_DIR,
        OUT,
        TaskSpec(
            target="visitors", metric=METRIC, notes=ABOUT, column_notes=COLUMN_NOTES,
            # The forecast window opens the day after training ends, so the
            # nearest row asked for is one day out — the floor across the block,
            # which is what this field means (0047).
            forecast_horizon=1.0,
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
    """One agent run; returns (transcript, output, ended). v8's driver verbatim."""
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


def summarise_reservations(run_dir: str) -> dict:
    """Registered prediction 1: did the run build on the cut stream?

    The columns are legal, described and present at serve time, so nothing refuses
    them. What this reports is whether they reached the shipped model, which is
    what the Kaggle score then tests.
    """
    j = journal.load(run_dir)
    best = j.get("best") or {}
    row = next(
        (r for r in (j.get("leaderboard") or []) if r["id"] == best.get("experiment_id")),
        {},
    )
    feats = row.get("features") or []
    reserve = [f for f in feats if "reserve" in f]
    return {
        "best_n_features": len(feats),
        "reservation_features_in_winner": reserve,
        "shipped_a_reservation_feature": bool(reserve),
        "tables_touching_reservations": [
            t.get("id") for t in (j.get("tables") or []) if "reserve" in json.dumps(t)
        ],
    }


def summarise_intervals(run_dir: str) -> dict:
    """Registered prediction 3: did any comparison come back unstable across blocks?

    0059's block mode is new and has never been seen by an agent. On Rossmann it
    reported `sign_stable` True for a feature that lost 0.03 externally, so the
    honest expectation is that it is weak here too — but this is the first run
    where it is even measurable.
    """
    j = journal.load(run_dir)
    out = []
    for t in j.get("tables") or []:
        rec = t.get("probe_vs_source_interval") or {}
        if not rec:
            continue
        out.append({
            "id": t.get("id"),
            "method": t.get("method"),
            "gain": t.get("probe_vs_source_table"),
            "block_sd": rec.get("block_sd"),
            "sign_stable": rec.get("sign_stable"),
            "blocks_for": f"{rec.get('blocks_favouring_a')}/{rec.get('n_blocks')}",
        })
    return {
        "n_with_interval": len(out),
        "n_unstable": sum(1 for r in out if r["sign_stable"] is False),
        "rows": out,
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
          f"winner was picked")
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
    print("\n  positive optimism = the validation half flattered that model.")
    print("  Sealed rows are a LATER period than validation, so this mixes")
    print("  selection with drift and cannot separate them (0041).")


def write_submission(run_dir: str) -> str | None:
    """Predict the competition's forecast window, or say why the winner cannot."""
    from tabular_agent.pipeline import load_bundle, predict_bundle

    bundle_dir = os.path.join(run_dir, "bundle")
    if not os.path.isdir(bundle_dir):
        print("\nno bundle on disk — the run never called finalize_run")
        return None

    test = pd.read_parquet(TEST_OUT)
    bundle = load_bundle(bundle_dir)
    try:
        preds = predict_bundle(bundle, test.drop(columns=["id"]))
    except Exception as e:
        print(f"\n!! cannot predict the test set: {type(e).__name__}: {e}")
        return None

    negative = int((preds < 0).sum())
    if negative:
        # Reported, not clipped — flooring in a script hides a defect in the fit.
        # `rmsle` is undefined below -1, so the submission itself clips at 0 and
        # says so; the count above is the honest reading of the model.
        print(f"   note: {negative} predictions below zero, reported not hidden; "
              "the CSV clips at 0 because rmsle is undefined there")

    path = os.path.join(run_dir, "submission.csv")
    pd.DataFrame({"id": test["id"], "visitors": preds.clip(min=0)}).to_csv(
        path, index=False
    )
    return path


def main() -> int:
    for p in (OUT, TEST_OUT):
        if not os.path.exists(p):
            print(f"missing {p} — run scripts/prep_restaurant.py first", file=sys.stderr)
            return 1
    os.makedirs(RUN_DIR, exist_ok=True)

    session = make_session()
    print(f"model {MODEL} | thinking "
          f"{MODEL_SETTINGS['extra_body']['reasoning_effort']}")
    print(f"metric {METRIC} | split_on {SPLIT_ON} | val {VAL_FRACTION} "
          f"| SEALED {HOLDOUT_FRACTION} | {MAX_TABLES} tables")
    rec = session.split_record()
    print(f"split: {rec['n_work']} work / {rec['n_val']} val / {rec['n_holdout']} sealed")

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
              "— what the features had to beat")
    elif wp:
        print(f"work table, same ruler: UNSCORED — {wp.get('probe_unavailable')}")
    for t in sorted(tables, key=lambda r: r["probe_score"]):
        print(f"  {t['id']:>4s}  {t.get('method') or '':<14s} {t['probe_score']:.5f}")

    print("\nterminal fits:")
    for r in rows:
        val = r.get("val_score")
        print(f"  {r['id']:>3s}  {r['model_type']:<12s} val "
              f"{'n/a' if val is None else f'{val:.5f}'}  "
              f"({(r.get('fit_seconds') or 0) / 60:.1f} min)")

    if j.get("best"):
        best = j["best"]
        print(f"\nbest {best['experiment_id']}: val {best['val_score']:.5f}  "
              f"{leaderboard_rank(best['val_score'])}")
        print("  " + " | ".join(f"{k} {v}" for k, v in LEADERBOARD.items()))

    report_the_seal(RUN_DIR)

    print(f"\n{'=' * 70}")
    print("REGISTERED 1 — did the run build on the cut reservation stream?")
    print(json.dumps(summarise_reservations(RUN_DIR), indent=2))
    print("\nREGISTERED 3 — did any comparison come back unstable across blocks?")
    print(json.dumps(summarise_intervals(RUN_DIR), indent=2))

    path = write_submission(RUN_DIR)
    if path:
        sub = pd.read_csv(path)
        print(f"\nsubmission: {path}")
        print(f"  {len(sub)} rows, visitors {sub.visitors.min():.2f} "
              f"to {sub.visitors.max():.2f}, mean {sub.visitors.mean():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
