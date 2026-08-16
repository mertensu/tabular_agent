"""Recruit Restaurant v2: the same run as v1, with the stream declared.

**One change from v1, deliberately.** Model, thinking budget, both split
fractions, the fit budget, the stop policy, the prompts and the serving schema are
v1's verbatim. What is added is the run's second inputs — `air_reserve` and
`hpg_reserve`, uncollapsed — and therefore the agent's ability to reach them with
`fit_stream_aggregate` (0062). Everything else is held still so the two runs are
comparable and the difference has one candidate cause.

## What this run is for

v1 established the trap exists and costs. The collapsed `air_reserve_visitors`
counts bookings made the same morning — 30% of training reservations had zero
days' notice — while no row in the forecast window can contain one. Measured by
hand on this data before the run, on the rows that have any reservation at all:

    collapsed mean 13.668   against   lead-39 mean 0.676

So at the far edge of the forecast window the honest column is 5% of the naive
one. v1 shipped six reservation features and scored **private 0.54398, rank
~1369/2150**.

This run asks the next question, which is **not** "does the lead help the score".
That is the leaderboard-shaped question CLAUDE.md rules out and 0062 refused
explicitly. It is:

    given the capability, does the harness's own ruler point the agent at the
    honest column or at the dishonest one?

Because the probe scores against a validation half drawn from the *same uncut
stream*, there is every reason to think it rewards the collapsed column and
penalises the restricted one. If so, supplying the capability is necessary and not
sufficient, and the finding belongs to 0029/0041 — the exam — rather than to the
tool.

## Registered before the run

1. **The agent will build at least one stream aggregate.** `inspect_data` now
   reports both streams with `notice_zero_share`, and the brief names the tool. If
   it does not, the surfacing failed and that is the finding.

2. **It will choose a lead at or near the declared horizon (1), not near the far
   edge (39).** The horizon is what the refusal names and the tool reports
   `block_span` beside it without recommending. Measured beforehand: a lead of 1
   answers 8.3% of rows, a lead of 39 answers 0.4%.

3. **The honest column will probe WORSE than the collapsed one.** The sharp one.
   Validation shares the uncut stream, so the ruler should prefer the column that
   peeks. If this holds, the run has demonstrated that the instrument contradicts
   the capability.

4. **The agent will keep the collapsed columns regardless.** They are numeric,
   present at serve time, and described. v1 shipped six of them. Nothing refuses
   them and nothing should (0029).

5. **Kaggle: still below the private median (0.53223).** v1 was +0.0118 above it.
   Registering the direction so the result is a reading either way, and stating
   plainly that a *worse* score than v1 would not by itself refute anything —
   the honest column is allowed to be worth less than the peeking one.

## What is deliberately NOT done

No feature is pre-built and the collapsed columns stay in. Handing over only the
honest version would answer the question this dataset was kept for (0029), and
choosing the lead for the agent would be the harness doing the modelling.
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
    FORECAST_HORIZON,
    METRIC,
    OUT,
    SPLIT_ON,
    STREAMS,
    TEST_OUT,
    leaderboard,
    leaderboard_rank,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(REPO, "_runs", "restaurant_v2")

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

FIRST = """Build the best model you can for this dataset of daily restaurant
visitor counts. The target is visitors. Work until you stop making progress, then
finalize."""

RESUME = """You are continuing work on this dataset that is already in progress.
Earlier attempts did things you have no memory of.

Carry on towards the best model you can. Call finalize_run only when the run is
genuinely finished."""


def serving_columns() -> list[str]:
    """What exists when this model is asked for a prediction — v1's, unchanged.

    Note what it still does not refuse: every collapsed reservation column is
    here, because they genuinely exist at predict time. The trap is not a missing
    column, it is a column *made differently* on the two sides, which no
    column-level declaration can express (0030 is about presence, not provenance).
    A stream aggregate is the first thing in this harness that can.
    """
    return [c for c in pd.read_parquet(TEST_OUT).columns if c != "id"]


def make_session() -> RunSession:
    return RunSession(
        RUN_DIR,
        OUT,
        TaskSpec(
            target="visitors", metric=METRIC, notes=ABOUT, column_notes=COLUMN_NOTES,
            forecast_horizon=FORECAST_HORIZON,
        ),
        stop_policy=StopPolicy(max_tables=MAX_TABLES),
        val_fraction=VAL_FRACTION,
        holdout_fraction=HOLDOUT_FRACTION,
        split_on=SPLIT_ON,
        serving_columns=serving_columns(),
        fit_time_budget_s=FIT_TIME_BUDGET_S,
        # The one change from v1.
        streams=STREAMS,
    )


def tool_calls(raw) -> list[str]:
    return [
        p["tool_name"]
        for m in raw
        for p in m.get("parts", [])
        if p.get("part_kind") == "tool-call"
    ]


def drive(agent, prompt, session, call_limit):
    """One agent run; returns (transcript, output, ended). v1's driver verbatim."""
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


def summarise_streams(run_dir: str) -> dict:
    """Registered 1, 2 and 4: did it reach the stream, at what lead, and did it
    keep the collapsed columns beside it?"""
    j = journal.load(run_dir)
    stream_tables = [
        t for t in (j.get("tables") or []) if t.get("method") == "stream_aggregate"
    ]
    best = j.get("best") or {}
    row = next(
        (r for r in (j.get("leaderboard") or []) if r["id"] == best.get("experiment_id")),
        {},
    )
    feats = row.get("features") or []
    return {
        "stream_tables_built": len(stream_tables),
        "leads_asked_for": sorted({x for t in stream_tables for x in (t.get("leads") or [])}),
        "streams_used": sorted({t.get("stream") for t in stream_tables}),
        # The two kinds of reservation column in the shipped model, side by side.
        # `_lead` is the honest one; the rest are the collapse.
        "honest_features_shipped": [f for f in feats if "_lead" in f],
        "collapsed_features_shipped": [
            f for f in feats if "reserve" in f and "_lead" not in f
        ],
        "best_n_features": len(feats),
    }


def compare_honest_against_collapsed(run_dir: str) -> dict:
    """Registered 3: which way did the ruler point?

    The comparison the run exists for. Every stream table's gain over its parent,
    beside the interval 0059 puts around that difference. A negative gain here is
    the probe saying the honest column is worse than what it replaced — which is
    the expected and interesting outcome, not a bug.
    """
    j = journal.load(run_dir)
    rows = []
    for t in j.get("tables") or []:
        if t.get("method") != "stream_aggregate":
            continue
        interval = t.get("probe_vs_source_interval") or {}
        rows.append({
            "id": t.get("id"),
            "stream": t.get("stream"),
            "leads": t.get("leads"),
            "probe_score": t.get("probe_score"),
            "gain_over_parent": t.get("probe_vs_source_table"),
            "block_sd": interval.get("block_sd"),
            "sign_stable": interval.get("sign_stable"),
        })
    negative = [r for r in rows if (r["gain_over_parent"] or 0) < 0]
    return {
        "n_stream_tables": len(rows),
        "n_scored_worse_than_parent": len(negative),
        "rows": rows,
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
    print("  0060: a sealed score is a LOWER BOUND on the error at the horizon you")
    print("  are graded at, not an estimate of it — v1 read +0.042 sealed->private.")


def write_submission(run_dir: str) -> str | None:
    """Predict the competition's forecast window, handing back the streams.

    **The same cut files go in.** That is not a shortcut: the competition ships
    reservations only up to the moment the forecast is issued, so a forecast row
    genuinely has nothing booked later. Handing these in is what serving actually
    looks like here — and it is why the honest column stays honest on Kaggle while
    the collapsed one silently changes meaning.
    """
    from tabular_agent.pipeline import load_bundle, predict_bundle

    bundle_dir = os.path.join(run_dir, "bundle")
    if not os.path.isdir(bundle_dir):
        print("\nno bundle on disk — the run never called finalize_run")
        return None

    test = pd.read_parquet(TEST_OUT)
    bundle = load_bundle(bundle_dir)
    needed = bundle["pipeline"].required_streams()
    if needed:
        print(f"   bundle requires streams {needed}; handing back the cut files")
    streams = {
        name: pd.read_parquet(spec["path"]) for name, spec in STREAMS.items()
    }
    try:
        preds = predict_bundle(bundle, test.drop(columns=["id"]), streams=streams)
    except Exception as e:
        print(f"\n!! cannot predict the test set: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None

    negative = int((preds < 0).sum())
    if negative:
        print(f"   note: {negative} predictions below zero, reported not hidden; "
              "the CSV clips at 0 because rmsle is undefined there")

    path = os.path.join(run_dir, "submission.csv")
    pd.DataFrame({"id": test["id"], "visitors": preds.clip(min=0)}).to_csv(
        path, index=False
    )
    return path


def main() -> int:
    for p in (OUT, TEST_OUT, *[s["path"] for s in STREAMS.values()]):
        if not os.path.exists(p):
            print(f"missing {p} — run scripts/prep_restaurant.py first", file=sys.stderr)
            return 1
    os.makedirs(RUN_DIR, exist_ok=True)

    session = make_session()
    print(f"model {MODEL} | thinking "
          f"{MODEL_SETTINGS['extra_body']['reasoning_effort']}")
    print(f"metric {METRIC} | split_on {SPLIT_ON} | val {VAL_FRACTION} "
          f"| SEALED {HOLDOUT_FRACTION} | {MAX_TABLES} tables")
    print(f"streams: {', '.join(session.stream_names())} "
          f"| forecast_horizon {FORECAST_HORIZON}")
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
    for t in sorted(tables, key=lambda r: r["probe_score"]):
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
        # Both boards, labelled. A val score is not a board score at all, so this
        # is a placement of a number against a distribution and nothing more —
        # 0061 is the record of quoting one board's median beside the other's
        # score.
        print(f"  private board: {leaderboard(private=True)}")
        print(f"  public  board: {leaderboard(private=False)}")

    report_the_seal(RUN_DIR)

    print(f"\n{'=' * 70}")
    print("REGISTERED 1, 2, 4 — did it reach the stream, at what lead, "
          "and did it keep the collapse?")
    print(json.dumps(summarise_streams(RUN_DIR), indent=2))
    print("\nREGISTERED 3 — which way did the ruler point?")
    print(json.dumps(compare_honest_against_collapsed(RUN_DIR), indent=2))

    path = write_submission(RUN_DIR)
    if path:
        sub = pd.read_csv(path)
        print(f"\nsubmission: {path}")
        print(f"  {len(sub)} rows, visitors {sub.visitors.min():.2f} "
              f"to {sub.visitors.max():.2f}, mean {sub.visitors.mean():.2f}")
        print("\n  v1 for reference: private 0.54398 "
              f"({leaderboard_rank(0.54398)}), public 0.51301 "
              f"({leaderboard_rank(0.51301, private=False)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
