"""Rossmann v8: the first run with history, on the worst freeze date in the file.

One change from v7: the agent has `fit_history` (slice 2 of
`docs/time_aware_features.md`, decided by 0047/0048), and `TaskSpec` declares
`forecast_horizon=1.0` so lag offsets are legal for it to try. Model, route,
thinking budget, metric, `split_on`, both split fractions, the fit budget, the
prompts and the data are all v7's, so this *is* a clean A/B in the way v7 was not.

## What this run is actually for

Not "does history improve Rossmann". 0049 already measured that by hand, and the
answer was no: `Store` history of `Sales` cost **-0.13244** on the probe, because
every validation row reads one freeze date and this split's freeze date is
**2015-05-01, a public holiday with 98.21% of stores shut**. `val_fraction` and
`holdout_fraction` are unchanged from v7, so **this run inherits that same
boundary.** The worst case is live, deliberately.

The question is one level up: **does the harness's own feedback stop a run from
shipping a feature that hurts?** Every table is scored the moment it exists and
`probe_vs_source_table` reports whether it beat its parent (0035, 0036). That
machinery has never been tested against a tool that is *available, legal,
correctly implemented, and bad for this dataset*. Everything the harness has
refused before was refused structurally — a leak, an unservable column, a
batch-dependent transform. This is the first capability whose failure is a matter
of degree, and the probe is the only thing standing between the agent and it.

## Registered before the run

1. **Mechanical.** `fit_history` is callable on this data, produces tables with
   `_last`/`_age` pairs, and — with `forecast_horizon=1.0` — accepts offsets. The
   ordering column is `Date`, which the run is already split on.

2. **Will the agent reach for it at all?** Genuinely unknown, and worth recording
   as unknown. It is one tool among ten and the brief does not push it. A run that
   never calls it tells us the vocabulary grew without the agent noticing, which
   is 0027's finding in a new place and would be worth its own record.

3. **Direction: history tables score WORSE than their source**, by roughly the
   margins in 0049 — around -0.13 for `Store`-keyed history of `Sales`, and
   near-neutral to slightly positive for `Store`+`DayOfWeek`. If the numbers here
   differ much from 0049's, the hand-driven probe and the in-run probe disagree
   and that is the finding.

4. **The one that matters: the run does NOT ship a history feature.** The probe
   tells the agent, per table, that these tables are worse than their parents. If
   the winning bundle contains a `_last` column anyway, the feedback loop is
   decorative — the same shape of failure 0031 recorded about a guard that looked
   like it was doing something. **This is the prediction I expect to be right and
   would learn most from being wrong about.**

5. **`Customers` is still the leak test**, unchanged since v5 and still not
   mentioned in the prompt. Note history adds a *new* way to use it legitimately:
   `fit_history(by=["Store"], value="Customers")` stores it and never needs the
   column at predict time. 0049 measured that at -0.14111, so taking that route
   would be both clever and wrong, which is a combination worth being able to see.

## What is deliberately unchanged from v7

Model, route, thinking budget, metric, `split_on`, `val_fraction`,
`holdout_fraction`, the fit time budget, the prompts, and the data.
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
from tabular_agent.harness.usage import context_profile, format_profile
from tabular_agent.taskspec import TaskSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_rossmann import ABOUT, COLUMN_NOTES, METRIC, OUT, SPLIT_ON, TEST_OUT  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(REPO, "_runs", "rossmann_v8")

#: The reasoning route, as in v6, house v8/v9 and seguro v3/v4.
MODEL = "fireworks/deepseek-v4-flash-0731"

#: A ceiling on one request, and it is load-bearing rather than tidy. v7's first
#: attempt sat for **105 minutes on a single POST** to the gateway: socket
#: ESTABLISHED, 0% CPU, no bytes, six tables already on disk and no seventh
#: coming. The driver's `MAX_PROVIDER_FAILURES` could not help — it retries
#: provider *errors*, and a hang is not an error, so nothing ever fired.
#:
#: This is the same shape as 0033's finding one layer up: the run had a guard for
#: the failure it had seen (a 502) and none for the failure where the thing simply
#: never answers. `run_sync` is one blocking request, so without this the run has
#: no upper bound at all.
#:
#: 15 minutes because the reasoning route genuinely takes minutes on a large
#: payload — v6's slowest legitimate response was well inside it — and anything
#: past that is a stall rather than deliberation.
REQUEST_TIMEOUT_S = 15 * 60

MODEL_SETTINGS = {
    "extra_body": {"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
    "timeout": REQUEST_TIMEOUT_S,
}

#: The unit is the table since 0035 — the agent cannot spend a call on a model, so
#: this is a ceiling on ideas rather than on fits. Below 0025's ~35 crossover, so
#: one conversation.
MAX_TABLES = 25
CALLS_PER_TABLE = 12

#: Kept at v5/v6's value, and the consequence is worth stating because it is not
#: obvious. 0.05 was chosen so the validation horizon (~45 days) matched Kaggle's
#: 48-day test window and the score could be read against the public leaderboard.
#:
#: With a sealed slice the order is `work | val | holdout`, so it is now the
#: **sealed** rows that occupy the last ~45 days and stand where Kaggle's test set
#: stands; validation is the ~45 days before that. Which is the right way round —
#: the sealed number is the one asked to predict unseen data, so it should be the
#: most test-like rows in the file (0041). The cost is that `val_score` here is
#: not measured over the same horizon v5's and v6's were, so those numbers are
#: not directly comparable to this one. Nothing in this run's claims depends on
#: comparing them.
VAL_FRACTION = 0.05
HOLDOUT_FRACTION = 0.05

#: Wall-clock ceiling on one cross-validated fit (v6 lost 113 of its 132 minutes
#: to a single fit whose parameters matched a 3.8-minute one). A safety net, not a
#: tuning policy.
FIT_TIME_BUDGET_S = 20 * 60

#: Provider failures tolerated before giving up. A 502 once killed a Rossmann run
#: that had nothing to do with the harness.
MAX_PROVIDER_FAILURES = 5

#: Where the field sits, for reading the final number against something.
LEADERBOARD = {"best": 0.08932, "top 10%": 0.10294, "median": 0.11677}

FIRST = """Build the best model you can for this dataset of daily store sales.
The target is Sales. Work until you stop making progress, then finalize."""

RESUME = """You are continuing work on this dataset that is already in progress.
Earlier attempts did things you have no memory of.

Carry on towards the best model you can. Call finalize_run only when the run is
genuinely finished."""


def serving_columns() -> list[str]:
    """What will exist when this model is asked for a prediction.

    Read from the competition's own test file rather than written by hand, so it
    cannot drift from the thing it describes. This is what refuses `Customers`.
    """
    return [c for c in pd.read_parquet(TEST_OUT).columns if c != "Id"]


def make_session() -> RunSession:
    return RunSession(
        RUN_DIR,
        OUT,
        TaskSpec(target="Sales", metric=METRIC, notes=ABOUT,
                 column_notes=COLUMN_NOTES,
                 # Kaggle's test block opens the day after training ends, so the
                 # earliest row asked for is one day out. That is the floor across
                 # the block, which is what this field means (0047); rows deeper in
                 # are 2-48 days out and their `_age` columns say so.
                 forecast_horizon=1.0),
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
    """One agent run; returns (transcript, output, ended).

    `capture_run_messages` keeps the transcript even when the run does not return
    normally — the case most worth reading. A provider error is reported as an
    ending like any other and is not retried here: the caller starts a fresh
    agent, which works because the state is the run directory and not the
    conversation (0003, 0024).
    """
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
            # A timeout is reported as a provider failure, not as an unknown
            # error, so it is retried on the same run dir rather than printing a
            # traceback that suggests a bug in the harness. It is neither — it is
            # the gateway not answering, which is the case `REQUEST_TIMEOUT_S`
            # exists to turn from an indefinite hang into a retry.
            if "timeout" in type(e).__name__.lower() or "timeout" in str(e).lower():
                print(f"  request timed out after {REQUEST_TIMEOUT_S / 60:.0f} min")
                return raw, f"[timeout: {type(e).__name__}]", "provider"
            traceback.print_exc()
            return raw, f"[{type(e).__name__}: {e}]", "error"


def summarise_customers(run_dir: str) -> dict:
    """Did the run keep the column that does not exist at prediction time?"""
    j = journal.load(run_dir)
    best = j.get("best") or {}
    row = next(
        (r for r in (j.get("leaderboard") or []) if r["id"] == best.get("experiment_id")),
        {},
    )
    feats = row.get("features") or []
    return {
        "best_n_features": len(feats),
        "kept_customers": "Customers" in feats,
        "tables_mentioning_customers": [
            t.get("id") for t in (j.get("tables") or []) if "Customers" in json.dumps(t)
        ],
    }


def summarise_history(run_dir: str) -> dict:
    """Did the run reach for history, and did it ship one? (registered 2 and 4)

    Two different questions and they fail in opposite directions. A run that never
    calls the tool says the vocabulary grew without the agent noticing. A run that
    calls it, is told by the probe that the table got worse, and ships it anyway
    says the feedback loop is decorative.
    """
    j = journal.load(run_dir)
    tables = [t for t in (j.get("tables") or []) if t.get("method") == "history"]
    best = j.get("best") or {}
    row = next(
        (r for r in (j.get("leaderboard") or []) if r["id"] == best.get("experiment_id")),
        {},
    )
    feats = row.get("features") or []
    shipped = [f for f in feats if f.endswith("_last") or "_last_age" in f or "_lag" in f]
    return {
        "history_tables": len(tables),
        "keys_tried": [t.get("group_by") for t in tables],
        "values_tried": [t.get("value_column") for t in tables],
        "offsets_tried": [t.get("offsets") for t in tables],
        # Negative means the table was worse than the one it was built on, which
        # is what 0049 predicts and what the agent was shown at the time.
        "probe_vs_source": [t.get("probe_vs_source_table") for t in tables],
        "history_columns_in_winner": shipped,
        "shipped_a_history_feature": bool(shipped),
    }


def summarise_thinking(run_dir: str) -> dict:
    """Evidence that the run actually thought, and what it cost."""
    out = {"thinking_parts": 0, "text_parts": 0, "input_tokens": 0, "output_tokens": 0}
    for name in sorted(os.listdir(run_dir)):
        if not (name.startswith("transcript_") and name.endswith(".json")):
            continue
        with open(os.path.join(run_dir, name)) as f:
            raw = json.load(f)
        for m in raw:
            for p in m.get("parts", []):
                if p.get("part_kind") == "thinking":
                    out["thinking_parts"] += 1
                elif p.get("part_kind") == "text":
                    out["text_parts"] += 1
            u = m.get("usage") or {}
            out["input_tokens"] += u.get("input_tokens") or 0
            out["output_tokens"] += u.get("output_tokens") or 0
    return out


def report_the_seal(run_dir: str) -> None:
    """Print the thing this run exists to produce (0023, 0041).

    Every number above it in the console output was measured on rows the search
    selected against. These were not.
    """
    j = journal.load(run_dir)
    record = j.get("holdout")
    if not record:
        print("\nNO SEALED SCORE — the run did not finalize, or was resumed from a "
              "run dir cut without a holdout")
        return

    split = j.get("split") or {}
    print(f"\n{'=' * 70}")
    print(f"THE SEALED SPLIT — {record['n_rows']} rows, scored once, "
          f"after the winner was picked")
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
    print("  On an ordered split the sealed rows are a LATER period than the")
    print("  validation rows, so this gap mixes selection with drift and this")
    print("  run cannot separate them (0041). Registered prediction: <= 0.")


def write_submission(run_dir: str) -> str | None:
    """Predict the real test set, or say precisely why the winner cannot.

    This is where a leak stops being a number and becomes a model that does not
    run. If the winning bundle needs `Customers`, the replay raises on a column
    that is not in `test.csv` and there is nothing to submit.
    """
    from tabular_agent.pipeline import load_bundle, predict_bundle

    bundle_dir = os.path.join(run_dir, "bundle")
    if not os.path.isdir(bundle_dir):
        print("\nno bundle on disk — the run never called finalize_run")
        return None

    test = pd.read_parquet(TEST_OUT)
    bundle = load_bundle(bundle_dir)
    try:
        preds = predict_bundle(bundle, test.drop(columns=["Id"]))
    except Exception as e:
        print(f"\n!! cannot predict the test set: {type(e).__name__}: {e}")
        # Two different failures name the same column and mean opposite things
        # about what the agent did. A wrong diagnosis here is worse than none.
        if "Customers" in str(e):
            if "Customers" in (bundle.get("feature_names") or []):
                print("   The winning model was TRAINED on Customers, which does not")
                print("   exist at prediction time. The leak, made concrete.")
            else:
                print("   The model does not use Customers, but a stored transform")
                print("   REFERENCES it and re-runs at predict time on rows that")
                print("   never had the column (0010: the code is the artifact).")
        return None

    negative = int((preds < 0).sum())
    if negative:
        # Reported, not clipped. Silently flooring it would hide a real defect.
        print(f"   note: {negative} predictions below zero, left as they are")

    path = os.path.join(run_dir, "submission.csv")
    pd.DataFrame({"Id": test["Id"], "Sales": preds}).to_csv(path, index=False)
    return path


def main() -> int:
    for p in (OUT, TEST_OUT):
        if not os.path.exists(p):
            print(f"missing {p} — run scripts/prep_rossmann.py first", file=sys.stderr)
            return 1
    os.makedirs(RUN_DIR, exist_ok=True)

    session = make_session()
    print(f"model {MODEL} | thinking "
          f"{MODEL_SETTINGS['extra_body']['reasoning_effort']}")
    print(f"metric {METRIC} | split_on {SPLIT_ON} | val {VAL_FRACTION} "
          f"| SEALED {HOLDOUT_FRACTION} | {MAX_TABLES} tables, one conversation")
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
        # The expected case on *this* dataset, and not an error: the work table
        # carries `Customers`, so the ruler refuses to score it at all (0030).
        # There is no "data as given" reference number here — the data as given
        # cannot be served.
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
        print(f"\nbest {best['experiment_id']}: val {best['val_score']:.5f}")
        print("  v5 (pre-0035) val 0.13760 -> Kaggle 0.135, PESSIMISTIC by 0.0026")
        print("  v6 (pre-0035) val 0.12741, never scored")
        print("  " + " | ".join(f"{k} {v}" for k, v in LEADERBOARD.items()))

    report_the_seal(RUN_DIR)

    profile = None
    for name in sorted(os.listdir(RUN_DIR)):
        if name.startswith("transcript_") and name.endswith(".json"):
            with open(os.path.join(RUN_DIR, name)) as f:
                profile = context_profile(json.load(f), experiments=len(rows))
    if profile:
        with open(os.path.join(RUN_DIR, "context_profile.json"), "w") as f:
            json.dump(profile, f, indent=2)
        print()
        print(format_profile(profile, "context"))

    print(f"\nhistory: {json.dumps(summarise_history(RUN_DIR), indent=2)}")
    hist = summarise_history(RUN_DIR)
    # Registered 4 said the run would not ship a history feature, on the reasoning
    # that the probe would report these tables as worse. Whether that reasoning
    # held is a question about the *deltas*, not about what shipped — so it is
    # read from them rather than assumed. The first version of this block asserted
    # the premise and printed a conclusion that the run's own numbers contradict.
    deltas = [d for d in hist["probe_vs_source"] if d is not None]
    helped = [d for d in deltas if d > 0]
    if hist["history_tables"] == 0:
        print("  -> the tool was never called: the vocabulary grew and the agent")
        print("     did not notice (registered 2, and its own finding).")
    elif hist["shipped_a_history_feature"] and len(helped) > len(deltas) / 2:
        print(f"  -> shipped, and the probe agreed: {len(helped)} of {len(deltas)} history")
        print("     tables beat their parents. Registered 4 is WRONG, and wrong")
        print("     because the feature helped — not because the loop failed.")
    elif hist["shipped_a_history_feature"]:
        print(f"  -> a history column IS in the winner, though only {len(helped)} of")
        print(f"     {len(deltas)} history tables beat their parents. Worth reading the")
        print("     per-table deltas before concluding anything about the loop.")
    else:
        print("  -> tried and not shipped: the probe steered the run off a feature")
        print("     that is legal, correct and bad here. Registered 4 holds.")

    print(f"\ncustomers: {json.dumps(summarise_customers(RUN_DIR), indent=2)}")
    print(f"\nthinking: {json.dumps(summarise_thinking(RUN_DIR), indent=2)}")

    path = write_submission(RUN_DIR)
    if path:
        print(f"\nsubmission: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
