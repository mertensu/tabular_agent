"""Rossmann v7: the first run with a sealed third split, on ordered data.

Two things land at once here, and saying so up front is the point of this
docstring — v7 is **not** a clean A/B against v6.

1. **0035.** Rossmann has never run under the feature-only harness. v5 and v6
   searched over models *and* tables; this agent has no tool that picks a model,
   and the terminal phase enumerates four fits at the end.
2. **0023/0041.** `holdout_fraction=0.05`: a sealed slice no tool can reach,
   scored once at `finalize_run`.

Both change the number, so "v7 beat v6" would be uninterpretable and is not
claimed. What v7 is for is the *sealed* number, which no previous run has, and
which is worth having most on this dataset specifically — see below.

## Why Rossmann is the right first sealed run

Every optimism figure this project owns came off a **random** split: house-prices
+0.00206 / +0.00944 / +0.00458, seguro +0.01828 / +0.00838. All positive, all
measured against Kaggle.

Rossmann is the one dataset where our exam came out **pessimistic** — v5 reported
rmspe 0.13760 and Kaggle said 0.135, better by 0.0026. 0029's explanation is that
an ordered split plus forward-chaining folds asks the competition's question, so
there is no interpolation premium to hand back.

The sealed split is the instrument that can say whether that survives being
measured in-house. It is also the case 0041 warns about hardest: the slices are
`work | val | holdout` **in time order**, so val and holdout are different
periods, and the gap between them mixes selection optimism with drift. This run
cannot separate them, and the report says so.

## Registered before the run

1. **Mechanical.** A three-way ordered cut lands on honest `Date` boundaries at
   ~942 distinct dates and ~1M rows, all three slices non-empty, and
   `finalize_run` produces a sealed score for the shipped model and for the
   terminal candidates it beat. Nothing before this has cut a large ordered table
   three ways.

2. **Direction: optimism at or below zero**, i.e. the sealed score is not much
   worse than `val_score`. This is the prediction that follows from v5's
   pessimistic exam, and it is the one most likely to be wrong for a reason we
   would want to know: the sealed slice is a *later* period than validation, so
   drift pushes the gap positive regardless of what the search did. A clearly
   positive gap here is therefore evidence about the confound, not proof of
   selection.

3. **The selected model is the more optimistic one** (0039, held twice in 0040).
   The confound named in both records is that the selected model has always been
   lgbm and the unselected one elasticnet. Rossmann is where that can break: on
   this data catboost has won every previous run, so if a non-probe family wins
   here, the pair is finally "selected vs unselected" with the family held
   constant on the other side. **If lgbm wins again, the confound survives and
   this claim is untested rather than passed.**

4. **`Customers` is still the leak test.** Unchanged since v5 and not mentioned in
   the prompt: it correlates 0.895 with `Sales`, is absent from `test.csv`, and is
   numeric, so doing nothing means training on it. `serving_columns` is what
   refuses it, and `write_submission` is what makes the consequence concrete.

## What is deliberately unchanged from v6

Model, route, thinking budget, metric, `split_on`, `val_fraction`, the fit time
budget, and the data. `val_fraction` stays at 0.05 even though the sealed slice
now takes the last ~45 days that used to be validation — see `VAL_FRACTION`.
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
RUN_DIR = os.path.join(REPO, "_runs", "rossmann_v7")

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
        TaskSpec(target="Sales", metric=METRIC, notes=ABOUT, column_notes=COLUMN_NOTES),
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

    print(f"\ncustomers: {json.dumps(summarise_customers(RUN_DIR), indent=2)}")
    print(f"\nthinking: {json.dumps(summarise_thinking(RUN_DIR), indent=2)}")

    path = write_submission(RUN_DIR)
    if path:
        print(f"\nsubmission: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
