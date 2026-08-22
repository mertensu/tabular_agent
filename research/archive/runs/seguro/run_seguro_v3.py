"""Porto Seguro, run 3: the feature-only harness on a classification problem.

Run 2 is the control (`_runs/seguro_v2`): same file, same prompt, same model and
route, same metric. **The only difference is the harness** — 0035's feature-only
search, the fixed probe, the enumerated terminal fit, one stop counter over
tables. This is the second pre-registered A/B after house-prices run 9, and the
first on a task that is not regression.

    run 2   11 experiments   val gini 0.28460 (e11) -> Kaggle 0.27724

Run 2's shipped model was e8 at Kaggle **0.26522**; e11 scored 0.27724 and lost
the selection by 0.00020 of val, a twentieth of a standard error (0034). Both
numbers are quoted because they answer different questions: 0.26522 is what the
old harness *delivered*, and 0.27724 is what it *had* and threw away.

## Why this dataset and not another house run

House-prices run 9 produced two findings that need a second dataset before they
mean anything, and this one is as different as our archive gets: binary target at
3.6% positive, a pure ranking metric, 595,212 rows against 1,460.

1. **Optimism belongs to the table, not the model** (0038). Four models on two
   tables clustered by table, threefold apart. If that is real it should appear
   here too, and this run can show it cheaply: the terminal fit produces three
   models on one table, and their optimism should be nearly identical.
2. **Less optimism under the new harness.** Run 9 went +0.0113 -> +0.00458 on
   rmsle. Run 2's optimism was +0.0183 over 11 experiments — the largest we have
   measured — so there is more room here than anywhere else.

## What this run tests that nothing has

**0033's elasticnet fix has never carried a real run.** Run 2 never called
elasticnet, and run 1 died inside one: 63 minutes at 100% of one core, a measured
~510 hours for the full search. The fix bounds the penalty search to 20,000
stratified rows and refits on all of them. The terminal phase now fits elasticnet
**unconditionally**, so this run either confirms the fix at 476k rows or fails
loudly in a place that cannot take the run with it — `_terminal_fits` skips a
family that raises.

**The probe on a ranking metric at scale.** One lgbm fit per table on 476k rows,
plus a bootstrap `probe_se` over 119k validation rows. Both paths are unit-tested
and neither has run on a table this size.

## The metric, and why it is `auc` rather than `gini`

The competition grades the Normalized Gini Coefficient. `gini.c` in the data
folder is the official implementation, and for binary labels it computes

    gini = 2 * auc - 1

a strictly increasing function of AUC. So every comparison the harness makes
comes out identically under either name, and a `gini` entry in
`SUPPORTED_METRICS` would buy no decision that cannot already be made. It is
reported in gini below, because that is the unit the leaderboard in the data
folder is denominated in.

## Registered before anything is submitted

1. **Kaggle at or above run 2's shipped 0.26522.** Beating 0.27724 — the model
   run 2 had and discarded — would be a stronger result and is not claimed.
2. **Optimism below run 2's +0.0183.**
3. **The three terminal models' optimism within ~0.002 of each other**, which is
   what 0038's table-not-model hypothesis predicts and what would falsify it.

As always, the honest number costs a Kaggle submission and 0023 remains unbuilt.
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

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO, "seguro_driver_prediction")
#: v2. v1 is kept intact and is the control: same script, same prompt, same
#: everything, run against the library *before* 0033. It reached 5 experiments and
#: was killed at 63 minutes inside a single elasticnet fit — so the one thing v2
#: can say that v1 could not is what a linear model does on this data, and whether
#: having that reading changes where the run goes next.
RUN_DIR = os.path.join(REPO, "_runs", "seguro_v3")
DATA = os.path.join(RAW, "train.csv")
TEST = os.path.join(RAW, "test.csv")
LEADERBOARD_CSV = os.path.join(
    RAW, "porto-seguro-safe-driver-prediction-publicleaderboard-2026-08-12T08:54:24.csv"
)

TARGET = "target"

#: See the module docstring: `auc` and the competition's Normalized Gini rank
#: identically, and `gini = 2*auc - 1` converts between them.
METRIC = "auc"

#: The reasoning route, as in Rossmann v6 and house v8. Same weights as the
#: `tensorx` route every earlier run used, served with thinking available.
MODEL = "fireworks/deepseek-v4-flash-0731"
MODEL_SETTINGS = {
    "extra_body": {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}
}

#: A pilot, below 0025's ~35 crossover, so this runs as one conversation. The
#: unit is the table now (0035): the agent cannot spend a call on a model, so
#: this is a ceiling on ideas rather than on fits.
MAX_TABLES = 25
CALLS_PER_TABLE = 12

#: 20% of 595,212 rows is ~119k held out, which makes `val_se` on a ranking metric
#: small enough that the stop rule has a usable yardstick. Left at the default
#: rather than shrunk: unlike Rossmann there is no forecast horizon to match, the
#: rows are exchangeable, and there is no shortage of them.
VAL_FRACTION = 0.2

#: Wall-clock ceiling on one cross-validated fit. A safety net, not a tuning
#: policy. Measured beforehand on the full table: lgbm 11s, catboost 21s across
#: five folds — so 10 minutes is ~30x the observed normal, and anything that
#: reaches it is pathological rather than merely ambitious.
FIT_TIME_BUDGET_S = 10 * 60

#: Provider failures tolerated before giving up. A 502 killed a Rossmann run that
#: had nothing to do with the harness.
MAX_PROVIDER_FAILURES = 5

ABOUT = """Auto insurance policy holders, one row per policy holder. The target
is whether that policy holder filed a claim: 1 if they did, 0 if they did not.
About 3.6% of rows are claims, so the classes are heavily imbalanced.

The column names carry the data owner's own grouping, and it is the only
documentation this dataset has -- the columns are otherwise anonymised, and no
description of what any individual one measures exists:

  * the tag after `ps_` groups related features: `ind` (individual), `reg`
    (region), `car` (car), `calc` (calculated).
  * a `_bin` suffix means the feature is binary.
  * a `_cat` suffix means the feature is *categorical*: the integers are labels,
    not quantities, and their order carries no meaning.
  * a column with neither suffix is continuous or ordinal.

**A value of -1 means the feature was missing for that observation.** It is a
sentinel, not a measurement, and it appears in 13 columns -- in two of them for
most of the rows.

Scoring uses the Normalized Gini Coefficient, which is a rescaling of AUC
(gini = 2*auc - 1) and depends only on the *order* the predictions put the rows
in. The magnitude of a predicted probability is never used."""

FIRST = """Build the best model you can for this dataset of insurance policy
holders. The target is target. Work until you stop making progress, then
finalize."""

RESUME = """You are continuing work on this dataset that is already in progress.
Earlier attempts did things you have no memory of.

Carry on towards the best model you can. Call finalize_run only when the run is
genuinely finished."""


def gini(auc: float) -> float:
    """The competition's unit, from ours. See the module docstring."""
    return 2 * auc - 1


def serving_columns() -> list[str]:
    """What will exist when this model is asked for a prediction.

    Read from the competition's own test file rather than written by hand, so it
    cannot drift. On this dataset it is every training column but the target —
    there is no `Customers` here — and it is set anyway, because a guard that is
    only switched on where it is known to be needed is not a guard.
    """
    return list(pd.read_csv(TEST, nrows=1).columns)


def make_session() -> RunSession:
    return RunSession(
        RUN_DIR,
        DATA,
        TaskSpec(target=TARGET, metric=METRIC, notes=ABOUT),
        stop_policy=StopPolicy(max_tables=MAX_TABLES),
        val_fraction=VAL_FRACTION,
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
            # Broad, and only at the driver: a Rossmann run once died holding
            # eight experiments and the best model any run had produced, with the
            # artifacts on disk the whole time. Printed in full rather than
            # swallowed — this catches so a run can be salvaged, not so a bug can
            # be ignored.
            traceback.print_exc()
            raw = json.loads(ModelMessagesTypeAdapter.dump_json(messages))
            return raw, f"[{type(e).__name__}: {e}]", "error"


#: The 20 columns widely reported to be noise. Here, and not in `ABOUT`, because
#: this is our outside knowledge and measuring whether a run finds it requires not
#: telling it. See measurement 4.
NOISE_PREFIX = "ps_calc_"

SENTINEL_COLUMNS = [
    "ps_car_03_cat", "ps_car_05_cat", "ps_reg_03", "ps_car_14", "ps_car_07_cat",
    "ps_ind_05_cat", "ps_car_09_cat", "ps_ind_02_cat", "ps_car_01_cat",
    "ps_ind_04_cat", "ps_car_02_cat", "ps_car_11", "ps_car_12",
]


def summarise_columns(run_dir: str) -> dict:
    """What the winning model was actually built on.

    Three of this run's five questions are answerable from one place — the feature
    list of the best experiment — so they are answered together rather than
    inferred separately from the transcript, which records intentions rather than
    outcomes.
    """
    j = journal.load(run_dir)
    rows = j.get("leaderboard") or []
    best = j.get("best") or {}
    best_row = next((r for r in rows if r["id"] == best.get("experiment_id")), {})
    feats = best_row.get("features") or []
    tables = j.get("tables") or []
    return {
        "best_n_features": len(feats),
        "kept_id": "id" in feats,
        "kept_calc_columns": sum(1 for c in feats if c.startswith(NOISE_PREFIX)),
        "kept_raw_cat_columns": sum(1 for c in feats if c.endswith("_cat")),
        "encodings_built": [
            {k: t.get(k) for k in ("id", "method", "columns", "new_columns")
             if t.get(k) is not None}
            for t in tables if t.get("method") in
            ("ordinal", "onehot", "frequency", "target", "encoding")
        ],
        "table_methods": dict(Counter(t.get("method") for t in tables)),
    }


def summarise_sentinel(run_dir: str) -> dict:
    """Did the documented `-1` get treated as missing anywhere?

    There is no single tool for "sentinel to NaN", so the trace is diffuse: it can
    appear as a `transform_features` body mentioning -1, or as an encoding that
    gives it its own level. Both count, and the raw evidence is printed rather
    than reduced to a yes/no, because the interesting outcome is *which* of the
    three available treatments a run chose.
    """
    j = journal.load(run_dir)
    mentions = [
        {"id": t.get("id"), "method": t.get("method"),
         "columns": t.get("columns"), "code": (t.get("code") or "")[:400]}
        for t in (j.get("tables") or [])
        if "-1" in json.dumps(t) or any(
            c in SENTINEL_COLUMNS for c in (t.get("columns") or [])
        )
    ]
    return {"tables_touching_the_sentinel": len(mentions), "detail": mentions}


def summarise_thinking(run_dir: str) -> dict:
    """Evidence that the run actually thought, and what it cost.

    A run with zero thinking parts is a non-reasoning run under a different model
    name, and every comparison drawn from it would be wrong the same way.
    """
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


def kaggle_rank(score_gini: float) -> str:
    if not os.path.exists(LEADERBOARD_CSV):
        return ""
    s = pd.read_csv(LEADERBOARD_CSV)["Score"].sort_values(ascending=False)
    s = s.reset_index(drop=True)
    rank = int((s > score_gini).sum()) + 1
    return f"rank ~{rank}/{len(s)} (top {rank / len(s):.0%})"


def write_submission(run_dir: str) -> str | None:
    """Predict the real test set so Kaggle can supply the honest number.

    892,816 rows, and the submission is a probability per `id`. Only the *order*
    is scored, so nothing here clips or calibrates — but predictions outside [0,1]
    are reported, because from a classifier they would mean something is wrong
    even though the metric would not notice.
    """
    from tabular_agent.pipeline import load_bundle, predict_bundle

    bundle_dir = os.path.join(run_dir, "bundle")
    if not os.path.isdir(bundle_dir):
        print("\nno bundle on disk — the run never finalized")
        return None

    test = pd.read_csv(TEST)
    try:
        preds = predict_bundle(load_bundle(bundle_dir), test)
    except Exception as e:
        print(f"\n!! cannot predict the test set: {type(e).__name__}: {e}")
        return None

    outside = int(((preds < 0) | (preds > 1)).sum())
    if outside:
        print(f"   !! {outside} predictions outside [0,1] from a classifier")
    path = os.path.join(run_dir, "submission.csv")
    pd.DataFrame({"id": test["id"], "target": preds}).to_csv(path, index=False)
    return path


def main() -> int:
    for p in (DATA, TEST):
        if not os.path.exists(p):
            print(f"missing {p} — download the competition data first", file=sys.stderr)
            return 1
    os.makedirs(RUN_DIR, exist_ok=True)

    session = make_session()
    print(f"model {MODEL} | thinking "
          f"{MODEL_SETTINGS['extra_body']['reasoning_effort']}")
    print(f"metric {METRIC} (gini = 2*auc - 1) | val_fraction {VAL_FRACTION} "
          f"| {MAX_TABLES} tables, one conversation")
    print("data handed over untouched: no prep, `id` included, `-1` not converted")

    episodes = []
    t_start = time.time()
    failures = 0
    attempt = 0
    while attempt < 1 + MAX_PROVIDER_FAILURES:
        attempt += 1
        j = journal.load(RUN_DIR)
        if j["stop"]["finalized"]:
            break
        resuming = bool(j.get("leaderboard"))
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

    # The agent normally calls `finalize_run` itself; when it does not, the run
    # has a leaderboard, a winner and nothing to show for it. Packaging here is a
    # driver decision (0012 gives the harness the stop, this is the harness
    # noticing the conversation ended without one).
    j = journal.load(RUN_DIR)
    if j.get("leaderboard") and not j["stop"]["finalized"]:
        from tabular_agent.harness import tools
        print("\nthe run ended without finalize_run — packaging its best model")
        tools.finalize_run(session)
        j = journal.load(RUN_DIR)

    print(f"\n{'=' * 70}")
    rows = j.get("leaderboard") or []
    tables = [t for t in (j.get("tables") or []) if t.get("probe_score") is not None]
    print(f"{len(j.get('tables') or [])} tables ({len(tables)} scored) and "
          f"{len(rows)} terminal fits in {(time.time() - t_start) / 60:.1f} min")
    if j.get("work_probe"):
        wp = j["work_probe"]["probe_score"]
        print(f"work table, same ruler: auc {wp:.5f} -> gini {gini(wp):.5f} "
              f"— what the features had to beat")
    for t in sorted(tables, key=lambda r: -r["probe_score"]):
        print(f"  {t['id']:>4s}  {t.get('method') or '':<12s} auc {t['probe_score']:.5f} "
              f"-> gini {gini(t['probe_score']):.5f}")
    print("\nterminal fits (0038 predicts their optimism is nearly identical):")
    for r in rows:
        print(f"  {r['id']:>3s}  {r['model_type']:<12s} val auc {r['val_score']:.5f} "
              f"-> gini {gini(r['val_score']):.5f}  ({r.get('fit_seconds', 0) / 60:.1f} min)")

    if j.get("best"):
        best, base = j["best"], j["baseline"]
        print(f"best {best['experiment_id']}: auc {best['val_score']:.5f} "
              f"-> gini {gini(best['val_score']):.5f}")
        print("  run 2 for comparison: shipped gini 0.26522 on Kaggle; the model it "
              "discarded scored 0.27724 (optimism +0.0183)")
        print(f"baseline:  auc {base['val_score']:.5f} "
              f"-> gini {gini(base['val_score']):.5f}")

    costs = [r["fit_seconds"] for r in rows if r.get("fit_seconds")]
    if costs:
        print(f"\nfit cost: {sum(costs) / 60:.1f} min over {len(costs)} terminal fits "
              f"| worst {max(costs) / 60:.1f} min")

    # Profiled per attempt, since a restarted run has more than one transcript and
    # the interesting number (peak input tokens) is a property of one conversation
    # rather than of the run. The last attempt is the one that reached the end.
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

    print(f"\ncolumns: {json.dumps(summarise_columns(RUN_DIR), indent=2)}")
    print(f"\nsentinel: {json.dumps(summarise_sentinel(RUN_DIR), indent=2)}")
    print(f"\nthinking: {json.dumps(summarise_thinking(RUN_DIR), indent=2)}")

    path = write_submission(RUN_DIR)
    if path:
        print(f"\nsubmission: {path}")
        if j.get("best"):
            g = gini(j["best"]["val_score"])
            print(f"  our val says gini {g:.5f} — {kaggle_rank(g)} if Kaggle agrees,")
            print("  which it will not. Score it to get the optimism (0023).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
