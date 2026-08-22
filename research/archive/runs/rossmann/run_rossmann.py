"""Rossmann store sales: the first run on a large, genuinely time-ordered dataset.

Run `scripts/prep_rossmann.py` first — this reads the parquet it writes.

Everything about the *exam* is fixed here and none of it is the agent's to change
(0029): the metric is `rmspe` because that is what the competition grades, the
split is ordered on `Date` because the task is forecasting, and `val_fraction` is
0.05 so the validation horizon (~45 days) matches Kaggle's 48 and the score can be
read against the public leaderboard in the data folder.

## What this run is actually measuring

**1. Does the agent drop `Customers`?** It correlates 0.895 with `Sales`, it is
absent from `test.csv`, and — this is the part that makes it a real test — it is
numeric, so it lands in the feature matrix **by default**. Doing nothing means
training on it. Removing it takes a deliberate `transform_features` call, and the
only reason to make one is the sentence in the data dictionary saying the column
does not exist at prediction time.

The prompt below does not mention it. If it were mentioned this would measure
whether the agent can follow an instruction, which we already know it often
cannot; what is unknown is whether documentation attached to a column changes what
the agent does with that column.

`summarise_customers` answers it, and `write_submission` makes the consequence
concrete. There are **three** outcomes, verified by hand before this script was
finished, and only the last is submittable:

    keeps Customers                        val 0.314  bundle CANNOT predict
    df.drop(columns=["Customers"])         val 0.372  bundle CANNOT predict
    df.drop(..., errors="ignore")          val 0.372  works

The middle one is the nasty one: the agent does the right thing, the score is
honest, and the bundle still fails — because per 0010 a transform is stored as
*code* and replayed on raw rows, which at predict time never had the column.
**Nothing in the harness can catch this**, since `_score_on_val` replays on
validation rows drawn from the training file, and those still have `Customers`.
The first prediction on genuinely unseen data is the first opportunity to find
out, which is exactly why this script tries it.

**2. Does any of this work at a million rows?** Every earlier run was 1460 rows.
A cross-validated lgbm here takes ~5s, so the cost is the model's latency rather
than ours, but nothing has been tried at this size.

**3. What does the cv-vs-val gap look like now that both forecast?** Under
forward-chaining folds (0029) `cv_score` should come out *worse* than `val_score`,
because early folds train on a fraction of the history — the reverse of every
previous run. A large gap the other way would mean something is wrong.

**4. (v5) Does the agent reach for the two tools it did not have?** `filter_rows`
and `fit_group_stats` were added because earlier runs demonstrated the gaps: 17%
of rows are ones `rmspe` does not score and could not be excluded from the fit,
and v4's first attempt died trying to build per-group aggregates of a column that
had no sanctioned path. Neither is mentioned in the prompt, for the same reason
`Customers` is not — what is being measured is whether the tool's own description
is enough. `summarise_new_tools` reports it.

## Episodes, or one conversation

0025 measured the crossover: episodes cost more per experiment below roughly 35
and become the only feasible option well above it. `MAX_EXPERIMENTS` decides, and
the default is a pilot — small enough that one conversation is both cheaper and
better informed, and large enough to answer the three questions above.
"""
import json
import os
import sys
import time
import traceback
from collections import Counter

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
from prep_rossmann import ABOUT, COLUMN_NOTES, METRIC, OUT, SPLIT_ON, TEST_OUT  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(REPO, "_runs", "rossmann_v6")

#: v6 changes the **route**, not the model. Every run before this one went through
#: `tensorx/deepseek-v4-flash-0731`, which serves this model with thinking mode
#: switched off — verified across five parameter shapes, and not something the
#: gateway's own `supports_reasoning: true` metadata will tell you (that flag
#: describes the model, not the route). `fireworks` serves the same weights with
#: thinking available.
#:
#: So every behavioural finding this project has (0024, 0025, 0027, 0032) was
#: measured on a model that was never allowed to think, and v6 is the controlled
#: repeat that says whether any of them survive. Everything else below —
#: experiments, prompts, budget, split, metric — is byte-identical to v5.
MODEL = "fireworks/deepseek-v4-flash-0731"

#: Thinking is off unless asked for, and asking takes both keys: the mode and the
#: budget. `high` rather than `max` because the point is to find out whether
#: deliberation changes behaviour at all, and `max` would confound "does it think"
#: with "how much".
#:
#: This costs real money in a way v5 did not: reasoning tokens are billed as
#: output, and v5's whole run spent 29k output tokens precisely because it never
#: thought.
MODEL_SETTINGS = {
    "extra_body": {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}
}

#: A pilot. Below 0025's ~35 crossover, so this runs as one conversation.
MAX_EXPERIMENTS = 25

#: 0025's rule, in the one place it currently has to live. Not in `harness/`
#: because *when* to use episodes is still an open decision, and burying it in the
#: library would settle it by accident.
EPISODE_CROSSOVER = 35
USE_EPISODES = MAX_EXPERIMENTS > EPISODE_CROSSOVER

TOOL_CALLS_PER_EPISODE = 16
MAX_EPISODES = 12

#: Backstop for the single-conversation path, sized *per experiment* rather than
#: absolutely. v1 spent 44 calls on 17 experiments; v2 spent 120 on 13 and was cut
#: off before it could finalize. Calls-per-experiment is a property of how the
#: agent happens to work on a dataset, so a fixed number is a budget for one run's
#: habits and a guillotine for another's.
CALLS_PER_EXPERIMENT = 12

#: ~45 days of validation against Kaggle's 48-day test window.
VAL_FRACTION = 0.05

#: Wall-clock ceiling on one cross-validated fit. A **safety net**, not a tuning
#: policy — the instruction to prefer features over hyper-parameters lives in the
#: prompt, and this catches what no instruction can.
#:
#: Sized from v6, where 13 of 15 experiments finished inside 4 minutes and one took
#: **113** with parameters identical to a 3.8-minute one. 20 minutes is therefore
#: five times the observed normal and a fifth of the observed pathology, which is
#: about as much daylight as a single threshold gets.
#:
#: Two caveats on that number. v6's timings are contaminated — other heavy work ran
#: on the same laptop during the long fit — so the pathology may be smaller than it
#: looks. And this is a property of *this machine*: on the GPU box the same fits
#: take seconds, and 20 minutes would never fire.
FIT_TIME_BUDGET_S = 20 * 60

#: From `rossmann-store-sales-publicleaderboard-*.csv`, 3300 teams.
LEADERBOARD = {"best": 0.08932, "top 10%": 0.10294, "median": 0.11677}

FIRST = """Build the best model you can for this dataset of daily store sales.
The target is Sales."""

RESUME = """You are continuing work on this dataset that is already in progress.
Earlier episodes did things you have no memory of.

Carry on towards the best model you can. As before: follow one idea far enough
to know whether it worked, then stop and summarise. Call finalize_run only when
the run is genuinely finished."""

EPISODE_FIRST = FIRST + """

Work in short stretches. When you have followed one idea far enough to know
whether it worked, stop and summarise what you did and what you would try next.
Do not call finalize_run unless you are genuinely finished with the whole run."""


def serving_columns() -> list[str]:
    """What will exist when this model is asked for a prediction.

    Read from the competition's own test file rather than written by hand, so it
    cannot drift from the thing it describes.
    """
    import pandas as pd

    return [c for c in pd.read_parquet(TEST_OUT).columns if c != "Id"]


def make_session() -> RunSession:
    return RunSession(
        RUN_DIR,
        OUT,
        TaskSpec(target="Sales", metric=METRIC, notes=ABOUT, column_notes=COLUMN_NOTES),
        stop_policy=StopPolicy(max_experiments=MAX_EXPERIMENTS),
        val_fraction=VAL_FRACTION,
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


#: Provider failures tolerated before giving up. The first attempt of this run
#: died on a 502 from the gateway after two experiments — nothing to do with the
#: harness, and it took the whole conversation with it.
MAX_PROVIDER_FAILURES = 5


def drive(agent, prompt, session, call_limit):
    """One agent run; returns (transcript, output, ended).

    `capture_run_messages` keeps the transcript even when the run does not return
    normally, which is the case most worth reading: a run ended by the harness or
    by the provider rather than by the agent.

    A transient provider error is caught here and reported as an ending like any
    other. It is *not* retried at this level — retrying inside one conversation
    would resend a context the provider just failed on. The caller starts a fresh
    agent instead, which is the same move an episode makes and works for the same
    reason: the state is the run directory, not the conversation (0003, 0024).
    """
    with capture_run_messages() as messages:
        try:
            result = agent.run_sync(
                prompt,
                deps=Deps(session=session),
                usage_limits=UsageLimits(tool_calls_limit=call_limit),
                # Passed here rather than baked into `build_agent`: which model
                # serves the run, and how hard it is asked to think, are the
                # driver's business (0013's division). The library stays
                # provider-agnostic.
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
            # Broad on purpose, and only at the driver. v3 died on an OSError from
            # a corrupt parquet while holding eight experiments and the best model
            # any run had produced — and lost all of it, because the exception went
            # straight past the finalize step. The artifacts were on disk the whole
            # time. Printed in full rather than swallowed: this catches so the run
            # can be salvaged, not so a bug can be ignored.
            traceback.print_exc()
            raw = json.loads(ModelMessagesTypeAdapter.dump_json(messages))
            return raw, f"[{type(e).__name__}: {e}]", "error"


def summarise_customers(run_dir: str) -> dict:
    """Did the agent train on the column that does not exist at predict time?

    The measurement this run exists for. `Customers` is numeric, so it is in the
    feature matrix unless the agent deliberately removed it — "did nothing" and
    "decided to keep it" look the same here, and both mean the winning model
    cannot be submitted.
    """
    j = journal.load(run_dir)
    rows = j.get("leaderboard") or []
    used = [r["id"] for r in rows if "Customers" in (r.get("features") or [])]
    best = j.get("best") or {}
    best_row = next((r for r in rows if r["id"] == best.get("experiment_id")), {})
    return {
        "experiments": len(rows),
        "experiments_using_customers": len(used),
        "best_used_customers": "Customers" in (best_row.get("features") or []),
        "tables_without_customers": [
            t["path"] for t in (j.get("tables") or [])
            if t.get("new_columns") and "Customers" not in t["new_columns"]
            and t.get("method") == "transform"
        ],
    }


def summarise_thinking(run_dir: str) -> dict:
    """Did the model actually think, and how much did it cost?

    Recorded because "we switched to a reasoning route" is a claim about
    configuration and this is the evidence. A run with zero thinking parts is
    v5 again under a different model name, and every comparison drawn from it
    would be wrong in the same direction.

    Output tokens are the price of the answer: on this provider the chain of
    thought is billed as output, so the ratio against v5's 29k is the cost of
    deliberation.
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
    out["v5_output_tokens_for_comparison"] = 29027
    return out


def summarise_new_tools(run_dir: str) -> dict:
    """Did the two tools added for this run get used, and did they pay?

    v4's first attempt died trying to build per-store aggregates it had no tool
    for, and 17% of this dataset's rows are ones the metric does not score. Both
    are now expressible. Whether the agent reaches for either is a behavioural
    result, so it is measured rather than assumed — and "built a table nobody
    trained on" is a distinct and interesting outcome from "never called it".
    """
    j = journal.load(run_dir)
    tables = j.get("tables") or []
    rows = j.get("leaderboard") or []
    trained_on = {r.get("source_table") for r in rows}

    def summarise(method: str) -> dict:
        built = [t for t in tables if t.get("method") == method]
        # Any experiment whose table has one of these in its history, not just as
        # its immediate source — the agent normally stacks another step on top.
        descended = [
            r["id"] for r in rows
            if any(t["path"] in journal.ancestry(j, r.get("source_table") or "")
                   for t in built)
        ]
        return {
            "tables_built": len(built),
            "trained_on_directly": sum(1 for t in built if t["path"] in trained_on),
            "experiments_downstream": len(descended),
            "detail": [
                {k: t.get(k) for k in ("id", "new_columns", "n_dropped", "group_by",
                                       "value_column", "stat") if t.get(k) is not None}
                for t in built
            ],
        }

    return {"filter_rows": summarise("filter"),
            "fit_group_stats": summarise("group_stats")}


def write_submission(run_dir: str) -> str | None:
    """Predict the real test set, or say precisely why the winner cannot.

    This is where a leak stops being a number and becomes a model that does not
    run. If the winning bundle needs `Customers`, the replay raises on a column
    that is not in `test.csv` and there is nothing to submit.

    The bundle is found on disk rather than taken from `finalize_run`'s return
    value, because the *agent* made that call and we never saw what it returned.
    """
    import pandas as pd

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
        # Two different failures name the same column, and they mean opposite
        # things about what the agent did. Distinguish them, because a wrong
        # diagnosis here is worse than none.
        if "Customers" in str(e):
            if "Customers" in (bundle.get("feature_names") or []):
                print("   The winning model was TRAINED on Customers, which does not")
                print("   exist at prediction time. The leak, made concrete.")
            else:
                print("   The model does not use Customers, but a stored transform")
                print("   REFERENCES it and re-runs at predict time on rows that")
                print("   never had the column (0010: the code is the artifact).")
                print("   `df.drop(columns=[...], errors='ignore')` survives the")
                print("   replay; a plain drop does not. Nothing in the harness can")
                print("   catch this — validation rows come from the training file")
                print("   and still have the column.")
        return None

    negative = int((preds < 0).sum())
    if negative:
        # Reported, not clipped. What the model outputs is the model's business;
        # silently flooring it here would hide a real defect in the fit.
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
    print(f"metric {METRIC} | split_on {SPLIT_ON} | val_fraction {VAL_FRACTION}")
    print(f"{MAX_EXPERIMENTS} experiments, "
          f"{'EPISODES' if USE_EPISODES else 'one conversation'} "
          f"(0025 crossover is ~{EPISODE_CROSSOVER})")

    episodes = []
    t_start = time.time()

    if USE_EPISODES:
        for i in range(1, MAX_EPISODES + 1):
            if journal.load(RUN_DIR)["stop"]["finalized"]:
                print(f"\n=== run finalized before episode {i} ===")
                break
            before = len(journal.load(RUN_DIR).get("leaderboard") or [])
            agent = build_agent(gateway_model(MODEL))
            t0 = time.time()
            raw, output, ended = drive(
                agent, EPISODE_FIRST if i == 1 else RESUME, session,
                TOOL_CALLS_PER_EPISODE,
            )
            calls = tool_calls(raw)
            after = len(journal.load(RUN_DIR).get("leaderboard") or [])
            episodes.append({
                "episode": i, "experiments_before": before, "experiments_after": after,
                "tool_calls": calls, "elapsed_min": round((time.time() - t0) / 60, 2),
                "ended_by": ended, "output": output,
                **{k: context_profile(raw).get(k)
                   for k in ("peak_input_tokens", "n_requests")},
            })
            print(f"\nEPISODE {i}: {after - before} new experiments, {len(calls)} calls, "
                  f"{episodes[-1]['elapsed_min']:.1f} min, ended by {ended}")
            print(f"  {dict(Counter(calls))}")
            with open(os.path.join(RUN_DIR, f"transcript_ep{i}.json"), "w") as f:
                json.dump(raw, f)
            # The loop guard: an episode that adds no experiment means the loop is
            # spinning, and spinning costs money.
            if after == before and not journal.load(RUN_DIR)["stop"]["finalized"]:
                print("  !! no experiments added — stopping the loop")
                break
    else:
        # One conversation, but restarted on a fresh agent if the provider drops
        # it. That restart *is* an episode — same run dir, new context — which is
        # why nothing extra is needed to make it work.
        failures = 0
        attempt = 0
        while attempt < 1 + MAX_PROVIDER_FAILURES:
            attempt += 1
            j = journal.load(RUN_DIR)
            resuming = bool(j.get("leaderboard"))
            if j["stop"]["finalized"]:
                break
            agent = build_agent(gateway_model(MODEL))
            t0 = time.time()
            raw, output, ended = drive(
                agent, RESUME if resuming else FIRST, session,
                MAX_EXPERIMENTS * CALLS_PER_EXPERIMENT,
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
            # `error` restarts for the same reason `provider` does, and v4 is why:
            # the agent tried to build per-store aggregates, the stability check
            # refused them three times, the tool's retry budget ran out, and
            # `UnexpectedModelBehavior` ended a 25-experiment run after 2.5
            # minutes with nothing on the leaderboard. One idea the harness cannot
            # express should cost an idea, not the run — and a fresh agent gets a
            # fresh retry budget plus a brief describing what already exists.
            print(f"  !! {ended} ({failures}/{MAX_PROVIDER_FAILURES}); "
                  "restarting on the same run dir")
            time.sleep(10)

    with open(os.path.join(RUN_DIR, "episodes.json"), "w") as f:
        json.dump(episodes, f, indent=2)

    # The agent normally calls `finalize_run` itself, and when it does not — it ran
    # out of calls, the provider gave up — the run has a leaderboard, a winner and
    # nothing to show for it. Packaging it here is a *driver* decision, not a
    # library one: 0012 gives the harness the stop, and this is the harness
    # noticing the conversation ended without one.
    j = journal.load(RUN_DIR)
    if j.get("leaderboard") and not j["stop"]["finalized"]:
        from tabular_agent.harness import tools
        print("\nthe run ended without finalize_run — packaging its best model")
        report = tools.finalize_run(session)
        print(f"  bundled {report['best_experiment_id']} "
              f"({report['metric']} {report['val_score']:.5f})")

    # ------------------------------------------------------------------ results
    print(f"\n{'=' * 70}")
    j = journal.load(RUN_DIR)
    rows = j.get("leaderboard") or []
    print(f"{len(rows)} experiments in {(time.time() - t_start) / 60:.1f} min")

    if j.get("best"):
        best = j["best"]
        row = next((r for r in rows if r["id"] == best["experiment_id"]), {})
        print(f"best {best['experiment_id']}: val {best['val_score']:.5f} "
              f"(cv {row.get('cv_score', float('nan')):.5f})")
        print("  Kaggle public LB: " + " | ".join(
            f"{k} {v}" for k, v in LEADERBOARD.items()))
        # cv should be WORSE than val here: forward-chaining folds train on less
        # history than the final fit. The other direction means something is off.
        if row.get("cv_score") and row["cv_score"] < best["val_score"]:
            print("  note: cv better than val — unexpected under forward chaining")

    # Where the wall clock actually went. v6 spent 113 of 132 minutes inside one
    # fit and nothing in the run said so — it had to be reconstructed from file
    # mtimes afterwards. Printed as a distribution rather than a total, because the
    # question is whether one experiment ate the run.
    costs = [(r["id"], r.get("fit_seconds")) for r in rows if r.get("fit_seconds")]
    if costs:
        worst = max(costs, key=lambda c: c[1])
        total = sum(c[1] for c in costs)
        print(f"\nfit cost: {total / 60:.1f} min over {len(costs)} experiments"
              f" | median {sorted(c[1] for c in costs)[len(costs) // 2] / 60:.1f} min"
              f" | worst {worst[0]} {worst[1] / 60:.1f} min"
              f" ({worst[1] / total:.0%} of all fitting)")

    print(f"\ncustomers: {json.dumps(summarise_customers(RUN_DIR), indent=2)}")
    print(f"\nnew tools: {json.dumps(summarise_new_tools(RUN_DIR), indent=2)}")
    print(f"\nthinking: {json.dumps(summarise_thinking(RUN_DIR), indent=2)}")

    # The consequence, not the diagnosis. A model that used `Customers` cannot
    # score the test set at all, and that is worth seeing rather than inferring.
    path = write_submission(RUN_DIR)
    if path:
        print(f"\nsubmission: {path}")
        print("  Score it on Kaggle to get the number our validation half is "
              "only an estimate of.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
