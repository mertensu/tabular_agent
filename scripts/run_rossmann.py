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
RUN_DIR = os.path.join(REPO, "_runs", "rossmann_v4")
MODEL = "tensorx/deepseek-v4-flash-0731"

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

    print(f"\ncustomers: {json.dumps(summarise_customers(RUN_DIR), indent=2)}")

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
