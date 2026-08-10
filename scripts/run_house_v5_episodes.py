"""house_prices v5: episodes with a *structural* cap, and whether a resumed agent orients itself.

Not library code on purpose. The episode loop is a parked design (see the
2026-08-06 handoff entry) and this is the experiment that decides whether it is
worth building — putting it in `harness/` first would be deciding the answer.

Two questions, and the second is the one that matters:

1. Does the context curve break? One `run_sync` grows linearly (~900 tok/turn,
   measured on v1-v3). A fresh agent per episode should reset it to near zero
   each time, turning the line into a sawtooth with flat peaks.

2. **Does a resumed agent find its own history?** The prompt below deliberately
   does NOT mention `explore_journal`. A prompt that named it would answer a
   question nobody asked. What we want to know is whether an agent told only
   "you are resuming, you have no memory" reaches for the tool that can tell it
   where it is — because if it does, the "derived brief" half of the parked
   design may be unnecessary.

v4 answered neither. Asked to stop after one idea, the agent ran the entire
25-experiment budget in a single episode and finalized, so there was never a
second episode. That is the same under-compliance-with-advice pattern this
project keeps hitting, and it means the parked design's claim — "the episode
stop is cheap and reversible, so the agent can own it" — proves only that the
agent *may* own it, not that it will.

So the cap is structural here: `UsageLimits(tool_calls_limit=...)`, with the
resulting exception treated as the episode boundary. That is safe for exactly
the reason the architecture was built this way — an episode cut mid-thought
loses a conversation, and `run_dir` does not care. Budgets are deliberately
small: what this experiment needs is *many episodes*, not many experiments, and
a tight cap makes mid-idea cuts more likely, which is the seam cost the parked
note flagged and v4 was biased against ever measuring.
"""
import json, os, sys, time
from collections import Counter

import logfire
from pydantic_ai import capture_run_messages
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import UsageLimits
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from pydantic_ai.messages import ModelMessagesTypeAdapter

from tabular_agent.harness import journal
from tabular_agent.harness.agent import Deps, build_agent, gateway_model
from tabular_agent.harness.session import RunSession
from tabular_agent.harness.usage import context_profile, format_profile
from tabular_agent.harness.stop import StopPolicy
from tabular_agent.taskspec import TaskSpec

REPO = "/Users/ulfmertens/Documents/tabular_agent"
RUN_DIR = os.path.join(REPO, "_runs", "house_prices_v5")
DATA = os.path.join(REPO, "house_prices", "train_no_id.csv")
MODEL = "tensorx/deepseek-v4-flash-0731"
SPANS = os.path.join(RUN_DIR, "spans.jsonl")

#: Bounded so a loop that stops making progress cannot bill all night.
MAX_EPISODES = 8

#: The episode boundary. Small on purpose — see the module docstring.
TOOL_CALLS_PER_EPISODE = 6

#: Lowered from the default 25 so the *run* does not end before enough episodes
#: have happened to learn anything from them.
MAX_EXPERIMENTS = 12

FIRST = """Build the best model you can for this dataset of house sales.
The target is SalePrice.

Work in short stretches. When you have followed one idea far enough to know
whether it worked, stop and summarise what you did and what you would try next.
Do not call finalize_run unless you are genuinely finished with the whole run."""

# No mention of explore_journal, and no counts, scores or table names. Whether
# the agent orients itself, and how, is the measurement.
RESUME = """You are continuing work on this dataset that is already in progress.
Earlier episodes did things you have no memory of.

Carry on towards the best model you can. As before: follow one idea far enough
to know whether it worked, then stop and summarise. Call finalize_run only when
the run is genuinely finished."""


class FileExporter(SpanExporter):
    def export(self, spans):
        with open(SPANS, "a") as f:
            for s in spans:
                f.write(json.dumps({
                    "name": s.name,
                    "duration_ms": round((s.end_time - s.start_time) / 1e6, 1),
                }) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


def n_experiments(run_dir):
    return len(journal.load(run_dir).get("leaderboard") or [])


def finalized(run_dir):
    return bool(journal.load(run_dir)["stop"]["finalized"])


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    open(SPANS, "w").close()
    logfire.configure(send_to_logfire=False, service_name="tabular_agent",
                      console=False,
                      additional_span_processors=[SimpleSpanProcessor(FileExporter())])
    logfire.instrument_pydantic_ai()

    session = RunSession(
        RUN_DIR, DATA, TaskSpec(target="SalePrice"),
        stop_policy=StopPolicy(max_experiments=MAX_EXPERIMENTS),
    )
    episodes = []
    t_start = time.time()

    for i in range(1, MAX_EPISODES + 1):
        if finalized(RUN_DIR):
            print(f"\n=== run finalized before episode {i} ===")
            break

        before = n_experiments(RUN_DIR)
        # A *fresh agent every time*: new context, same run_dir. This is the
        # whole mechanism — `RunSession` holds only pointers, so the state the
        # next episode inherits is the directory, not the conversation.
        agent = build_agent(gateway_model(MODEL))
        prompt = FIRST if i == 1 else RESUME

        t0 = time.time()
        # `capture_run_messages` keeps the transcript even when the cap fires,
        # which is the case we most want to read: an episode ended by the
        # harness rather than by the agent.
        hit_cap = False
        with capture_run_messages() as messages:
            try:
                result = agent.run_sync(
                    prompt,
                    deps=Deps(session=session),
                    usage_limits=UsageLimits(tool_calls_limit=TOOL_CALLS_PER_EPISODE),
                )
                output = result.output
                raw = json.loads(result.all_messages_json())
            except UsageLimitExceeded:
                hit_cap = True
                output = f"[episode cut at {TOOL_CALLS_PER_EPISODE} tool calls]"
                raw = json.loads(ModelMessagesTypeAdapter.dump_json(messages))
        elapsed = time.time() - t0

        calls = [
            p["tool_name"]
            for m in raw
            for p in m.get("parts", [])
            if p.get("part_kind") == "tool-call"
        ]
        profile = context_profile(raw)
        after = n_experiments(RUN_DIR)

        episodes.append({
            "episode": i,
            "experiments_before": before,
            "experiments_after": after,
            "tool_calls": calls,
            "first_five_calls": calls[:5],
            "called_explore_journal": "explore_journal" in calls,
            "peak_input_tokens": profile.get("peak_input_tokens"),
            "n_requests": profile.get("n_requests"),
            "elapsed_min": round(elapsed / 60, 2),
            "ended_by": "cap" if hit_cap else "agent",
            "output": output,
        })

        print(f"\n{'=' * 70}")
        print(f"EPISODE {i}: {after - before} new experiments ({before} -> {after}), "
              f"{len(calls)} calls, peak {profile.get('peak_input_tokens'):,} tokens, "
              f"{elapsed / 60:.1f} min, ended by {'CAP' if hit_cap else 'agent'}")
        print(f"  opened with: {' -> '.join(calls[:5])}")
        print(f"  explore_journal: {'YES' if 'explore_journal' in calls else 'no'}")
        print(f"  {dict(Counter(calls))}")

        with open(os.path.join(RUN_DIR, "episodes.json"), "w") as f:
            json.dump(episodes, f, indent=2)
        with open(os.path.join(RUN_DIR, f"transcript_ep{i}.json"), "w") as f:
            json.dump(raw, f)

        # The loop guard from the parked design. An episode that adds no
        # experiment means the loop is spinning, and spinning costs money.
        if after == before and not finalized(RUN_DIR):
            print("  !! no experiments added — stopping the loop")
            break

    print(f"\n{'=' * 70}")
    print(f"{len(episodes)} episodes, {n_experiments(RUN_DIR)} experiments, "
          f"{(time.time() - t_start) / 60:.1f} min total")
    peaks = [e["peak_input_tokens"] for e in episodes]
    print(f"peak context per episode: {peaks}")
    print(f"vs one long run (v3): a single climb to 43,691")
    j = journal.load(RUN_DIR)
    if j.get("best"):
        print(f"best: {j['best']['experiment_id']} val {j['best']['val_score']:.5f}")


if __name__ == "__main__":
    sys.exit(main())
