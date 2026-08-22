"""house_prices v7: does the derived brief (0026) actually cut the orientation tax?

**A controlled repeat of v6, and deliberately nothing else.** Same dataset, same
model, same cap, same budget, same prompts — including the resume prompt, which
still says "earlier episodes did things you have no memory of" even though the
brief now contradicts it. Rewriting that sentence would improve the prompt and
destroy the comparison; the *only* difference between v6 and v7 must be the brief,
or the number this script prints means nothing.

The claim under test, from 0025's measurement:

    v6 orientation share per episode:  26%, 50%, 26%, 43%

where orientation = every `explore_journal` call plus every `inspect_data` after
the run's first. If the brief works, that share falls. If it does not move, 0026
is a well-argued no-op and should be marked as one.

Two secondary things worth reading off the output:

* **Does the redundant `inspect_data` stop?** v5 and v6 opened *every* episode
  with one, because `SYSTEM_PROMPT` said "always". It now says that only applies
  to a run that has not started. This is the cheapest possible check that a
  prompt change landed.
* **Do experiments per episode go up?** The tax is only worth removing if the
  freed calls turn into work. Cutting orientation and doing nothing with the room
  would be a wash.

Not library code, for the same reason v4-v6 were not: the loop is still a parked
design (0024), and moving it into `harness/` is a separate decision that should
not ride along with an experiment meant to test something else.
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
from tabular_agent.harness.brief import render_brief
from tabular_agent.harness.session import RunSession
from tabular_agent.harness.usage import context_profile
from tabular_agent.harness.stop import StopPolicy
from tabular_agent.taskspec import TaskSpec

REPO = "/Users/ulfmertens/Documents/tabular_agent"
RUN_DIR = os.path.join(REPO, "_runs", "house_prices_v7")
DATA = os.path.join(REPO, "house_prices", "train_no_id.csv")
MODEL = "tensorx/deepseek-v4-flash-0731"
SPANS = os.path.join(RUN_DIR, "spans.jsonl")

#: All four identical to v6. Changing any of them forfeits the comparison.
MAX_EPISODES = 8
TOOL_CALLS_PER_EPISODE = 16
MAX_EXPERIMENTS = 30

#: What v6 spent orienting, per episode, **recomputed with `orientation_share`
#: below** from `_runs/house_prices_v6/episodes.json`. It is deliberately not
#: 0025's published 26/50/26/43: those were counted by hand, and a baseline you
#: did not compute with the same code is not a baseline. The first run of this
#: script printed the hand-counted numbers and the comparison was misleading.
V6_ORIENTATION = [0.08, 0.32, 0.50, 0.32, 0.50]

FIRST = """Build the best model you can for this dataset of house sales.
The target is SalePrice.

Work in short stretches. When you have followed one idea far enough to know
whether it worked, stop and summarise what you did and what you would try next.
Do not call finalize_run unless you are genuinely finished with the whole run."""

# Byte-identical to v6's, on purpose. See the module docstring.
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


def orientation_share(calls: list[str], is_first_episode: bool) -> float:
    """The fraction of an episode spent working out where it already was.

    Counted exactly as 0025 counted it, so the numbers are comparable: every
    `explore_journal`, plus every `inspect_data` except the one the first episode
    is supposed to make.
    """
    if not calls:
        return 0.0
    inspects = calls.count("inspect_data")
    redundant = max(inspects - 1, 0) if is_first_episode else inspects
    return (calls.count("explore_journal") + redundant) / len(calls)


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
        # Captured before the episode runs: this is exactly the text the fresh
        # agent is about to be handed, and it belongs in the record next to what
        # the agent then did with it.
        brief = render_brief(session)
        agent = build_agent(gateway_model(MODEL))
        prompt = FIRST if i == 1 else RESUME

        t0 = time.time()
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
        share = orientation_share(calls, i == 1)

        episodes.append({
            "episode": i,
            "experiments_before": before,
            "experiments_after": after,
            "tool_calls": calls,
            "orientation_share": round(share, 3),
            "brief_chars": len(brief or ""),
            "brief": brief,
            "opened_with_inspect_data": calls[:1] == ["inspect_data"],
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
        print(f"  brief: {len(brief or '')} chars")
        print(f"  opened with: {' -> '.join(calls[:5])}")
        print(f"  orientation: {share:.0%}")
        print(f"  {dict(Counter(calls))}")

        with open(os.path.join(RUN_DIR, "episodes.json"), "w") as f:
            json.dump(episodes, f, indent=2)
        with open(os.path.join(RUN_DIR, f"transcript_ep{i}.json"), "w") as f:
            json.dump(raw, f)

        if after == before and not finalized(RUN_DIR):
            print("  !! no experiments added — stopping the loop")
            break

    print(f"\n{'=' * 70}")
    shares = [e["orientation_share"] for e in episodes]
    print(f"{len(episodes)} episodes, {n_experiments(RUN_DIR)} experiments, "
          f"{(time.time() - t_start) / 60:.1f} min total")
    print(f"orientation, v7 (brief):    {[f'{s:.0%}' for s in shares]}")
    print(f"orientation, v6 (no brief): {[f'{s:.0%}' for s in V6_ORIENTATION]}")
    if shares:
        print(f"mean {sum(shares) / len(shares):.0%} vs "
              f"{sum(V6_ORIENTATION) / len(V6_ORIENTATION):.0%}")
    redundant = sum(1 for e in episodes[1:] if e["opened_with_inspect_data"])
    print(f"episodes after the first that still opened with inspect_data: "
          f"{redundant}/{max(len(episodes) - 1, 0)} (v6: all of them)")
    print(f"peak context per episode: {[e['peak_input_tokens'] for e in episodes]}")
    j = journal.load(RUN_DIR)
    if j.get("best"):
        print(f"best: {j['best']['experiment_id']} val {j['best']['val_score']:.5f}")


if __name__ == "__main__":
    sys.exit(main())
