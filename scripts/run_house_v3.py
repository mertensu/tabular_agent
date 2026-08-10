"""house_prices run 3: target transforms available, traced with logfire offline.

Control is run 2 (`house_prices_no_id`, Kaggle 0.123). Same data, same prompt,
same model — the only change is that `train_tabular_model` now accepts
`target_transform`, so any difference is attributable to it plus model
nondeterminism.
"""
import json, os, sys, time
from collections import Counter

import logfire
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from tabular_agent.harness.agent import Deps, build_agent, gateway_model
from tabular_agent.harness.session import RunSession
from tabular_agent.harness.usage import context_profile, format_profile
from tabular_agent.taskspec import TaskSpec

REPO = "/Users/ulfmertens/Documents/tabular_agent"
RUN_DIR = os.path.join(REPO, "_runs", "house_prices_v3")
DATA = os.path.join(REPO, "house_prices", "train_no_id.csv")
MODEL = "tensorx/deepseek-v4-flash-0731"
SPANS = os.path.join(RUN_DIR, "spans.jsonl")

# Unchanged from runs 1 and 2, deliberately: the transform has to be discovered
# from the tool signature, not from a hint in the prompt. If it needs telling,
# that is a finding about the docstring.
PROMPT = """Build the best model you can for this dataset of house sales.
The target is SalePrice. Work until you stop making progress, then finalize."""


class FileExporter(SpanExporter):
    """Spans to a local JSONL. No collector, no network, greppable afterwards."""

    def export(self, spans):
        with open(SPANS, "a") as f:
            for s in spans:
                f.write(json.dumps({
                    "name": s.name,
                    "start": s.start_time,
                    "duration_ms": round((s.end_time - s.start_time) / 1e6, 1),
                    "attributes": {k: str(v)[:2000] for k, v in dict(s.attributes or {}).items()},
                }) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    open(SPANS, "w").close()

    logfire.configure(
        send_to_logfire=False,
        service_name="tabular_agent",
        console=False,
        additional_span_processors=[SimpleSpanProcessor(FileExporter())],
    )
    logfire.instrument_pydantic_ai()

    session = RunSession(RUN_DIR, DATA, TaskSpec(target="SalePrice"))
    agent = build_agent(gateway_model(MODEL))

    t0 = time.time()
    with logfire.span("house_prices run"):
        result = agent.run_sync(PROMPT, deps=Deps(session=session))
    elapsed = time.time() - t0

    print("=" * 70)
    print(result.output)
    print("=" * 70)
    print(f"elapsed: {elapsed / 60:.1f} min | usage: {result.usage}")

    with open(os.path.join(RUN_DIR, "transcript.json"), "wb") as f:
        f.write(result.all_messages_json())

    calls = [
        p.tool_name
        for m in result.all_messages()
        for p in getattr(m, "parts", [])
        if type(p).__name__ == "ToolCallPart"
    ]

    messages = json.loads(result.all_messages_json())
    with open(os.path.join(RUN_DIR, "run_journal.json")) as f:
        n_exp = len(json.load(f)["leaderboard"])
    profile = context_profile(messages, experiments=n_exp)
    with open(os.path.join(RUN_DIR, "context_profile.json"), "w") as f:
        json.dump(profile, f, indent=2)

    with open(os.path.join(RUN_DIR, "run_meta.json"), "w") as f:
        json.dump({
            "model": MODEL, "prompt": PROMPT, "data": DATA,
            "elapsed_min": round(elapsed / 60, 2),
            "final_output": result.output, "tool_calls": calls,
            "usage": str(result.usage),
        }, f, indent=2)

    print(f"\n{len(calls)} tool calls: {dict(Counter(calls).most_common())}")
    print()
    print(format_profile(profile, "context"))
    print(f"\nspans: {sum(1 for _ in open(SPANS))} in {SPANS}")


if __name__ == "__main__":
    sys.exit(main())
