"""Live agent run on Kaggle house_prices, deepseek-v4-flash-0731 via Requesty."""
import json, os, sys, time

from tabular_agent.harness.agent import Deps, build_agent, gateway_model
from tabular_agent.harness.session import RunSession
from tabular_agent.taskspec import TaskSpec

REPO = "/Users/ulfmertens/Documents/tabular_agent"
RUN_DIR = os.path.join(REPO, "_runs", "house_prices")
DATA = os.path.join(REPO, "house_prices", "train.csv")
MODEL = "tensorx/deepseek-v4-flash-0731"

# Deliberately un-nudged. It does not mention `explore_experiment` or residuals:
# the open question from this session is whether a model reaches for the new tool
# on its own after a disappointing score, and a prompt that tells it to would
# answer a different question.
PROMPT = """Build the best model you can for this dataset of house sales.
The target is SalePrice. Work until you stop making progress, then finalize."""


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    session = RunSession(RUN_DIR, DATA, TaskSpec(target="SalePrice"))
    agent = build_agent(gateway_model(MODEL))

    t0 = time.time()
    error = None
    try:
        result = agent.run_sync(PROMPT, deps=Deps(session=session))
    except Exception as e:
        # A run that dies mid-way is still worth studying — arguably more so.
        # The journal on disk survives regardless; this keeps the transcript.
        error = f"{type(e).__name__}: {e}"
        print("RUN FAILED:", error)
        raise
    finally:
        elapsed = time.time() - t0
        print(f"\nelapsed: {elapsed / 60:.1f} min")

    print("=" * 70)
    print(result.output)
    print("=" * 70)
    print(f"usage: {result.usage}")

    # --- everything below is for studying the run afterwards ----------------
    transcript = os.path.join(RUN_DIR, "transcript.json")
    with open(transcript, "wb") as f:
        f.write(result.all_messages_json())

    calls = []
    for msg in result.all_messages():
        for part in getattr(msg, "parts", []):
            if type(part).__name__ == "ToolCallPart":
                calls.append(part.tool_name)

    with open(os.path.join(RUN_DIR, "run_meta.json"), "w") as f:
        json.dump({
            "model": MODEL,
            "prompt": PROMPT,
            "data": DATA,
            "elapsed_min": round(elapsed / 60, 2),
            "error": error,
            "final_output": result.output,
            "tool_calls": calls,
            "usage": str(result.usage),
        }, f, indent=2)

    from collections import Counter
    print(f"\n{len(calls)} tool calls:")
    for name, n in Counter(calls).most_common():
        print(f"  {n:>3}  {name}")
    print("\norder:", " -> ".join(calls))
    print(f"\nsaved: {transcript}")


if __name__ == "__main__":
    sys.exit(main())
