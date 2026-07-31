# 0002 — PydanticAI as the harness framework

- **Date:** 2026-07-28
- **Status:** accepted (was a leaning; has held)
- **Decided by:** Ulf
- **Follows:** [0001](0001-own-the-harness.md)

## The problem

Having decided to own the agent loop (0001), we need something to build it on.
An "agent loop" here is small but fiddly: send the conversation to a model, get
back either text or a request to call a tool, run the tool, append the result,
repeat until the model stops asking. The hard parts are tool schemas, message
history, retries, and streaming.

Requirements, in priority order:

1. **Model-agnostic.** "Only the LLM is swappable" is the point of 0001. A
   framework tied to one vendor negates it.
2. **Typed tools with no hidden extras.** Our tool set *is* the safety argument,
   so a framework that injects its own tools — a file reader, a code interpreter
   — is disqualified outright.
3. **Headless and embeddable.** It has to run inside a notebook cell and inside a
   test, with no UI.
4. **Message history we can manipulate.** Long runs will need compaction, so we
   must be able to see and rewrite the history rather than have it hidden.

## The decision

**PydanticAI.** Tools are plain Python functions with type hints; the schema is
derived from the signature and docstring. It supports any provider, including
OpenAI-compatible gateways. It is a library rather than an application. History
is a list we can inspect.

## Alternatives considered

**The Claude Agent SDK, or OpenAI's.** Both are more polished. Both are built
around one vendor's model, which is exactly the lock-in requirement 1 rules out.

**smolagents and other code-first agent frameworks.** These work by having the
model *write Python* which the framework executes — an elegant idea, and for this
project an anti-fit. The escape hatch that decision 0001 exists to close is
promoted to the framework's core feature. We do give the agent a Python sandbox
later (see 0009–0011), but a deliberately crippled one, which is the opposite
posture.

**Write the loop ourselves.** Perhaps 200 lines for the happy path. Rejected
because the unglamorous parts — tool-call schemas per provider, retries,
streaming, structured errors — are where the time actually goes, and none of it
is our problem to solve.

## Consequences

**Good**

- Swapping models is one argument. `build_agent()` takes either a provider string
  (`"anthropic:claude-opus-5"`) or a `Model` object, so running the whole harness
  on a non-Anthropic model through an OpenAI-compatible gateway needed no changes
  to any tool.
- Tool docstrings become the tool documentation the model reads. That is a nice
  forcing function: the docstring has to be written for its actual audience.

**Bad**

- The library is young and moves fast. We already depend on things that could
  churn: `Agent(retries=...)`, `Agent(max_concurrency=...)`, `ModelRetry`, and —
  worse — the private `agent._function_toolset.tools` that the tool-surface test
  reaches into. That test is the one most likely to break on upgrade, and it is
  load-bearing.
- Its list of known model names lags releases, so new models fail validation.
  Worked around with `defer_model_check=True`, which passes the string through to
  the provider unchecked.

## Where it lives in the code

- `src/tabular_agent/harness/agent.py` — the entire loop, deliberately thin.
  Behaviour lives in `tools.py`; this file only says which functions the model may
  call and which model to use.
- `pydantic-ai` is a hard dependency (2.19.0 at time of writing).
