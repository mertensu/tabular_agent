# 0001 — Own a constrained harness rather than serve other agents

- **Date:** 2026-07-28
- **Status:** accepted
- **Decided by:** Ulf
- **Commit:** `dfc6e7b`

## The problem

The original plan was to publish this project's capabilities as an **MCP server**.
MCP (Model Context Protocol) is a standard way to expose tools to an LLM client:
you run a server offering `inspect_data`, `train_model` and so on, and someone
else's agent — Claude Desktop, Claude Code, a custom client — calls them.

That is an attractive distribution story. It is also incompatible with the one
thing this project claims to guarantee.

The guarantee is **leakage-safe by construction**. Not "we warn you about
leakage", but "you cannot express it". Data leakage is when information that
would not be available at prediction time gets into training — the classic case
being an encoder fitted on the whole dataset, including the rows the model is
about to be scored on, which makes the cross-validation score look better than
the model really is.

Preventing that by construction requires one property: **our tools are the only
path to the data.**

## Why MCP breaks it

An MCP server is one tool source among several. A general-purpose client like
Claude Code also has a shell, a Python interpreter, and a scratch directory. So
the agent's cheapest path to "encode this column" is not our careful per-fold
`fit_encoding` — it is three lines of pandas in a scratch file, which is faster
to write, needs no argument validation, and silently fits on everything.

The escape hatch is not a hypothetical. It is the *path of least resistance*, and
guidance in a system prompt does not compete with convenience. A guarantee that
holds only when the agent chooses to cooperate is not a guarantee.

## The decision

**Own the harness.** We ship the loop that drives the model, and that loop
registers exactly our tools and nothing else. The LLM stays swappable — that is
genuinely valuable and costs us nothing — but the tool surface does not.

MCP is demoted from "the product" to "an optional internal transport". If we
later want the harness to reach tools over MCP, fine; what we will not do is hand
our tools to an agent whose other tools we do not control.

## Alternatives considered

**Ship the MCP server and document the risk.** Rejected: the honest version of
that documentation is "this is leakage-safe if the agent doesn't use its other
tools", which no one will read and nothing will enforce.

**Ship both.** Rejected for now. Two surfaces means the weaker one defines what
we can actually claim, and it is the one people would use first.

## Consequences

**Good**

- The claim becomes checkable. There is a test asserting the agent's tool set is
  *exactly* the expected functions, so adding a `read_file` fails the build
  rather than quietly reopening the hole (`tests/test_harness_agent.py`).
- Every later decision gets simpler, because "can the agent do X" always has a
  definite answer.
- The system prompt shrinks. It never has to say "please don't train outside the
  tool", because there is no other tool.

**Bad**

- We give up the distribution story. Nobody discovers this by adding a server to
  their existing client.
- We now own an agent loop: model configuration, retries, message history,
  budgets. All of it is work that MCP would have made someone else's problem, and
  decisions 0007, 0008, 0012 and 0013 exist only because of it.
- The spec's title still says "MCP-Server für …", which is now wrong. Left in
  place to avoid breaking references.

## Where it lives in the code

- `src/tabular_agent/harness/` — the whole harness.
- `src/tabular_agent/harness/agent.py` — `build_agent()` registers the tools;
  this list *is* the guarantee.
- `tests/test_harness_agent.py` — asserts the tool surface, deliberately
  brittle.
