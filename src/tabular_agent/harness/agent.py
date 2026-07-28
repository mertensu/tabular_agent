"""The PydanticAI loop (spec §13 revision).

This module is deliberately thin. All the behaviour lives in ``tools.py``; here
we only (a) declare which of those functions the model may call and (b) pick a
model. Keeping it thin is the point: the tool set is auditable at a glance, and
swapping Claude for GPT or Gemini is a one-line change.

`pydantic_ai` is imported lazily by ``build_agent`` so the tool layer stays
usable — and testable — without it.
"""

from dataclasses import dataclass

from . import tools
from .instructions import SYSTEM_PROMPT
from .session import RunSession

#: Default model. Any PydanticAI model string works — `openai:gpt-...`,
#: `google-gla:gemini-...` — because nothing above this line is Claude-specific.
DEFAULT_MODEL = "anthropic:claude-opus-5"


@dataclass
class Deps:
    """What the tools need but the model must not see or be able to forge."""

    session: RunSession


def build_agent(model: str = DEFAULT_MODEL):
    """An agent whose *entire* tool surface is the three functions below."""
    from pydantic_ai import Agent, RunContext

    agent = Agent(
        model,
        deps_type=Deps,
        system_prompt=SYSTEM_PROMPT,
        # The model name may be newer than the installed pydantic-ai's list of
        # known names; the string is passed through to the provider regardless.
        defer_model_check=True,
    )

    @agent.tool
    def inspect_data(ctx: RunContext[Deps], table: str | None = None) -> dict:
        """Profile a table: shape, per-column dtype/missing/unique, task type and metric.

        Args:
            table: Path to the table. Omit to use the run's raw input table.
        """
        return tools.inspect_data(ctx.deps.session, table)

    @agent.tool
    def train_tabular_model(
        ctx: RunContext[Deps],
        table: str | None = None,
        model_type: str = "lgbm",
        params: dict | None = None,
    ) -> dict:
        """Train a model with k-fold cross-validation and record the experiment.

        Args:
            table: Path to the feature table. Omit to use the run's raw input table.
            model_type: One of "baseline", "lgbm", "xgb", "catboost".
            params: Optional model hyper-parameters.
        """
        return tools.train_tabular_model(
            ctx.deps.session, table=table, model_type=model_type, params=params
        )

    @agent.tool
    def finalize_run(ctx: RunContext[Deps]) -> dict:
        """Package the best experiment into a predictor bundle and write the report.

        Call this exactly once, when you are done experimenting.
        """
        return tools.finalize_run(ctx.deps.session)

    return agent


def run(session: RunSession, prompt: str, model: str = DEFAULT_MODEL):
    """Drive one agent run against ``session``. Returns the PydanticAI result."""
    return build_agent(model).run_sync(prompt, deps=Deps(session=session))
