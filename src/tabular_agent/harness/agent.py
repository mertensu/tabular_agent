"""The PydanticAI loop (spec §13 revision).

This module is deliberately thin. All the behaviour lives in ``tools.py``; here
we only (a) declare which of those functions the model may call and (b) pick a
model. Keeping it thin is the point: the tool set is auditable at a glance, and
swapping Claude for GPT or Gemini is a one-line change.

`pydantic_ai` is imported lazily by ``build_agent`` so the tool layer stays
usable — and testable — without it.
"""

import os
from dataclasses import dataclass

from . import tools
from .instructions import SYSTEM_PROMPT
from .session import RunSession

#: Default model. Any PydanticAI model string works — `openai:gpt-...`,
#: `google-gla:gemini-...` — because nothing above this line is Claude-specific.
DEFAULT_MODEL = "anthropic:claude-opus-5"

#: OpenAI-compatible gateway used when there is no first-party provider key.
REQUESTY_BASE_URL = "https://router.requesty.ai/v1"

#: Untracked `KEY=value` file `load_api_key` falls back to (gitignored: `*.env`).
DEFAULT_KEY_PATH = "config.env"

#: Model served through the gateway. Must support tool calling — the harness is
#: nothing but tool calls, so a chat-only model fails on the first turn.
DEFAULT_GATEWAY_MODEL = "tensorx/deepseek-v4-flash"


def _find_key_file(key_path: str) -> str | None:
    """Locate a relative key file from anywhere sane — notably `notebooks/`.

    A notebook's working directory is the notebook's own folder, not the repo
    root, so a bare relative path resolves to nothing. Walk up from the working
    directory and from this package, and take the first hit.
    """
    if os.path.isabs(key_path):
        return key_path if os.path.exists(key_path) else None
    starts = [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]
    seen = set()
    for start in starts:
        d = start
        while d not in seen:
            seen.add(d)
            candidate = os.path.join(d, key_path)
            if os.path.exists(candidate):
                return candidate
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
    return None


def load_api_key(env_var: str = "REQUESTY_API_KEY", key_path: str = DEFAULT_KEY_PATH) -> str:
    """The API key from the environment, falling back to an untracked env file.

    The file form exists so a notebook can be run on a fresh machine without
    exporting anything first. Everything matching `*.env` is gitignored, and the
    key is never written as a literal anywhere in the repo — which is also why
    this returns the key instead of logging or echoing it.
    """
    key = os.environ.get(env_var)
    if key:
        return key.strip()
    found = _find_key_file(key_path)
    if found:
        with open(found) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, _, value = line.partition("=")
                if name.strip() == env_var:
                    return value.strip().strip("'\"")
    raise RuntimeError(
        f"no API key: set ${env_var} or put `{env_var}=...` in {key_path} (untracked)."
    )


def gateway_model(model_name: str, *, base_url: str = REQUESTY_BASE_URL, api_key: str | None = None):
    """A model served by an OpenAI-compatible gateway (Requesty by default).

    Returns a PydanticAI ``Model`` rather than a string, which is the whole
    point of `build_agent` accepting either: the harness stays provider-agnostic
    (§13) and the choice of who actually serves the tokens is made at the call
    site, not baked into the agent.
    """
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(base_url=base_url, api_key=api_key or load_api_key()),
    )


@dataclass
class Deps:
    """What the tools need but the model must not see or be able to forge."""

    session: RunSession


def _retryable(call):
    """Turn a rejected argument into a message the model can act on.

    Every tool here validates its inputs and raises ``ValueError`` — an
    undeclared ordinal level, a missing column, encoding the target. Letting
    that propagate ends the run, throwing away every experiment already
    completed because of one malformed argument. As ``ModelRetry`` it costs a
    single turn instead, and the message *is* the correction: the validation
    errors are written to say what was wrong and what to pass.

    Only ``ValueError`` is caught. A bug in the pipeline should still stop the
    run loudly rather than be handed to the model to guess at.
    """
    from pydantic_ai import ModelRetry

    try:
        return call()
    except ValueError as e:
        raise ModelRetry(str(e)) from e


def build_agent(model=DEFAULT_MODEL):
    """An agent whose *entire* tool surface is the four functions below.

    ``model`` is either a PydanticAI model string (``"anthropic:..."``,
    ``"openai:..."``) or a ``Model`` object — e.g. from :func:`gateway_model`,
    for a provider reached through an OpenAI-compatible gateway.
    """
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
        return _retryable(lambda: tools.inspect_data(ctx.deps.session, table))

    @agent.tool
    def fit_encoding(
        ctx: RunContext[Deps],
        columns: list[str],
        method: str = "frequency",
        table: str | None = None,
        categories: dict[str, list] | None = None,
        smoothing: float = 10.0,
    ) -> dict:
        """Encode categorical columns, writing a new feature table to train on.

        Args:
            columns: The categorical columns to encode.
            method: "frequency" (how common each category is), "ordinal" (ordered
                categories -> their rank; requires `categories`), or "target"
                (out-of-fold target mean; strongest, but only worth it for
                columns with enough rows per category).
            table: Path to the source table. Omit to use the run's raw input table.
            categories: For method="ordinal" only: {column: [level, ...]} in
                increasing order. State the real order (e.g. small, medium,
                large) — it is never inferred from the data.
            smoothing: For method="target" only: pulls small categories towards
                the global mean.
        """
        return _retryable(
            lambda: tools.fit_encoding(
                ctx.deps.session,
                columns=columns,
                method=method,
                table=table,
                categories=categories,
                smoothing=smoothing,
            )
        )

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
        return _retryable(
            lambda: tools.train_tabular_model(
                ctx.deps.session, table=table, model_type=model_type, params=params
            )
        )

    @agent.tool
    def finalize_run(ctx: RunContext[Deps]) -> dict:
        """Package the best experiment into a predictor bundle and write the report.

        Call this exactly once, when you are done experimenting.
        """
        return _retryable(lambda: tools.finalize_run(ctx.deps.session))

    return agent


def run(session: RunSession, prompt: str, model=DEFAULT_MODEL):
    """Drive one agent run against ``session``. Returns the PydanticAI result."""
    return build_agent(model).run_sync(prompt, deps=Deps(session=session))
