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
from .brief import render_brief
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

#: Corrections allowed per tool before the run is abandoned. Bounded on purpose:
#: a model looping on the same rejected argument should stop, not burn the budget.
TOOL_RETRIES = 3


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

    ``ValueError`` (our own input validation), ``FileNotFoundError`` (a path the
    model invented) and ``SandboxError`` (code the model wrote that raised, timed
    out, or imported a blocked library) are all argument problems — the argument
    just happens to be a program in the last case, and those messages are written
    to teach. Anything else is a bug in the pipeline and should stop the run
    loudly rather than be handed to the model to guess at.
    """
    from pydantic_ai import ModelRetry

    from ..sandbox import SandboxError

    try:
        return call()
    except (ValueError, FileNotFoundError, SandboxError) as e:
        raise ModelRetry(str(e)) from e


def build_agent(model=DEFAULT_MODEL):
    """An agent whose *entire* tool surface is the eight functions below.

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
        # PydanticAI allows one retry per tool by default. Exhausting it ends the
        # run and discards every experiment already completed, which is a brutal
        # price for two bad arguments in a run that makes dozens of calls.
        retries=TOOL_RETRIES,
    )
    # NOTE ON CONCURRENT TOOL CALLS (0031). A model may put several calls in one
    # message, and PydanticAI runs those concurrently by default.
    #
    # This constructor used to pass `max_concurrency=1` with a comment claiming it
    # meant "one tool call at a time". It does not — it limits concurrent agent
    # *runs*. The error was invisible for months because earlier models emitted
    # one call per message, and Rossmann v3 found it: four duplicate table ids and
    # a parquet corrupted mid-write, killing a run that held the best model any
    # run had produced.
    #
    # The fix is deliberately *not* a setting here. Correctness now lives in the
    # primitives — `journal.next_table_id` and `next_experiment_id` claim by
    # exclusive create, `io.save_parquet` replaces atomically — so concurrent
    # calls are safe rather than merely unlikely. That also covers two *processes*
    # sharing a run dir, which no in-process setting can.
    #
    # A driver that wants ordering as well can wrap its run in
    # `agent.parallel_tool_call_execution_mode("sequential")`; nothing depends on
    # it for correctness.

    @agent.system_prompt(dynamic=True)
    def run_state(ctx: RunContext[Deps]) -> str:
        """Where this run currently stands, for an agent that was not here for it.

        Dynamic rather than baked into `SYSTEM_PROMPT` because it is a function of
        the journal at the moment the episode starts, and an episode is a fresh
        agent on a run it has no memory of (0024). Returns "" on a run where
        nothing has happened, so a first episode is byte-identical to a run that
        never had episodes at all.
        """
        return render_brief(ctx.deps.session) or ""

    @agent.tool
    def inspect_data(ctx: RunContext[Deps], table: str | None = None) -> dict:
        """Profile a table: shape, per-column dtype/missing/unique, task type and metric.

        Read `columns_not_available_at_predict_time` if it is there: those columns
        exist while you train and will not exist when the model is asked for a
        prediction, so training on them is refused. Removing one takes a guarded
        drop — `df.drop(columns=[...], errors="ignore")` — because the transform
        code is stored and re-run on rows that never had the column.

        Args:
            table: Path to the table. Omit to use the run's work table.
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
            table: Path to the source table. Omit to use the run's work table.
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
    def transform_features(
        ctx: RunContext[Deps],
        code: str,
        table: str | None = None,
    ) -> dict:
        """Derive new columns by running your own pandas/numpy code, into a new table.

        For row-local feature engineering: ratios, differences, distances, date
        parts, flags. The code runs in a sandbox with pandas, numpy and scipy
        available; the model and sklearn libraries are blocked, so anything that
        has to *learn* from the data (a category's target mean, a vocabulary)
        belongs in `fit_encoding` instead.

        Args:
            code: Python defining `transform_features(df)`, returning a DataFrame
                with the **same rows in the same order** (add or replace columns;
                never drop, add or reorder rows). The target column is **not** in
                `df` — it is not available to your code, at training or at
                predict time. Example:

                    import numpy as np
                    def transform_features(df):
                        df = df.copy()
                        df["area"] = df["width"] * df["height"]
                        df["log_income"] = np.log1p(df["income"])
                        return df

            table: Path to the source table. Omit to use the run's raw input
                table; pass a table from an earlier call to stack on top of it.
        """
        return _retryable(lambda: tools.transform_features(ctx.deps.session, code, table=table))

    @agent.tool
    def train_tabular_model(
        ctx: RunContext[Deps],
        table: str | None = None,
        model_type: str = "lgbm",
        params: dict | None = None,
        target_transform: str = "none",
    ) -> dict:
        """Train a model with k-fold cross-validation and record the experiment.

        `vs_baseline` compares against the trivial constant model.
        `vs_source_table` compares against the same model, params and
        `target_transform` on this table's `source_table`, so the columns you
        added are the only difference — it is `null` when you change more than
        one of those at once, and `vs_source_basis` then says what would
        produce it.

        Returns `trained: false` instead of an experiment once this run has
        stopped making progress; call `finalize_run` at that point.

        Args:
            table: Path to the feature table. Omit to use the run's work table.
            model_type: One of "baseline", "lgbm", "xgb", "catboost".
            params: Optional model hyper-parameters.
            target_transform: What the model is fitted on — "none" (the target
                as-is), "log1p", or "sqrt". Regression only, and "log1p"/"sqrt"
                need a non-negative target. Changes only what is *fitted*: every
                score is still reported on the original scale, so a transformed
                and an untransformed experiment stay directly comparable. Worth
                an experiment when the target is strongly skewed, or when the
                metric already works on a compressed scale — fitting on the
                scale the metric measures makes the model optimise the thing it
                is judged on.
        """
        return _retryable(
            lambda: tools.train_tabular_model(
                ctx.deps.session,
                table=table,
                model_type=model_type,
                params=params,
                target_transform=target_transform,
            )
        )

    @agent.tool
    def explain_table(
        ctx: RunContext[Deps],
        table: str | None = None,
        model_type: str = "lgbm",
        params: dict | None = None,
        target_transform: str = "none",
    ) -> dict:
        """Measure what each tool call's columns contributed, by refitting without each.

        Refits holding out one contributing table's columns at a time; each
        `contribution` in `credit_by_table` is what the score loses without them.
        Leave-one-out, so each is that table's worth given the rest. Needs no
        earlier experiment on the columns it scores, which makes it the only way
        to separate ideas added in different steps.

        Costs one fit per contributing table plus one, records no experiment, and
        counts against no budget — including on a table training was declined for.

        Args:
            table: Path to the feature table. Omit to use the run's work table.
            model_type: One of "baseline", "lgbm", "xgb", "catboost".
            params: Optional model hyper-parameters.
            target_transform: Match the experiment you are explaining, or the
                contributions describe a fit you never ran.
        """
        return _retryable(
            lambda: tools.explain_table(
                ctx.deps.session,
                table=table,
                model_type=model_type,
                params=params,
                target_transform=target_transform,
            )
        )

    @agent.tool
    def explore_experiment(
        ctx: RunContext[Deps],
        exp_id: str,
        code: str,
    ) -> dict:
        """Look at where a trained model was wrong, by running your own code on its rows.

        Answers "which rows is this model failing on", which is the question a
        score cannot answer and the fastest route to the next feature worth
        building. Nothing computed here is kept or can become a feature — the
        result is shown to you and discarded — so group by the target freely.

        Costs nothing and records no experiment, including on a table training
        was declined for.

        Args:
            exp_id: The `experiment_id` of an earlier `train_tabular_model` call.
            code: Python defining `explore(df)`, returning a small summary — a
                number, a dict, or a DataFrame/Series (capped at 200 rows).
                Anything printed comes back too. `df` is the experiment's whole
                source table — including the string columns the model could not
                use — plus:

                    oof_pred    the out-of-fold prediction for that row
                                (multiclass: `oof_pred_<class>` per class,
                                `oof_pred_label`, `p_true`, `correct`)
                    residual    target - oof_pred, signed (regression/binary)
                    fold        which CV fold the row was held out in

                Example:

                    def explore(df):
                        by_city = df.groupby("city")["residual"].agg(["mean", "size"])
                        print(df["residual"].describe())
                        return by_city.round(3).to_dict("index")
        """
        return _retryable(
            lambda: tools.explore_experiment(ctx.deps.session, exp_id=exp_id, code=code)
        )

    @agent.tool
    def explore_journal(ctx: RunContext[Deps], code: str) -> dict:
        """Query this run's own history — every experiment and table so far — with code.

        Use instead of comparing results by eye. It answers questions across the
        whole run: which encoding method actually paid, whether a model type is
        better once the feature table is held constant, which tables have never
        been trained on. Costs nothing and records nothing.

        Args:
            code: Python defining `explore(...)`, taking any of the frames below
                by name and returning a small summary (capped at 200 rows).
                Anything printed comes back too.

                    lb      one row per experiment: id, model_type, params (a
                            JSON string), target_transform, source_table, metric,
                            val_score, val_se, cv_score, fold_spread, vs_baseline,
                            vs_source_table, n_features
                    tables  one row per feature table: id, path, method,
                            source_table, n_columns, columns, code
                    notes   whatever has been appended to the run log
                    run     dict: task, split, stop, best, baseline,
                            higher_is_better

                Remember `higher_is_better` when ranking — for most metrics here
                the better score is the lower one. Example:

                    def explore(lb, tables):
                        m = lb.merge(tables, left_on="source_table",
                                     right_on="path", how="left")
                        return m.groupby("method")["val_score"].agg(
                            ["mean", "min", "size"]).round(4).to_dict("index")
        """
        return _retryable(lambda: tools.explore_journal(ctx.deps.session, code))

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
