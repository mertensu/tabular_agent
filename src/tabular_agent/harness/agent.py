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
from .instructions import FORECAST_SYSTEM_PROMPT, SYSTEM_PROMPT
from .session import RunSession

#: Default model. Any PydanticAI model string works — `openai:gpt-...`,
#: `google:gemini-...` — because nothing above this line is Claude-specific.
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


def _retryable(call, ctx=None):
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

    ``ctx`` is the tool's ``RunContext``. Given it, the *last* allowed correction
    lands softly instead of off a cliff: PydanticAI counts retries per tool name
    and, once ``retries >= max_retries``, converts the next ``ModelRetry`` into an
    ``UnexpectedModelBehavior`` that ends the whole run — discarding every
    experiment already completed and denying the model the chance to finalize. A
    model looping on the same rejected ``transform_features`` code hit exactly
    that in the Rossmann RCs. So on the attempt that would trip the ceiling we
    return a normal tool result rather than raising: the loop lives, and the model
    can take a different approach or call ``finalize_run`` on the best table it
    already has. The earlier corrections (attempts within budget) are unchanged.
    """
    from pydantic_ai import ModelRetry

    from ..sandbox import SandboxError

    try:
        return call()
    except (ValueError, FileNotFoundError, SandboxError) as e:
        if ctx is not None and ctx.max_retries and ctx.retry >= ctx.max_retries:
            return {
                "rejected": True,
                "attempts": ctx.retry + 1,
                "reason": str(e),
                "message": (
                    f"This call to {ctx.tool_name!r} was rejected {ctx.retry + 1} "
                    "times for the same kind of reason and produced no table. Stop "
                    "retrying this approach: repeating it will not help. Either take "
                    "a substantially different approach — a different tool, or "
                    "clearly different code — or call finalize_run to finish with "
                    "the best table you already have."
                ),
            }
        raise ModelRetry(str(e)) from e


def _build_minimal_agent(model):
    """Register only the six operations in the v0.2 product contract."""
    from pydantic_ai import Agent, RunContext

    agent = Agent(
        model,
        deps_type=Deps,
        system_prompt=SYSTEM_PROMPT,
        defer_model_check=True,
        retries=TOOL_RETRIES,
    )

    @agent.system_prompt(dynamic=True)
    def workflow_contract(ctx: RunContext[Deps]) -> str:
        return FORECAST_SYSTEM_PROMPT if ctx.deps.session.workflow == "forecast" else ""

    @agent.system_prompt(dynamic=True)
    def run_state(ctx: RunContext[Deps]) -> str:
        return render_brief(ctx.deps.session) or ""

    @agent.tool
    def inspect_data(ctx: RunContext[Deps], table: str | None = None) -> dict:
        """Inspect schema, target, metric and prediction-time availability."""
        return _retryable(lambda: tools.inspect_data(ctx.deps.session, table), ctx)

    @agent.tool
    def fit_encoding(
        ctx: RunContext[Deps], columns: list[str], method: str = "frequency",
        table: str | None = None, categories: dict[str, list] | None = None,
        smoothing: float = 10.0, stats: list[str] | None = None,
    ) -> dict:
        """Fit a replayable frequency, ordinal, or cross-fitted target encoding."""
        return _retryable(lambda: tools.fit_encoding(
            ctx.deps.session, columns=columns, method=method, table=table,
            categories=categories, smoothing=smoothing, stats=stats,
        ), ctx)

    @agent.tool
    def fit_group_stats(
        ctx: RunContext[Deps], by: list[str], value: str, stat: str = "mean",
        table: str | None = None,
    ) -> dict:
        """Fit and replay one non-target statistic per group."""
        return _retryable(lambda: tools.fit_group_stats(
            ctx.deps.session, by=by, value=value, stat=stat, table=table,
        ), ctx)

    @agent.tool
    def filter_rows(
        ctx: RunContext[Deps], code: str, table: str | None = None,
    ) -> dict:
        """Select training rows only; the filter never runs when serving."""
        return _retryable(lambda: tools.filter_rows(ctx.deps.session, code, table=table), ctx)

    @agent.tool
    def transform_features(
        ctx: RunContext[Deps], code: str, table: str | None = None,
    ) -> dict:
        """Run target-hidden, row-local feature code and persist its lineage."""
        return _retryable(lambda: tools.transform_features(ctx.deps.session, code, table=table), ctx)

    @agent.tool
    def finalize_run(ctx: RunContext[Deps]) -> dict:
        """Fit one fixed model on the best table and write the bundle and report."""
        return _retryable(lambda: tools.finalize_run(ctx.deps.session), ctx)

    return agent


def build_agent(model=DEFAULT_MODEL, *, profile: str = "full", workflow: str = "general"):
    """An agent whose entire tool surface is selected by profile and workflow.

    ``model`` is either a PydanticAI model string (``"anthropic:..."``,
    ``"openai:..."``, ``"google:..."``) or a ``Model`` object — e.g. from
    :func:`gateway_model`, for a provider reached through an OpenAI-compatible
    gateway.
    """
    if profile not in {"full", "minimal"}:
        raise ValueError(f"unknown agent profile {profile!r}; use 'full' or 'minimal'")
    if workflow not in {"general", "iid", "forecast"}:
        raise ValueError(
            f"unknown agent workflow {workflow!r}; use 'general', 'iid', or 'forecast'"
        )
    if profile == "minimal":
        if workflow == "forecast":
            raise ValueError("the forecast workflow requires the full profile")
        return _build_minimal_agent(model)

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
    def workflow_contract(ctx: RunContext[Deps]) -> str:
        return FORECAST_SYSTEM_PROMPT if ctx.deps.session.workflow == "forecast" else ""

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
        return _retryable(lambda: tools.inspect_data(ctx.deps.session, table), ctx)

    @agent.tool
    def fit_encoding(
        ctx: RunContext[Deps],
        columns: list[str],
        method: str = "frequency",
        table: str | None = None,
        categories: dict[str, list] | None = None,
        smoothing: float = 10.0,
        stats: list[str] | None = None,
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
            stats: For method="target" only. Which statistics of the target to
                summarise each category by — any of "mean", "std", "median",
                "count". Defaults to ["mean"], the classical target encoding.
                The mean alone hides how spread out a category's target is and
                how many rows the estimate rests on; asking for "std" and
                "count" alongside it gives the model both, in one call and one
                new table, instead of a separate table per idea.
        """
        return _retryable(
            lambda: tools.fit_encoding(
                ctx.deps.session,
                columns=columns,
                method=method,
                table=table,
                categories=categories,
                smoothing=smoothing,
                stats=stats,
            ),
            ctx,
        )

    @agent.tool
    def fit_group_stats(
        ctx: RunContext[Deps],
        by: list[str],
        value: str,
        stat: str = "mean",
        table: str | None = None,
    ) -> dict:
        """Summarise a column per group — the group's typical value, as a new column.

        For "what is normal for this customer / route / location": the average,
        median or spread of some column across all the rows sharing a key. The
        same thing written as `groupby(...).transform("mean")` in
        `transform_features` is rejected, because that recomputes the number from
        whatever rows are in the frame; here it is fitted once and looked up, so
        a single row gets the same feature it would in a batch of a million.

        Always fitted out of fold, so a row never contributes to the statistic it
        receives. Adds one column named `<by>_<value>_<stat>`.

        `value` does not need to exist at predict time — only `by` does. That is
        what makes this the way to use a quantity you can only measure after the
        fact: as its group's history, never as itself. Drop the column itself
        before training if it will not be there.

        Args:
            by: Column(s) forming the group key. These are read at predict time,
                so they must exist then.
            value: The column being summarised. Not the target — a per-group
                target mean is `fit_encoding(method="target")`.
            stat: One of "mean", "median", "std", "min", "max", "sum", "count",
                "nunique".
            table: Path to the source table. Omit to use the run's work table.
        """
        return _retryable(
            lambda: tools.fit_group_stats(
                ctx.deps.session, by=by, value=value, stat=stat, table=table
            ),
            ctx,
        )

    def fit_history(
        ctx: RunContext[Deps],
        by: list[str],
        value: str,
        windows: list,
        keep: int | None = None,
        table: str | None = None,
    ) -> dict:
        """Give each row a trailing average of its own group's recent past.

        For the level a group has been running at lately — the one fact about a
        group that depends on *when* as well as *how much*, which
        `fit_group_stats` and `fit_encoding` both average away by summarising all
        of a group's rows at once.

        Every column this adds is an average over a stretch of the past, beside a
        count of how many observations it found. **There is no lag and no
        last-known-value option.** A single row's value is not offered, because
        this tool's history is frozen: one stored observation is the *same number*
        for every row you predict, while it varies across the folds you train on,
        so a model learns to use it and is then served a constant.

        Unlike `fit_group_stats`, `value` may be the target — a summary of the
        target's own past can only come from stored history, never from a served
        frame.

        Each fold's history stops at the end of that fold's training rows, so a
        validation row sees what a real prediction the same distance ahead would
        see. This scores lower than a window that peeks at neighbouring rows, and
        that gap is what the peeking was worth. Only for runs split on an
        ordering column; rows in the first period keep a missing value.

        Args:
            by: Column(s) forming the group key — the series to store one of.
                Read at predict time, so they must exist then.
            value: The column whose past is stored. May be the target.
            keep: Observations retained per group. Defaults to just enough for
                the longest window; raise it for a series with gaps.
            windows: Trailing averages, as [length, statistic] pairs — e.g.
                [[90, "mean"], [364, "median"]] for "this group's mean over the
                90 periods before this row" and "its median over the last 364".
                Statistics are mean, std and median. A window averages a stretch
                of observations, so it cannot be thrown by a single unusual day —
                which is the entire reason this tool has no lag argument. To
                reach a *seasonal* value, put the season in the key rather than
                in a distance: group by the entity **and** its position in the
                cycle, then take a window spanning several cycles. That averages
                the comparable periods instead of reading the single one that
                happens to sit a fixed step back.

                Length is the whole game, and it fails at *both* ends. Too short
                and the window holds almost nothing, because stored history stops
                where the forecast starts; each length adds a `_count` column
                saying how many observations it really held.

                Too long and the folds cannot fill it while a served row can, so
                the column you train on and the one you serve summarise different
                numbers of observations. The result reports
                `shortest_training_prefix` and `longest_training_prefix` so you
                can see where a window sits, but **be warned that none of these
                numbers reliably predicts whether a length will work** — on one
                dataset the same window was worth +0.047 under one fold count and
                −0.093 under another with identical counts on both sides. Try a
                length, read the probe score, and trust that over the diagnostics.
            table: Path to the source table. Omit to use the run's work table.
        """
        return _retryable(
            lambda: tools.fit_history(
                ctx.deps.session, by=by, value=value,
                keep=keep, table=table, windows=windows,
            ),
            ctx,
        )

    if workflow != "iid":
        agent.tool(fit_history)

    def fit_stream_aggregate(
        ctx: RunContext[Deps],
        stream: str,
        by: list[str],
        value: str,
        leads: list[float],
        stats: list[str] | None = None,
        table: str | None = None,
    ) -> dict:
        """Summarise a second input per row, counting only what was knowable in time.

        Use when the run declares a **stream** — `inspect_data` lists them. A
        stream is a related file at a finer grain than the table: several of its
        records belong to one row of yours, and each record says both what time it
        is *about* and when it became *knowable*.

        That second time is the point. Records keep arriving right up to the
        moment a thing happens, so a training row can be shown bookings made the
        same day while a row you are asked to forecast never can. Summing the file
        as it stands builds exactly that column, it scores beautifully in
        validation — which is drawn from the same uncut file — and it is worth
        much less than it looks at predict time.

        `leads` fixes it. `leads=[7]` counts only the records observed at least 7
        periods before each row, which is what a forecast issued a week ahead
        would have had. Several leads are several features, not alternatives.

        Every statistic gets a `_count` column beside it. Read them together: an
        aggregate over one record and one over forty are the same number under the
        same name, and only the count separates them.

        Args:
            stream: Name of a stream declared on this run — see `inspect_data`.
            by: Column(s) forming the join key. Must carry the **same names** in
                the table and in the stream, and must exist at predict time.
            value: The stream column to aggregate. It lives in the stream, not in
                your table.
            leads: How far ahead of a row a record had to be observed to count,
                in the ordering column's units — e.g. [7, 30]. Each must be at
                least the task's declared forecast_horizon, since no row you are
                scored on could have anything more recent. Note the horizon is the
                *floor* of a scored block: a lead at the floor is honest for the
                block's earliest rows and generous towards its far edge, and the
                result reports `block_span` so you can pick against it.
            stats: Which statistics of `value`, from sum, mean and std. Defaults
                to ["sum"]. The count comes with all of them, always.
            table: Path to the source table. Omit to use the run's work table.
        """
        return _retryable(
            lambda: tools.fit_stream_aggregate(
                ctx.deps.session, stream=stream, by=by, value=value, leads=leads,
                stats=stats, table=table,
            ),
            ctx,
        )

    if workflow != "iid":
        agent.tool(fit_stream_aggregate)

    def fit_future_covariate_window(
        ctx: RunContext[Deps],
        value: str,
        aggregates: list,
        by: list[str] | None = None,
        table: str | None = None,
    ) -> dict:
        """Summarise a *future-known* covariate over each series and delivered block.

        The exogenous counterpart to `fit_history`. History gives a row a trailing
        average of its own group's *target* past. This gives it aggregates of a
        covariate whose values over the forecast horizon are **known at issue
        time** — a weather forecast, a planned promotion, a published price — so it
        may look *forward* across the horizon, which history cannot: the target is
        unknown there, a future-known covariate is not.

        Only a column the run declared `future_known` is eligible; the target and
        any undeclared column are refused, because a value merely *present* over
        the horizon in your data (a realised future weather column in a test file)
        is not the same as one *known* at issue time in production, and only the
        run's owner can assert which it is.

        Every aggregate is keyed within one series and one delivered serving block
        (one forecast horizon long) and resets at the block start. At serving you
        are handed exactly one such block per series, so the same block-local
        statistic is recomputed from the delivered rows alone — no target, no
        stored freeze.

        Args:
            value: The future-known covariate to aggregate. Must be a column the
                plan declared in `future_known`.
            aggregates: The windows to build, as a list of objects. Families:
                - {"family": "period", "unit": "day", "stat": "sum"} — a whole
                  calendar unit ("hour"/"day"/"week"/"month", or "block" for the
                  whole delivered block), which may look forward *within the
                  delivered block*, e.g. total irradiance this day.
                - {"family": "trailing", "window": 4, "stat": "mean"} — a causal
                  rolling window of `window` rows that resets at the block start.
                - {"family": "forward", "window": 12, "stat": "mean"} — a window of
                  `window` rows starting at the row and running ahead, clipped at
                  the block end, e.g. mean forecast temperature over the next 3h.
                Statistics: mean, sum, min, max, std, count. Unlike a history
                window, sum/min/max are allowed — the covariate is a known fact
                over the block, not a single draw of the target.
            by: Column(s) forming the series key — read at predict time, so they
                must be serving columns. Omit for a single series. An aggregate
                never crosses a series or a block boundary.
            table: Path to the source table. Omit to use the run's work table.
        """
        return _retryable(
            lambda: tools.fit_future_covariate_window(
                ctx.deps.session, value=value, aggregates=aggregates,
                by=by, table=table,
            ),
            ctx,
        )

    if workflow == "forecast":
        agent.tool(fit_future_covariate_window)

    @agent.tool
    def filter_rows(
        ctx: RunContext[Deps],
        code: str,
        table: str | None = None,
    ) -> dict:
        """Write a table with fewer rows, to train on a subset — never to score one.

        Use when some rows should not be learned from: rows the metric does not
        score, rows from a regime you are not being asked to predict, outliers
        whose error swamps the fit. This is the only step that does **not** run at
        predict time — a served model must answer for every row it is given — so
        it changes what the model learns from and nothing about how it is graded.

        `val_score` is still measured on the whole held-back half, so a filtered
        experiment stays directly comparable to an unfiltered one. `cv_score` does
        not: it becomes the mean over the kept rows only.

        Args:
            code: Python defining `keep_rows(df)`, returning a **boolean Series**
                — one value per row, True to keep, indexed like `df`. Unlike
                `transform_features`, `df` here **does** include the target
                column, and the code may use whole-column statistics: it is
                evaluated once and never replayed. Example:

                    def keep_rows(df):
                        return (df["is_active"] == 1) & (df["amount"] > 0)

            table: Path to the source table. Omit to use the run's work table.
        """
        return _retryable(lambda: tools.filter_rows(ctx.deps.session, code, table=table), ctx)

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
        return _retryable(lambda: tools.transform_features(ctx.deps.session, code, table=table), ctx)

    @agent.tool
    def fit_reference_target(
        ctx: RunContext[Deps],
        table: str | None = None,
        baseline_column: str | None = None,
        scale_column: str | None = None,
    ) -> dict:
        """Measure a residual, ratio, or scaled-residual target with the fixed probe.

        This changes what the model learns, not the feature rows it receives:
        it fits on ``(target - baseline_column) / scale_column``, with either
        column optional, and automatically returns predictions to original target
        units before every score. The model, parameters, folds and target_transform
        stay fixed, so the result isolates the target definition.

        Both columns must already be numeric and serving-available on ``table``.
        A revealed-target lag is legal only when it was produced by
        ``fit_target_history``; this tool consumes that guarded column
        but never reads hidden targets itself. A scale containing zero, missing or
        non-finite values is refused because the inverse would be undefined.

        Args:
            table: Feature table to fit. Omit for the work table.
            baseline_column: Optional serving-known reference to subtract, such
                as a revealed target lag.
            scale_column: Optional serving-known non-zero scale to divide by,
                such as installed capacity.
        """
        return _retryable(
            lambda: tools.fit_reference_target(
                ctx.deps.session,
                table=table,
                baseline_column=baseline_column,
                scale_column=scale_column,
            ),
            ctx,
        )

    @agent.tool
    def explain_table(
        ctx: RunContext[Deps],
        table: str | None = None,
        model_type: str = "lgbm",
        params: dict | None = None,
        target_transform: str = "none",
        baseline_column: str | None = None,
        scale_column: str | None = None,
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
            model_type: One of "baseline", "elasticnet", "lgbm", "xgb",
                "catboost". "elasticnet" is a regularised linear model that
                tunes its own penalty; it is worth one experiment on any table
                you care about, because a tree model that cannot clearly beat
                it is telling you the signal is in the features rather than the
                model.
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
                reference_target={
                    key: value
                    for key, value in {
                        "baseline_column": baseline_column,
                        "scale_column": scale_column,
                    }.items()
                    if value is not None
                } or None,
            ),
            ctx,
        )

    @agent.tool
    def explore_table(
        ctx: RunContext[Deps],
        table: str | None = None,
        code: str = "",
    ) -> dict:
        """Look at where the ruler is wrong on a table, by running your own code on its rows.

        Answers "which rows is this failing on", which is the question a score
        cannot answer and the fastest route to the next feature worth building.
        Nothing computed here is kept or can become a feature — the result is
        shown to you and discarded — so group by the target freely.

        Records nothing and costs no budget, but it does refit across folds to
        get honest out-of-fold errors, so it is slower than building a table.

        Args:
            table: Path to the table to diagnose. Omit for the run's work table.
            code: Python defining `explore(df)`, returning a small summary — a
                number, a dict, or a DataFrame/Series (capped at 200 rows).
                Anything printed comes back too. `df` is the whole table —
                including the string columns the model could not use — plus:

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
            lambda: tools.explore_table(ctx.deps.session, table=table, code=code), ctx
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
                            vs_source_table, n_features, fit_seconds
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
        return _retryable(lambda: tools.explore_journal(ctx.deps.session, code), ctx)

    @agent.tool
    def finalize_run(ctx: RunContext[Deps]) -> dict:
        """Package the best experiment into a predictor bundle and write the report.

        Call this exactly once, when you are done experimenting.
        """
        return _retryable(lambda: tools.finalize_run(ctx.deps.session), ctx)

    return agent


#: Requests one agent run may make before PydanticAI stops it. Its default is 50,
#: which a rich dataset exhausts on feature experiments *before* it can finalize —
#: the run then raised `UsageLimitExceeded` and shipped nothing (Rossmann,
#: 2026-09-16, ~900k rows). Raised to a generous ceiling, and the ceiling is caught
#: below so hitting it finalizes whatever the search built rather than throwing —
#: the same "an interrupted search still ships" stance as the episode stop (0024).
REQUEST_LIMIT = 300


def run(session: RunSession, prompt: str, model=DEFAULT_MODEL):
    """Drive one agent run against ``session``.

    Returns the PydanticAI result, or ``None`` if the request ceiling was hit. The
    caller finalizes what the search built either way (see `Agent.run`), so a long
    search that runs out of requests degrades to "ship what we have" instead of
    crashing with no bundle.
    """
    from pydantic_ai.exceptions import UsageLimitExceeded
    from pydantic_ai.usage import UsageLimits

    agent = build_agent(
        model, profile=session.profile, workflow=getattr(session, "workflow", "general")
    )
    try:
        return agent.run_sync(
            prompt,
            deps=Deps(session=session),
            usage_limits=UsageLimits(request_limit=REQUEST_LIMIT),
        )
    except UsageLimitExceeded:
        return None
