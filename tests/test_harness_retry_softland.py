"""The retry ceiling must land softly, not end the run.

A model that keeps writing the same rejected `transform_features` code used to
crash the whole episode: PydanticAI counts retries per tool name and, once the
budget is spent, turns the next `ModelRetry` into an `UnexpectedModelBehavior`
that discards every experiment already completed and — the part that actually
cost us — denies the model the chance to call `finalize_run`. Both Rossmann
minimal-profile RCs hit exactly this and had to be force-finalized by the driver.

The fix keeps the correction budget unchanged and only changes what happens *at*
the ceiling: the otherwise-fatal attempt returns a normal tool result telling the
model to stop retrying, so the loop lives and the model finalizes on its own.
These tests pin both halves — the soft-landing unit, and the end-to-end loop that
used to crash now finishing by the agent's own `finalize_run`.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pydantic_ai")

from tabular_agent.harness import journal  # noqa: E402
from tabular_agent.harness.agent import Deps, TOOL_RETRIES, _retryable, build_agent  # noqa: E402
from tabular_agent.harness.session import RunSession  # noqa: E402
from tabular_agent.taskspec import TaskSpec  # noqa: E402


class _FakeCtx:
    """Just the two fields `_retryable` reads off a real `RunContext`."""

    def __init__(self, tool_name, retry, max_retries):
        self.tool_name = tool_name
        self.retry = retry
        self.max_retries = max_retries


def test_below_the_ceiling_still_raises_a_correction():
    """Attempts within budget are unchanged: a rejected argument is a ModelRetry."""
    from pydantic_ai import ModelRetry

    ctx = _FakeCtx("transform_features", retry=TOOL_RETRIES - 1, max_retries=TOOL_RETRIES)
    with pytest.raises(ModelRetry, match="bad column"):
        _retryable(lambda: (_ for _ in ()).throw(ValueError("bad column")), ctx)


def test_at_the_ceiling_lands_softly_instead_of_raising():
    """The attempt that would trip PydanticAI's ceiling returns a result instead."""
    ctx = _FakeCtx("transform_features", retry=TOOL_RETRIES, max_retries=TOOL_RETRIES)
    out = _retryable(lambda: (_ for _ in ()).throw(ValueError("bad column")), ctx)
    assert out["rejected"] is True
    assert out["attempts"] == TOOL_RETRIES + 1
    assert "bad column" in out["reason"]
    assert "finalize_run" in out["message"]


def test_without_a_ctx_the_ceiling_behaviour_is_unavailable():
    """Direct callers (tests, non-agent code) keep the raise-only contract."""
    from pydantic_ai import ModelRetry

    with pytest.raises(ModelRetry, match="bad column"):
        _retryable(lambda: (_ for _ in ()).throw(ValueError("bad column")))


def _session(tmp_path):
    rng = np.random.default_rng(0)
    n = 240
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "y": 3 * x + rng.normal(scale=0.2, size=n) + 10})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return RunSession(
        str(tmp_path / "run"), str(path), TaskSpec(target="y", n_splits=3),
        profile="minimal",
    )


# Two different tools, two different rejection causes — the cliff was generic
# (a live RC died on the 4th `transform_features` in one episode and the 4th
# `filter_rows` in the next), so the fix and its proof must be generic too. Each
# body always fails: a blocked import, then a NameError inside the sandbox.
_LOOPING_CASES = {
    "transform_features": "import sklearn\ndef transform_features(df):\n    return df\n",
    "filter_rows": "def keep_rows(df):\n    return data\n",
}


def _looping_model(tool_name, bad_code):
    """A scripted model that does what the deepseek runs did: loop on bad code.

    It reads only the *latest* tool results (the last request) and reacts:

    - nothing yet            -> inspect_data
    - inspect_data returned  -> `tool_name` with code that always fails
    - the tool was refused (a RetryPromptPart) -> try the same bad code again
    - the tool *returned*     -> only possible once it lands softly; finalize_run
    - finalize returned       -> stop with a final message

    On the unfixed harness the fourth refusal raises `UnexpectedModelBehavior`
    before any result for `tool_name` ever comes back, so `finalize_run` is never
    reached — which is the whole bug.
    """
    from pydantic_ai.messages import (
        ModelResponse,
        RetryPromptPart,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
    )

    stats = {"attempts": 0}

    def fn(messages, info):
        last = messages[-1]
        returns = {p.tool_name for p in last.parts if isinstance(p, ToolReturnPart)}
        retried = {p.tool_name for p in last.parts if isinstance(p, RetryPromptPart)}

        if "finalize_run" in returns:
            return ModelResponse(parts=[TextPart("done")])
        if tool_name in returns:
            # Only reachable once the ceiling lands softly with a real result.
            return ModelResponse(parts=[ToolCallPart("finalize_run", {})])
        if tool_name in retried or "inspect_data" in returns:
            stats["attempts"] += 1
            if stats["attempts"] > TOOL_RETRIES + 2:
                # A guard so a regression loops-then-fails instead of hanging.
                raise AssertionError(f"{tool_name} kept looping past the ceiling")
            return ModelResponse(parts=[ToolCallPart(tool_name, {"code": bad_code})])
        return ModelResponse(parts=[ToolCallPart("inspect_data", {})])

    return fn, stats


@pytest.mark.parametrize("tool_name", sorted(_LOOPING_CASES))
def test_a_looping_model_finalizes_itself_instead_of_crashing(tmp_path, tool_name):
    from pydantic_ai.models.function import FunctionModel

    session = _session(tmp_path)
    fn, stats = _looping_model(tool_name, _LOOPING_CASES[tool_name])
    agent = build_agent(FunctionModel(fn), profile="minimal")

    # No UnexpectedModelBehavior: the run completes and the agent's own
    # finalize_run is what ends it — whichever tool it got stuck on.
    agent.run_sync("build the best model you can", deps=Deps(session=session))

    # The ceiling was actually reached — the model was refused its whole budget
    # and then one more (the soft landing), so it stopped retrying the tool.
    assert stats["attempts"] == TOOL_RETRIES + 1

    j = journal.load(session.run_dir)
    assert j["stop"]["finalized"] is True
    trained = [r for r in j["leaderboard"] if r["model_type"] != "baseline"]
    assert len(trained) == 1
