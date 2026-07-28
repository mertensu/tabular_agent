"""The constrained-surface guarantee, asserted rather than assumed.

"Leakage-safe by construction" only holds if the agent's tool set is exactly the
leakage-safe tools. This file fails loudly if a tool is ever added without a
deliberate decision — a shell, a file reader, or a raw-data getter would each
quietly reopen the escape hatch the whole design exists to close.
"""

import pytest

pytest.importorskip("pydantic_ai")

from tabular_agent.harness.agent import DEFAULT_MODEL, build_agent  # noqa: E402

EXPECTED_TOOLS = {"inspect_data", "train_tabular_model", "finalize_run"}


def test_the_agent_has_exactly_the_leakage_safe_tools():
    agent = build_agent()
    assert set(agent._function_toolset.tools) == EXPECTED_TOOLS


def test_every_tool_returns_a_summary_not_a_table():
    """Slow context growth (§19.1) depends on this: dict out, never a DataFrame."""
    import inspect as _inspect

    from tabular_agent.harness import tools

    for name in EXPECTED_TOOLS:
        sig = _inspect.signature(getattr(tools, name))
        assert sig.return_annotation is dict, f"{name} must return a compact dict"


def test_default_model_is_swappable():
    # Nothing about the harness is provider-specific; the model is one string.
    assert DEFAULT_MODEL.startswith("anthropic:")
    agent = build_agent("openai:gpt-4o")
    assert set(agent._function_toolset.tools) == EXPECTED_TOOLS
