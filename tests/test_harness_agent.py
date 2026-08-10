"""The constrained-surface guarantee, asserted rather than assumed.

"Leakage-safe by construction" only holds if the agent's tool set is exactly the
leakage-safe tools. This file fails loudly if a tool is ever added without a
deliberate decision — a shell, a file reader, or a raw-data getter would each
quietly reopen the escape hatch the whole design exists to close.
"""

import pytest

pytest.importorskip("pydantic_ai")

from tabular_agent.harness.agent import DEFAULT_MODEL, build_agent  # noqa: E402

EXPECTED_TOOLS = {
    "inspect_data",
    "fit_encoding",
    "transform_features",
    "train_tabular_model",
    # Refits on columns the agent already has, and returns scores. It cannot
    # reach anything `train_tabular_model` cannot.
    "explain_table",
    # Runs agent-written code, but read-only: the frame it builds comes from an
    # experiment that already exists, and its result has no persistence path, so
    # it cannot produce a feature or reach a row training could not.
    "explore_experiment",
    # Sees the run's records only — no rows at all, so nothing to gate.
    "explore_journal",
    "finalize_run",
}


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


def test_the_prompt_does_not_describe_the_reader_to_itself():
    """Second person throughout: naming the reader in the third person is wasted words."""
    from tabular_agent.harness.instructions import SYSTEM_PROMPT

    assert "agent" not in SYSTEM_PROMPT.lower()


def test_the_prompt_introduces_no_vocabulary_of_its_own():
    """Every noun in it must already be one the tools force.

    Words like "block", "parent" and "branch" have each been introduced here at
    some point and each had to be removed again: they name something that already
    has a name, and the reader is left holding two words for one thing. The tools
    speak in tables, columns and `source_table`; so does this.
    """
    from tabular_agent.harness.instructions import SYSTEM_PROMPT

    prompt = SYSTEM_PROMPT.lower()
    for word in ("block", "parent", "branch", "lineage", "node", "step in the chain"):
        assert word not in prompt, f"{word!r} is vocabulary the tools do not use"


def test_the_prompt_does_not_repeat_what_a_tool_docstring_already_says():
    """Per-tool facts ride along with the tool; the prompt is for what spans tools."""
    from tabular_agent.harness.instructions import SYSTEM_PROMPT

    agent = build_agent()
    prompt = " ".join(SYSTEM_PROMPT.lower().split())
    for name, tool in agent._function_toolset.tools.items():
        for sentence in (tool.function_schema.description or "").split("."):
            words = " ".join(sentence.lower().split())
            if len(words.split()) < 8:
                continue  # too short to be a meaningful duplicate
            assert words not in prompt, f"{name}'s docstring is repeated in the prompt"


def test_default_model_is_swappable():
    # Nothing about the harness is provider-specific; the model is one string.
    assert DEFAULT_MODEL.startswith("anthropic:")
    agent = build_agent("openai:gpt-4o")
    assert set(agent._function_toolset.tools) == EXPECTED_TOOLS


def test_key_file_is_found_from_a_subdirectory(tmp_path, monkeypatch):
    """Notebooks run with CWD set to their own folder, not the repo root."""
    from tabular_agent.harness.agent import load_api_key

    (tmp_path / "config.env").write_text("# comment\nREQUESTY_API_KEY=abc123\n")
    sub = tmp_path / "notebooks"
    sub.mkdir()
    monkeypatch.chdir(sub)
    monkeypatch.delenv("REQUESTY_API_KEY", raising=False)

    assert load_api_key() == "abc123"


def test_env_var_wins_over_the_file(tmp_path, monkeypatch):
    from tabular_agent.harness.agent import load_api_key

    (tmp_path / "config.env").write_text("REQUESTY_API_KEY=from_file\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REQUESTY_API_KEY", "from_env")

    assert load_api_key() == "from_env"


def test_missing_key_says_what_to_do(tmp_path, monkeypatch):
    from tabular_agent.harness.agent import load_api_key

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("REQUESTY_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="no API key"):
        load_api_key(key_path="definitely_absent.env")


def test_a_bad_argument_is_retryable_not_fatal(tmp_path):
    """One malformed argument must not throw away a run's completed experiments."""
    import pandas as pd
    from pydantic_ai import ModelRetry

    from tabular_agent.harness.agent import _retryable
    from tabular_agent.harness.session import RunSession
    from tabular_agent.harness.tools import fit_encoding, inspect_data
    from tabular_agent.taskspec import TaskSpec

    path = str(tmp_path / "d.csv")
    pd.DataFrame({"size": ["small", "large"] * 20, "y": [0, 1] * 20}).to_csv(path, index=False)
    s = RunSession(str(tmp_path / "run"), path, TaskSpec(target="y", n_splits=2))
    inspect_data(s)

    with pytest.raises(ModelRetry, match="no level order given"):
        _retryable(lambda: fit_encoding(s, ["size"], "ordinal", categories={"size": []}))


def test_code_the_model_wrote_badly_is_retryable_too(tmp_path):
    """A sandbox rejection is an argument problem — the argument is a program."""
    import pandas as pd
    from pydantic_ai import ModelRetry

    from tabular_agent.harness.agent import _retryable
    from tabular_agent.harness.session import RunSession
    from tabular_agent.harness.tools import inspect_data, transform_features
    from tabular_agent.taskspec import TaskSpec

    path = str(tmp_path / "d.csv")
    pd.DataFrame({"a": [1.0, 2.0] * 20, "y": [0, 1] * 20}).to_csv(path, index=False)
    s = RunSession(str(tmp_path / "run"), path, TaskSpec(target="y", n_splits=2))
    inspect_data(s)

    code = "import sklearn\ndef transform_features(df):\n    return df\n"
    with pytest.raises(ModelRetry, match="fit_encoding"):
        _retryable(lambda: transform_features(s, code))


def test_tool_calls_are_serialised(tmp_path):
    """Parallel calls would race on the unlocked next_table_id/next_experiment_id."""
    agent = build_agent()
    assert agent._concurrency_limiter.max_running == 1


def test_a_real_bug_still_stops_the_run():
    from tabular_agent.harness.agent import _retryable

    def boom():
        raise KeyError("this is a bug, not a bad argument")

    with pytest.raises(KeyError):
        _retryable(boom)
