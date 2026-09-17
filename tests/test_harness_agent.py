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
    # Fits a per-group statistic of a non-target column out of fold and stores
    # it. Cannot see the target (it is refused as a `value`), and the stored
    # mapping is a lookup, so it cannot become batch-dependent at predict.
    "fit_group_stats",
    # Stores each group's recent past and replays it as a value plus its age.
    # May point at the target, where the others may not — a lag of the target can
    # only come from stored history, never from a served frame. History is frozen
    # per fold at the end of the training prefix, so a validation row reads what a
    # served row the same distance ahead would; and it is refused outright on
    # unordered runs, where that freeze would mean nothing.
    "fit_history",
    # Aggregates a declared second input per row, counting only the records
    # observed a stated lead before that row. The one tool that reads a file the
    # agent cannot otherwise reach — and it never returns rows from it, only
    # aggregates, so the stream stays outside the agent's read surface. Safe for a
    # different reason than `fit_history`: the restriction is backward from each
    # row individually, so no fold freeze is needed and none would help. Refused
    # on unordered runs, and refused below the declared forecast horizon, where
    # the column would be built from knowledge no scored row could have.
    "fit_stream_aggregate",
    "transform_features",
    # Runs agent-written code that *can* see the target, which is only safe
    # because nothing it produces is replayed: it selects work-half rows to fit
    # on, and the validation half never passes through it.
    "filter_rows",
    # The fixed probe may measure a different, exactly invertible target
    # definition. It exposes only serving-known column names, never target code,
    # and does not choose a model.
    "fit_reference_target",
    # No model-selection tool: since 0035 the harness fixes the probe and
    # enumerates model families at finalize.
    # Refits on columns the agent already has, and returns scores. It cannot
    # reach anything building a table cannot.
    "explain_table",
    # Runs agent-written code, but read-only: the frame it builds comes from a
    # table that already exists, and its result has no persistence path, so it
    # cannot produce a feature or reach a row training could not.
    "explore_table",
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


def test_the_run_state_reaches_the_model_only_once_there_is_a_run(tmp_path):
    """The brief is wired as a *dynamic* system prompt, so it is rendered when the
    episode starts rather than baked in when the agent is built.

    Driven through `TestModel` because the wiring is the thing under test, not the
    model: what matters is that a fresh agent on an existing `run_dir` is handed
    the position, and that a run which has done nothing is handed nothing.
    """
    import numpy as np
    import pandas as pd
    from pydantic_ai.models.test import TestModel

    from tabular_agent.harness.agent import Deps
    from tabular_agent.harness.session import RunSession
    from tabular_agent.harness.tools import fit_encoding, inspect_data
    from tabular_agent.taskspec import TaskSpec

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"area": rng.uniform(30, 160, 200), "rooms": rng.integers(1, 6, 200)})
    df["price"] = df["area"] * 1000 + rng.normal(0, 5000, 200)
    data = tmp_path / "d.csv"
    df.to_csv(data, index=False)
    session = RunSession(str(tmp_path / "run"), str(data), TaskSpec(target="price"))

    def system_text(sess):
        agent = build_agent(TestModel(call_tools=[]))
        result = agent.run_sync("go", deps=Deps(session=sess))
        return "\n".join(
            part.content
            for message in result.all_messages()
            for part in message.parts
            if part.__class__.__name__ == "SystemPromptPart"
        )

    assert "RUN STATE" not in system_text(session)

    inspect_data(session)
    fit_encoding(session, ["rooms"], "frequency")

    # A *fresh agent* on the same run dir: the episode boundary, in one call.
    resumed = RunSession(session.run_dir, session.source_path, TaskSpec(target="price"))
    text = system_text(resumed)
    assert "RUN STATE" in text and "rmsle" in text and "Best table so far" in text


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


def test_concurrent_id_claims_never_collide(tmp_path):
    """The invariant the old test *named* and did not check.

    It asserted `agent._concurrency_limiter.max_running == 1` — that a constructor
    argument had been passed. The argument limits concurrent agent *runs*, not
    concurrent tool calls, so the property in the test's name was false the whole
    time it was green. Rossmann v3 paid for it: four duplicate table ids, a
    parquet corrupted mid-write, and a dead run.

    So this exercises the claim directly, from threads, and asserts what actually
    has to hold — no two callers get the same id, whoever runs when.
    """
    import threading

    from tabular_agent.harness import journal

    run_dir = str(tmp_path / "run")
    journal.load(run_dir)  # materialise an empty journal

    for claim, prefix in ((journal.next_table_id, "t"), (journal.next_experiment_id, "e")):
        got: list[str] = []
        lock = threading.Lock()

        def worker():
            claimed = claim(run_dir)
            with lock:
                got.append(claimed)

        threads = [threading.Thread(target=worker) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(got)) == len(got), f"{prefix} ids collided: {sorted(got)}"
        assert all(g.startswith(prefix) for g in got)


def test_concurrent_journal_mutations_are_one_transaction(tmp_path):
    """Concurrent writers preserve every fact and always leave valid JSON.

    Atomic replace alone only protects readers. Without a lock around load plus
    mutation plus replace, writers can share the temp path or overwrite facts
    another writer appended after their stale read.
    """
    from concurrent.futures import ThreadPoolExecutor
    import json

    from tabular_agent.harness import journal

    run_dir = str(tmp_path / "run")
    notes = [{"worker": i, "fact": f"note-{i}"} for i in range(64)]
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(journal.append_note, run_dir, note) for note in notes]
        for future in futures:
            future.result()

    state = journal.load(run_dir)
    assert {row["worker"] for row in state["log"]} == set(range(64))
    with open(journal.journal_path(run_dir)) as handle:
        assert json.load(handle) == state
    assert not list((tmp_path / "run").glob(".run-journal.*.tmp"))


def test_a_table_write_is_all_or_nothing(tmp_path):
    """A reader must see the old file or the new one, never a half-written one.

    `journal.save` became atomic in August; the parquet files it *describes* did
    not, and that is what corrupted a run — `OSError: Corrupt snappy compressed
    data` on the next read.
    """
    import pandas as pd

    from tabular_agent.io import load_table, save_parquet

    path = str(tmp_path / "t1.parquet")
    save_parquet(pd.DataFrame({"a": range(50)}), path)
    save_parquet(pd.DataFrame({"a": range(50, 100)}), path)

    assert load_table(path)["a"].tolist() == list(range(50, 100))
    # No temporary files left behind to be mistaken for tables.
    assert [p.name for p in tmp_path.iterdir()] == ["t1.parquet"]


def test_a_real_bug_still_stops_the_run():
    from tabular_agent.harness.agent import _retryable

    def boom():
        raise KeyError("this is a bug, not a bad argument")

    with pytest.raises(KeyError):
        _retryable(boom)


def test_run_returns_none_when_request_ceiling_hit(monkeypatch):
    """A long feature search that exhausts the request ceiling must not crash the
    run: run() catches UsageLimitExceeded and returns None so the caller finalizes
    whatever was built (Rossmann 2026-09-16)."""
    from pydantic_ai.exceptions import UsageLimitExceeded

    from tabular_agent.harness import agent as agent_mod

    class _Stub:
        def run_sync(self, *a, **k):
            raise UsageLimitExceeded("ceiling")

    class _Session:
        profile = "minimal"

    monkeypatch.setattr(agent_mod, "build_agent", lambda *a, **k: _Stub())
    assert agent_mod.run(_Session(), "build the best table") is None


def test_iid_and_forecast_workflows_expose_distinct_temporal_tools():
    iid_tools = set(build_agent(workflow="iid")._function_toolset.tools)
    forecast_tools = set(build_agent(workflow="forecast")._function_toolset.tools)

    assert iid_tools == EXPECTED_TOOLS - {"fit_history", "fit_stream_aggregate"}
    assert forecast_tools == EXPECTED_TOOLS


def test_forecast_contract_reaches_only_forecast_runs(tmp_path):
    import pandas as pd
    from pydantic_ai.models.test import TestModel

    from tabular_agent.harness.agent import Deps
    from tabular_agent.harness.session import RunSession
    from tabular_agent.taskspec import TaskSpec

    path = tmp_path / "d.csv"
    pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=40),
        "Store": [1, 2] * 20,
        "y": range(40),
    }).to_csv(path, index=False)

    def system_text(session, workflow):
        agent = build_agent(TestModel(call_tools=[]), workflow=workflow)
        result = agent.run_sync("go", deps=Deps(session=session))
        return "\n".join(
            part.content
            for message in result.all_messages()
            for part in message.parts
            if part.__class__.__name__ == "SystemPromptPart"
        )

    iid = RunSession(
        str(tmp_path / "iid"), str(path), TaskSpec(target="y"),
        workflow="iid",
    )
    forecast = RunSession(
        str(tmp_path / "forecast"), str(path),
        TaskSpec(target="y", forecast_horizon=7),
        split_on="Date", workflow="forecast",
    )

    assert "This is a forecast" not in system_text(iid, "iid")
    text = system_text(forecast, "forecast")
    assert "This is a forecast" in text
    assert "categorical interactions" in text
    assert "trailing-history family" in text
