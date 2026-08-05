"""Delegation through the real graph: orchestrator -> `task` -> subagent -> tool.

This is the assumption that would otherwise only be checked by a paid run —
DeepAgents composes each subagent's tool set from its spec plus the filesystem
middleware, and if that composition is wrong the specialists either have no ML
tools or no way to read what the previous stage wrote.

One scripted model instance backs every agent in the graph, so its responses are
consumed in call order: orchestrator, then subagent, then back out.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from autoxgb_agent.orchestrator import build_agent
from tests.conftest import ScriptedChatModel


os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-not-used-in-these-tests")


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def _delegate(subagent: str, instruction: str, call_id: str) -> AIMessage:
    return _tool_call(
        "task", {"description": instruction, "subagent_type": subagent}, call_id
    )


def _run(graph, config: dict[str, Any]) -> list[Any]:
    """Drive to completion, collecting every message the graph emits."""
    seen: list[Any] = []
    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": "build the pipeline"}]},
        config,
        stream_mode="updates",
    ):
        if not isinstance(chunk, dict):
            continue
        assert "__interrupt__" not in chunk, "no approval gate should fire in this flow"
        for update in chunk.values():
            if isinstance(update, dict):
                seen.extend(update.get("messages", []) or [])
    return seen


@pytest.fixture
def config(tmp_path: Path) -> dict[str, Any]:
    return {"configurable": {"thread_id": tmp_path.name}, "recursion_limit": 60}


def test_orchestrator_delegates_and_the_subagent_runs_its_own_tool(
    make_run, classification_csv, tmp_path, config
):
    ctx = make_run(classification_csv, goal="predict churn")

    model = ScriptedChatModel(
        responses=[
            # Orchestrator: hand the job to the profiler.
            _delegate("data-profiler", "Profile the dataset.", "t1"),
            # Profiler: call its own tool, then edit what it wrote. An edit only
            # succeeds if the subagent can both read and write that path, which
            # is the filesystem access DeepAgents supplies via middleware rather
            # than through the spec's `tools` list.
            _tool_call("profile_dataset", {}, "t2"),
            _tool_call(
                "edit_file",
                {
                    "file_path": "/profile_report.md",
                    "old_string": "# Data profile",
                    "new_string": "# Data profile (reviewed)",
                },
                "t3",
            ),
            AIMessage(content="Profiled. churned looks like the target."),
            # Orchestrator: wrap up.
            AIMessage(content="Profiling complete."),
        ]
    )

    graph = build_agent(tmp_path / "run", model=model, require_approval=False)
    messages = _run(graph, config)

    # The subagent's ML tool actually executed...
    assert (ctx.run_dir / "column_stats.json").exists(), (
        "the profiler could not reach profile_dataset"
    )
    # ...and its filesystem tools reached the run directory.
    assert "# Data profile (reviewed)" in (ctx.run_dir / "profile_report.md").read_text()

    # The orchestrator gets back the subagent's final report only — intermediate
    # tool output stays in the subagent's own context, which is the point.
    task_results = [
        str(getattr(m, "content", ""))
        for m in messages
        if getattr(m, "name", None) == "task"
    ]
    assert task_results, "the orchestrator never delegated"
    assert "churned looks like the target" in task_results[0]
    assert "Data profile" not in task_results[0], (
        "subagent tool output leaked into the orchestrator's context"
    )


def test_agent_file_tools_are_rooted_at_the_run_directory(
    make_run, classification_csv, tmp_path, config
):
    """The agent's filesystem is the run dir — the dataset itself is out of reach."""
    make_run(classification_csv, goal="predict churn")

    model = ScriptedChatModel(
        responses=[
            _tool_call("write_file", {"file_path": "/scratch.md", "content": "hello"}, "w1"),
            _tool_call("ls", {"path": "/"}, "l1"),
            AIMessage(content="Wrote a scratch file."),
        ]
    )
    run_dir = tmp_path / "run"
    graph = build_agent(run_dir, model=model, require_approval=False)
    messages = _run(graph, config)

    assert (run_dir / "scratch.md").read_text() == "hello"

    listing = next(
        str(getattr(m, "content", ""))
        for m in messages
        if getattr(m, "name", None) == "ls"
    )
    assert "scratch.md" in listing
    assert "churn.csv" not in listing, "the raw dataset must not be in the agent's FS"
