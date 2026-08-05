"""Structural checks on the agent graph.

No model is ever called here — these assert the wiring that would otherwise only
fail halfway through a paid run: which tools each agent can reach, that the shell
is not among them, that permission rules survive our middleware substitution, and
that the approval gate is armed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from deepagents import FilesystemMiddleware

from autoxgb_agent.orchestrator import (
    DEFAULT_MODEL,
    FS_TOOLS,
    PERMISSIONS,
    build_agent,
    initial_message,
    resolve_model,
)
from autoxgb_agent.subagents import APPROVAL_TOOL, build_subagents, load_prompt
from autoxgb_agent.tools import ALL_TOOLS

# Constructing the graph instantiates a chat model, which wants a key present.
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-not-used-in-these-tests")

FORBIDDEN_TOOLS = {"execute", "delete"}


@pytest.fixture
def agent(tmp_path: Path):
    return build_agent(tmp_path / "run")


def _tool_names(graph) -> set[str]:
    return set(graph.nodes["tools"].bound._tools_by_name)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


def test_orchestrator_has_planning_and_delegation_but_no_shell(agent):
    names = _tool_names(agent)
    assert "write_todos" in names, "the planning tool is why we use DeepAgents"
    assert "task" in names, "the orchestrator must be able to delegate"
    assert FORBIDDEN_TOOLS.isdisjoint(names), f"shell/delete leaked into {names}"
    assert set(FS_TOOLS) <= names


def test_orchestrator_cannot_call_the_ml_tools_directly(agent):
    """ML work belongs to the specialists, so the orchestrator stays in context."""
    names = _tool_names(agent)
    assert {tool.name for tool in ALL_TOOLS}.isdisjoint(names)


def test_run_directory_is_created_eagerly(tmp_path: Path):
    run_dir = tmp_path / "nested" / "run"
    build_agent(run_dir)
    assert run_dir.is_dir()


# --------------------------------------------------------------------------- #
# Subagents
# --------------------------------------------------------------------------- #


def test_every_subagent_is_complete_and_scoped():
    subagents = build_subagents()
    assert len(subagents) == 7

    seen_tools: set[str] = set()
    for spec in subagents:
        assert spec["name"] and spec["description"] and spec["system_prompt"]
        tools = spec.get("tools") or []
        assert tools, f"{spec['name']} has no tools"
        names = {t.name for t in tools}
        assert FORBIDDEN_TOOLS.isdisjoint(names)
        # Tool ownership is exclusive: no two specialists share a tool, so the
        # model never has to guess which agent should do a given job.
        assert seen_tools.isdisjoint(names), f"{spec['name']} duplicates {seen_tools & names}"
        seen_tools |= names

    assert seen_tools == {tool.name for tool in ALL_TOOLS}


def test_only_the_target_detector_can_set_the_task_plan():
    owners = [
        spec["name"]
        for spec in build_subagents()
        if any(t.name == APPROVAL_TOOL for t in spec.get("tools") or [])
    ]
    assert owners == ["target-detector"]


def test_subagent_specs_carry_no_middleware_of_their_own():
    """`build_agent` owns middleware, so the specs stay reusable and inspectable."""
    assert all("middleware" not in spec for spec in build_subagents())


def test_build_agent_attaches_permissioned_filesystem_to_subagents(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    import autoxgb_agent.orchestrator as orch

    real = orch.create_deep_agent

    def spy(**kwargs):
        captured.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(orch, "create_deep_agent", spy)
    orch.build_agent(tmp_path / "run")

    for spec in captured["subagents"]:
        middleware = spec["middleware"]
        assert len(middleware) == 1
        fs = middleware[0]
        assert isinstance(fs, FilesystemMiddleware)
        # Substituting our own filesystem middleware bypasses the permission
        # injection DeepAgents would otherwise do, so it must carry the rules.
        assert getattr(fs, "_permissions", None) == PERMISSIONS


def test_approval_gate_is_armed_by_default(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    import autoxgb_agent.orchestrator as orch

    real = orch.create_deep_agent
    monkeypatch.setattr(
        orch, "create_deep_agent", lambda **kw: (captured.update(kw), real(**kw))[1]
    )

    orch.build_agent(tmp_path / "a")
    assert captured["interrupt_on"] == {APPROVAL_TOOL: True}
    assert captured["permissions"] == PERMISSIONS

    orch.build_agent(tmp_path / "b", require_approval=False)
    assert captured["interrupt_on"] is None


# --------------------------------------------------------------------------- #
# Prompts and model selection
# --------------------------------------------------------------------------- #


def test_every_prompt_file_exists_and_is_substantive():
    names = [
        "orchestrator",
        "profiler",
        "target_detector",
        "feature_engineer",
        "modeler",
        "tuner",
        "evaluator",
        "packager",
    ]
    for name in names:
        text = load_prompt(name)
        assert len(text) > 200, f"{name}.md looks like a stub"


def test_orchestrator_prompt_names_every_subagent():
    prompt = load_prompt("orchestrator")
    for spec in build_subagents():
        assert spec["name"] in prompt, f"{spec['name']} is unreachable from the plan"


def test_model_resolution_order(monkeypatch):
    monkeypatch.delenv("AUTOXGB_MODEL", raising=False)
    assert resolve_model() == DEFAULT_MODEL
    assert resolve_model("anthropic:claude-haiku-4-5") == "anthropic:claude-haiku-4-5"

    monkeypatch.setenv("AUTOXGB_MODEL", "anthropic:claude-sonnet-5")
    assert resolve_model() == "anthropic:claude-sonnet-5"
    assert resolve_model("anthropic:claude-opus-5") == "anthropic:claude-opus-5"


def test_kickoff_message_carries_the_run_details():
    message = initial_message(Path("/data/churn.csv"), "predict churn", "run-1")
    assert "predict churn" in message
    assert "churn.csv" in message
    assert "run-1" in message
    assert "data-profiler" in message
