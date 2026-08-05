"""Assembles the DeepAgents graph."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from deepagents import FilesystemMiddleware, FilesystemPermission, create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware import ModelCallLimitMiddleware, TodoListMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver

from autoxgb_agent.subagents import APPROVAL_TOOL, build_subagents, load_prompt

DEFAULT_MODEL = "anthropic:claude-opus-5"

# The agent's filesystem is the run directory. `execute` (shell) and `delete` are
# withheld: this build does no code generation, so a shell is pure blast radius,
# and nothing in the pipeline needs to remove an artifact it already wrote.
FS_TOOLS = ["ls", "read_file", "write_file", "edit_file", "glob", "grep"]

# Belt and braces. `virtual_mode` already confines paths to the run directory;
# this stops a stray secret from being readable if someone points the run
# directory somewhere unexpected. Rules are first-match-wins, top to bottom, and
# paths are absolute within the agent's virtual filesystem.
PERMISSIONS: list[FilesystemPermission] = [
    FilesystemPermission(
        operations=["read", "write"],
        paths=["/**/.env*", "/.env*", "/**/*.pem", "/**/*.key", "/**/id_rsa*"],
        mode="deny",
    ),
    FilesystemPermission(operations=["read", "write"], paths=["/**"], mode="allow"),
]

# A full run is roughly 30-60 model calls across all agents. This is a runaway
# guard, not a budget — it should never bind on a healthy run.
DEFAULT_MODEL_CALL_LIMIT = 250


def resolve_model(model: str | BaseChatModel | None = None) -> str | BaseChatModel:
    """CLI flag, then AUTOXGB_MODEL, then the default.

    A `BaseChatModel` instance passes straight through, which is how the tests
    drive the graph without calling a real model.
    """
    return model or os.environ.get("AUTOXGB_MODEL") or DEFAULT_MODEL


def build_agent(
    run_dir: Path,
    *,
    model: str | BaseChatModel | None = None,
    require_approval: bool = True,
    model_call_limit: int = DEFAULT_MODEL_CALL_LIMIT,
) -> Any:
    """Build the orchestrator graph rooted at `run_dir`.

    Args:
        run_dir: Directory the agent's file tools see as their filesystem. All
            artifacts are written here.
        model: `provider:model` string. Defaults to `AUTOXGB_MODEL` or Claude Opus 5.
        require_approval: Pause for human approval before the task plan is
            recorded. Disable only for unattended runs.
        model_call_limit: Hard stop on model calls per thread.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    backend = FilesystemBackend(root_dir=str(run_dir))

    # Built once and shared: `create_deep_agent` replaces its default filesystem
    # middleware with this one by name, and each subagent needs its own copy in
    # `middleware` or it would fall back to the unrestricted default.
    #
    # `_permissions` must be passed explicitly. DeepAgents normally injects the
    # `permissions=` rules into the filesystem middleware it builds itself;
    # substituting our own by name bypasses that, so omitting this would silently
    # discard every rule below.
    filesystem = FilesystemMiddleware(
        backend=backend, tools=FS_TOOLS, _permissions=PERMISSIONS
    )

    subagents = build_subagents()
    for spec in subagents:
        spec["middleware"] = [filesystem]

    interrupt_on = {APPROVAL_TOOL: True} if require_approval else None

    return create_deep_agent(
        model=resolve_model(model),
        system_prompt=load_prompt("orchestrator"),
        subagents=subagents,
        backend=backend,
        permissions=PERMISSIONS,
        interrupt_on=interrupt_on,
        middleware=[
            TodoListMiddleware(),
            filesystem,
            ModelCallLimitMiddleware(thread_limit=model_call_limit, exit_behavior="end"),
        ],
        checkpointer=InMemorySaver(),
        name="autoxgb-orchestrator",
    )


def initial_message(dataset_path: Path, goal: str, run_id: str) -> str:
    """The kickoff turn. Everything the orchestrator needs to start planning."""
    return (
        f"Build an XGBoost pipeline for this dataset.\n\n"
        f"- Goal: {goal}\n"
        f"- Dataset: {dataset_path.name} (at {dataset_path})\n"
        f"- Run id: {run_id}\n\n"
        "The tools already know where the dataset is — you never pass file paths "
        "to them. Your file tools are rooted at the run directory, which is empty "
        "until the pipeline writes to it.\n\n"
        "Start by planning the pipeline with write_todos, then delegate to "
        "data-profiler."
    )
