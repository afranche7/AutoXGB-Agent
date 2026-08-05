"""The approval gate, driven through a real LangGraph with a scripted model.

The gate is the one interactive path in the product and the only place a wrong
answer fails silently rather than loudly, so it is worth exercising for real:
these tests run an actual agent graph, hit the interrupt, and resume it — with a
fake model so nothing is billed. They also pin the CLI's parsing of the interrupt
payload, which is the part most likely to drift when LangChain changes shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from autoxgb_agent.cli import _action_requests, _decide_one
from autoxgb_agent.subagents import APPROVAL_TOOL
from autoxgb_agent.tools import set_task_plan
from tests.conftest import ScriptedChatModel


GOOD_PLAN = {
    "target": "churned",
    "task_type": "binary_classification",
    "dropped_columns": ["customer_id", "internal_note", "always_one"],
    "reasoning": "churned is the binary outcome the goal names.",
}
WRONG_PLAN = {**GOOD_PLAN, "target": "region", "task_type": "multiclass_classification"}


def _call(plan: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": APPROVAL_TOOL, "args": {"plan": plan}, "id": call_id}],
    )


def _build(responses: list[AIMessage]):
    model = ScriptedChatModel(responses=responses)
    return create_agent(
        model=model,
        tools=[set_task_plan],
        middleware=[HumanInTheLoopMiddleware(interrupt_on={APPROVAL_TOOL: True})],
        checkpointer=InMemorySaver(),
    )


def _drive(graph, payload: Any, config: dict[str, Any]) -> list[Any]:
    """Run until the graph interrupts or finishes; return pending interrupts."""
    for chunk in graph.stream(payload, config, stream_mode="updates"):
        if isinstance(chunk, dict) and "__interrupt__" in chunk:
            return list(chunk["__interrupt__"])
    return []


@pytest.fixture
def config(tmp_path: Path) -> dict[str, Any]:
    return {"configurable": {"thread_id": tmp_path.name}, "recursion_limit": 50}


def test_plan_is_not_recorded_until_the_user_approves(
    make_run, classification_csv, config
):
    ctx = make_run(classification_csv, goal="predict churn")
    graph = _build([_call(GOOD_PLAN, "c1"), AIMessage(content="Plan recorded.")])
    start = {"messages": [{"role": "user", "content": "decide the target"}]}

    pending = _drive(graph, start, config)
    assert pending, "the graph should have paused before writing the plan"
    assert not (ctx.run_dir / "task_plan.json").exists(), (
        "the tool must not run before approval"
    )

    actions = _action_requests(pending[0])
    assert [a["name"] for a in actions] == [APPROVAL_TOOL]
    assert actions[0]["args"]["plan"]["target"] == "churned"

    assert not _drive(graph, Command(resume={"decisions": [{"type": "approve"}]}), config)
    written = json.loads((ctx.run_dir / "task_plan.json").read_text())
    assert written["target"] == "churned"
    assert written["task_type"] == "binary_classification"


def test_rejection_feeds_the_reason_back_and_lets_the_agent_retry(
    make_run, classification_csv, config
):
    ctx = make_run(classification_csv, goal="predict churn")
    graph = _build(
        [
            _call(WRONG_PLAN, "c1"),
            _call(GOOD_PLAN, "c2"),
            AIMessage(content="Corrected and recorded."),
        ]
    )
    start = {"messages": [{"role": "user", "content": "decide the target"}]}

    assert _drive(graph, start, config)
    rejection = Command(
        resume={
            "decisions": [
                {"type": "reject", "message": "Wrong target — the goal is churn."}
            ]
        }
    )
    pending = _drive(graph, rejection, config)
    assert not (ctx.run_dir / "task_plan.json").exists(), "rejected plan must not run"

    # The model sees the rejection and proposes again; approving that one writes it.
    assert pending, "the second proposal should also pause for approval"
    assert not _drive(graph, Command(resume={"decisions": [{"type": "approve"}]}), config)
    assert json.loads((ctx.run_dir / "task_plan.json").read_text())["target"] == "churned"

    # The reason reached the model rather than being swallowed.
    history = graph.get_state(config).values["messages"]
    assert any("Wrong target" in str(getattr(m, "content", "")) for m in history)


def test_disarming_the_gate_runs_straight_through(make_run, classification_csv, config):
    """`--yes` builds the graph without the interrupt middleware."""
    ctx = make_run(classification_csv, goal="predict churn")
    model = ScriptedChatModel(responses=[_call(GOOD_PLAN, "c1"), AIMessage(content="Done.")])
    graph = create_agent(model=model, tools=[set_task_plan], checkpointer=InMemorySaver())

    assert not _drive(graph, {"messages": [{"role": "user", "content": "go"}]}, config)
    assert (ctx.run_dir / "task_plan.json").exists()


# --------------------------------------------------------------------------- #
# CLI prompt handling
# --------------------------------------------------------------------------- #


def test_cli_approve_and_reject_map_to_the_right_decisions(monkeypatch):
    action = {"name": APPROVAL_TOOL, "args": {"plan": GOOD_PLAN}}

    monkeypatch.setattr("typer.prompt", lambda *a, **k: "y")
    assert _decide_one(action) == {"type": "approve"}

    answers = iter(["n", "use `renewed` as the target"])
    monkeypatch.setattr("typer.prompt", lambda *a, **k: next(answers))
    assert _decide_one(action) == {
        "type": "reject",
        "message": "use `renewed` as the target",
    }


def test_cli_rejection_without_feedback_still_tells_the_model_something(monkeypatch):
    answers = iter(["no", "   "])
    monkeypatch.setattr("typer.prompt", lambda *a, **k: next(answers))
    decision = _decide_one({"name": APPROVAL_TOOL, "args": {"plan": GOOD_PLAN}})
    assert decision["type"] == "reject"
    assert decision["message"], "an empty reason would leave the agent no way forward"
