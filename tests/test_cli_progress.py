"""The CLI's progress surfaces, driven through the real commands.

`autoxgb run` is exercised end to end with a scripted model: the recorder is
opened, the stream is tracked, the tools report themselves and the log is closed
even when the run falls over. `autoxgb watch` and `autoxgb runs` are then pointed
at what it wrote.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from typer.testing import CliRunner

from autoxgb_agent import cli
from autoxgb_agent.orchestrator import build_agent
from autoxgb_agent.progress import (
    DONE,
    EVENTS_FILE,
    FAILED,
    read_approval_request,
    read_progress,
    read_state_file,
    write_approval_response,
)
from tests.conftest import ScriptedChatModel

runner = CliRunner()


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-used-in-these-tests")


def _script() -> list[AIMessage]:
    """Orchestrator plans, delegates once, and the profiler does real work."""
    return [
        AIMessage(
            content="Planning the pipeline.",
            tool_calls=[
                {
                    "name": "write_todos",
                    "args": {"todos": [{"content": "profile the data", "status": "in_progress"}]},
                    "id": "w1",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "task",
                    "args": {"description": "Profile it.", "subagent_type": "data-profiler"},
                    "id": "t1",
                }
            ],
        ),
        AIMessage(content="", tool_calls=[{"name": "profile_dataset", "args": {}, "id": "t2"}]),
        AIMessage(content="Profiled. churned looks like the target."),
        AIMessage(content="Profiling complete."),
    ]


@pytest.fixture
def scripted(monkeypatch):
    """Swap the real model for a scripted one, leaving the graph itself real."""

    def build(responses: list[AIMessage] | None = None):
        model = ScriptedChatModel(responses=responses or _script())

        def fake_build_agent(run_dir: Path, **kwargs: Any):
            return build_agent(
                run_dir,
                model=model,
                require_approval=kwargs.get("require_approval", True),
            )

        monkeypatch.setattr(cli, "build_agent", fake_build_agent)
        return model

    return build


def _invoke(*args: str):
    return runner.invoke(cli.app, list(args), catch_exceptions=False)


# --------------------------------------------------------------------------- #
# `autoxgb run`
# --------------------------------------------------------------------------- #


def test_a_run_records_its_stages_tools_and_artifacts(scripted, tmp_path, classification_csv):
    scripted()
    result = _invoke(
        "run",
        str(classification_csv),
        "--goal",
        "predict churn",
        "-o",
        str(tmp_path),
        "--run-id",
        "r1",
        "--yes",
        "--plain",
        "--quiet",
    )
    assert result.exit_code == 0, result.output

    run_dir = tmp_path / "r1"
    assert (run_dir / EVENTS_FILE).exists()

    state = read_progress(run_dir)
    assert state.run_id == "r1"
    assert state.goal == "predict churn"
    assert state.status == DONE
    assert state.stages["data-profiler"].status == DONE
    assert [call.tool for call in state.stages["data-profiler"].tool_calls] == [
        "profile_dataset"
    ]
    assert state.n_tool_calls == 1
    assert state.n_errors == 0

    # The plan and the files the stage produced both made it into the log.
    assert [todo["content"] for todo in state.todos] == ["profile the data"]
    assert "profile_report.md" in state.artifacts
    assert "column_stats.json" in state.artifacts

    # Stages nobody reached stay honestly untouched.
    assert state.stages["packager"].status == "pending"
    assert state.completed_stages() == 1


def test_the_run_says_where_to_watch_it_and_reports_progress_without_a_terminal(
    scripted, tmp_path, classification_csv
):
    scripted()
    result = _invoke(
        "run",
        str(classification_csv),
        "--goal",
        "predict churn",
        "-o",
        str(tmp_path),
        "--run-id",
        "r2",
        "--yes",
        "--plain",
    )
    assert "autoxgb watch r2" in result.output
    assert "progress: 1/7 stages" in result.output, (
        "a non-terminal run still needs progress lines"
    )


def test_a_run_that_blows_up_still_closes_its_log(tmp_path, classification_csv, monkeypatch):
    def explode(*args: Any, **kwargs: Any):
        raise RuntimeError("the model provider fell over")

    monkeypatch.setattr(cli, "build_agent", explode)
    result = runner.invoke(
        cli.app,
        [
            "run",
            str(classification_csv),
            "--goal",
            "predict churn",
            "-o",
            str(tmp_path),
            "--run-id",
            "r3",
            "--yes",
            "--plain",
        ],
    )
    assert result.exit_code != 0

    state = read_progress(tmp_path / "r3")
    assert state.status == FAILED
    assert "the model provider fell over" in state.error


def test_the_approval_prompt_refuses_to_run_where_nothing_can_answer_it(
    scripted, tmp_path, classification_csv
):
    scripted()
    result = _invoke(
        "run",
        str(classification_csv),
        "--goal",
        "predict churn",
        "-o",
        str(tmp_path),
        "--run-id",
        "r4",
    )
    assert result.exit_code == 2
    assert "--approve-via file" in result.output


def test_an_unknown_approval_mode_is_rejected(scripted, tmp_path, classification_csv):
    scripted()
    result = _invoke(
        "run",
        str(classification_csv),
        "--goal",
        "predict churn",
        "-o",
        str(tmp_path),
        "--approve-via",
        "telepathy",
    )
    assert result.exit_code == 2


# --------------------------------------------------------------------------- #
# The approval gate, answered from outside the process
# --------------------------------------------------------------------------- #


PLAN = {
    "target": "churned",
    "task_type": "binary_classification",
    "dropped_columns": ["customer_id", "internal_note", "always_one"],
    "reasoning": "churned is the binary outcome the goal names.",
}


def test_an_approval_published_to_the_run_directory_is_answered_and_the_run_continues(
    scripted, tmp_path, classification_csv
):
    """The handshake the UI uses: the run publishes, something else answers."""
    scripted(
        [
            # Orchestrator hands the decision to the specialist that owns it…
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {
                            "description": "Pick the target.",
                            "subagent_type": "target-detector",
                        },
                        "id": "t1",
                    }
                ],
            ),
            # …which proposes a plan. The gate fires here, inside the subagent.
            AIMessage(
                content="",
                tool_calls=[{"name": "set_task_plan", "args": {"plan": PLAN}, "id": "p1"}],
            ),
            AIMessage(content="Target is churned, binary classification."),
            AIMessage(content="Plan approved and recorded."),
        ]
    )
    run_dir = tmp_path / "r5"

    def approve_when_asked() -> None:
        deadline = time.time() + 30
        while time.time() < deadline:
            request = read_approval_request(run_dir)
            if request is not None:
                write_approval_response(run_dir, str(request["id"]), [{"type": "approve"}])
                return
            time.sleep(0.05)

    answerer = threading.Thread(target=approve_when_asked, daemon=True)
    answerer.start()
    result = _invoke(
        "run",
        str(classification_csv),
        "--goal",
        "predict churn",
        "-o",
        str(tmp_path),
        "--run-id",
        "r5",
        "--approve-via",
        "file",
        "--plain",
        "--quiet",
    )
    answerer.join(timeout=5)

    assert result.exit_code == 0, result.output
    assert json.loads((run_dir / "task_plan.json").read_text())["target"] == "churned"

    state = read_progress(run_dir)
    assert state.approval is None, "the gate should be cleared once answered"
    assert any(
        event["type"] == "approval_resolved" for event in _events(run_dir)
    ), "the decision itself belongs in the log"
    # And the handshake files do not outlive the run.
    assert read_approval_request(run_dir) is None


# --------------------------------------------------------------------------- #
# `autoxgb watch` and `autoxgb runs`
# --------------------------------------------------------------------------- #


def test_watch_replays_a_finished_run(scripted, tmp_path, classification_csv):
    scripted()
    _invoke(
        "run",
        str(classification_csv),
        "--goal",
        "predict churn",
        "-o",
        str(tmp_path),
        "--run-id",
        "r6",
        "--yes",
        "--plain",
        "--quiet",
    )

    result = _invoke("watch", "r6", "-o", str(tmp_path), "--no-follow")
    assert result.exit_code == 0
    assert "data-profiler" in result.output
    assert "profile_dataset" in result.output
    assert "wrote profile_report.md" in result.output


def test_watch_defaults_to_the_most_recent_run(scripted, tmp_path, classification_csv):
    scripted()
    for run_id in ("older", "newer"):
        scripted()
        _invoke(
            "run",
            str(classification_csv),
            "--goal",
            f"predict churn ({run_id})",
            "-o",
            str(tmp_path),
            "--run-id",
            run_id,
            "--yes",
            "--plain",
            "--quiet",
        )

    result = _invoke("watch", "-o", str(tmp_path), "--no-follow")
    assert "newer" in result.output


def test_watch_can_answer_the_approval_a_run_is_blocked_on(tmp_path):
    """The other way to answer a `--approve-via file` run: a second terminal."""
    from autoxgb_agent.progress import ProgressRecorder, write_approval_request

    recorder = ProgressRecorder(tmp_path / "blocked")
    recorder.run_started(run_id="blocked", goal="predict churn", dataset="d", model="m")
    recorder.stage_started("target-detector", "Pick the target.")
    actions = [{"name": "set_task_plan", "args": {"plan": PLAN}}]
    request_id = recorder.approval_requested(actions)
    write_approval_request(recorder.run_dir, request_id, actions)

    result = runner.invoke(
        cli.app,
        ["watch", "blocked", "-o", str(tmp_path), "--approve", "--no-follow"],
        input="n\nuse `renewed` as the target\n",
    )
    assert result.exit_code == 0, result.output
    assert "Approval needed" in result.output

    from autoxgb_agent.progress import read_approval_response

    assert read_approval_response(recorder.run_dir, request_id) == [
        {"type": "reject", "message": "use `renewed` as the target"}
    ]


def test_watch_says_so_when_there_is_nothing_to_watch(tmp_path):
    (tmp_path / "empty").mkdir()
    assert _invoke("watch", "-o", str(tmp_path), "--no-follow").exit_code == 2
    assert _invoke("watch", "nope", "-o", str(tmp_path)).exit_code == 2


def test_the_runs_listing_shows_how_far_each_run_got(scripted, tmp_path, classification_csv):
    scripted()
    _invoke(
        "run",
        str(classification_csv),
        "--goal",
        "predict churn",
        "-o",
        str(tmp_path),
        "--run-id",
        "r7",
        "--yes",
        "--plain",
        "--quiet",
    )

    result = _invoke("runs", "-o", str(tmp_path))
    assert result.exit_code == 0
    assert "r7" in result.output
    assert "1/7" in result.output
    assert read_state_file(tmp_path / "r7")["status"] == DONE


def _events(run_dir: Path) -> list[dict[str, Any]]:
    from autoxgb_agent.progress import read_events

    return read_events(run_dir)
