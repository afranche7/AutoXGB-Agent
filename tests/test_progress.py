"""Progress tracking: the event log, the reducer, and what feeds them.

The dashboard, `autoxgb watch` and the UI are all views over one event log, so
these tests work on the log rather than on any rendering: if the events are right
every surface is right.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from autoxgb_agent.console import StreamTracker
from autoxgb_agent.progress import (
    BLOCKED,
    DONE,
    EVENTS_FILE,
    FAILED,
    PENDING,
    RUNNING,
    STAGE_BY_TOOL,
    STAGE_KEYS,
    STAGES,
    ApprovalTimeout,
    ProgressReader,
    ProgressRecorder,
    apply_event,
    await_approval_response,
    format_duration,
    read_approval_request,
    read_progress,
    read_state_file,
    replay,
    reset_recorder,
    set_recorder,
    tool_progress,
    write_approval_request,
    write_approval_response,
)
from autoxgb_agent.subagents import build_subagents
from autoxgb_agent.tools import ALL_TOOLS, preview_data, profile_dataset, tune_xgboost


@pytest.fixture
def recorder(tmp_path: Path) -> ProgressRecorder:
    return ProgressRecorder(tmp_path / "run")


@pytest.fixture
def recording(recorder: ProgressRecorder):
    """Install `recorder` as the ambient one, the way the CLI does."""
    token = set_recorder(recorder)
    yield recorder
    reset_recorder(token)


# --------------------------------------------------------------------------- #
# The stage map is the contract between the tools and every progress view
# --------------------------------------------------------------------------- #


def test_stage_keys_match_the_real_subagents():
    """A renamed subagent must not silently stop reporting progress."""
    assert STAGE_KEYS == tuple(spec["name"] for spec in build_subagents())


def test_every_tool_maps_to_the_subagent_that_owns_it():
    for spec in build_subagents():
        for tool in spec["tools"]:
            assert STAGE_BY_TOOL.get(tool.name) == spec["name"], (
                f"{tool.name} is owned by {spec['name']} but maps elsewhere"
            )


def test_no_tool_is_left_untracked():
    assert {tool.name for tool in ALL_TOOLS} == set(STAGE_BY_TOOL)


# --------------------------------------------------------------------------- #
# Reducer
# --------------------------------------------------------------------------- #


def test_stage_lifecycle_moves_pending_to_running_to_done():
    state = replay(
        [
            {"ts": 100.0, "type": "run_started", "run_id": "r", "goal": "predict churn"},
            {"ts": 101.0, "type": "stage_started", "stage": "data-profiler"},
            {"ts": 102.0, "type": "tool_started", "stage": "data-profiler", "tool": "profile_dataset"},
            {
                "ts": 105.0,
                "type": "tool_finished",
                "stage": "data-profiler",
                "tool": "profile_dataset",
                "ok": True,
                "duration": 3.0,
                "summary": "Profiled 900 rows x 11 columns.",
            },
            {"ts": 106.0, "type": "stage_finished", "stage": "data-profiler", "ok": True},
        ]
    )

    profiler = state.stages["data-profiler"]
    assert profiler.status == DONE
    assert profiler.elapsed() == pytest.approx(5.0)
    assert profiler.tool_calls[0].duration == pytest.approx(3.0)
    assert profiler.tool_calls[0].ok is True
    assert state.completed_stages() == 1
    assert state.stages["modeler"].status == PENDING
    assert state.status == RUNNING


def test_a_tool_call_starts_its_stage_even_without_a_delegation_event():
    """The orchestrator can call a tool itself; the stage still has to light up."""
    state = replay(
        [{"ts": 1.0, "type": "tool_started", "stage": "modeler", "tool": "train_xgboost"}]
    )
    assert state.stages["modeler"].status == RUNNING
    assert state.current_stage().key == "modeler"


def test_a_long_tool_call_can_say_where_it_has_got_to():
    state = replay(
        [
            {"ts": 1.0, "type": "tool_started", "stage": "tuner", "tool": "tune_xgboost"},
            {
                "ts": 2.0,
                "type": "tool_progress",
                "stage": "tuner",
                "tool": "tune_xgboost",
                "message": "trial 7/25 — best roc_auc 0.8123",
                "fraction": 0.28,
            },
        ]
    )
    call = state.stages["tuner"].tool_calls[0]
    assert call.progress == "trial 7/25 — best roc_auc 0.8123"
    assert call.fraction == pytest.approx(0.28)
    assert call.ended_at is None, "reporting progress does not end the call"
    assert "trial 7/25" in state.stages["tuner"].detail


def test_a_failed_tool_is_counted_and_surfaced_on_its_stage():
    state = replay(
        [
            {"ts": 1.0, "type": "stage_started", "stage": "modeler"},
            {"ts": 2.0, "type": "tool_started", "stage": "modeler", "tool": "train_xgboost"},
            {
                "ts": 3.0,
                "type": "tool_finished",
                "stage": "modeler",
                "tool": "train_xgboost",
                "ok": False,
                "summary": "ERROR in train_xgboost: no processed data",
            },
        ]
    )
    assert state.n_errors == 1
    assert "no processed data" in state.stages["modeler"].error
    # A recoverable tool error is not a dead stage — the agent gets to retry.
    assert state.stages["modeler"].status == RUNNING


def test_approval_blocks_its_stage_and_resuming_unblocks_it():
    events = [
        {"ts": 1.0, "type": "stage_started", "stage": "target-detector"},
        {
            "ts": 2.0,
            "type": "approval_requested",
            "id": "abc",
            "actions": [{"name": "set_task_plan", "args": {"plan": {"target": "churned"}}}],
        },
    ]
    state = replay(events)
    assert state.stages["target-detector"].status == BLOCKED
    assert state.approval["id"] == "abc"

    apply_event(state, {"ts": 3.0, "type": "approval_resolved", "decision": "approve"})
    assert state.stages["target-detector"].status == RUNNING
    assert state.approval is None


def test_finishing_a_run_closes_whatever_was_still_open():
    state = replay(
        [
            {"ts": 1.0, "type": "run_started", "run_id": "r"},
            {"ts": 2.0, "type": "stage_started", "stage": "tuner"},
            {"ts": 9.0, "type": "run_finished", "status": FAILED, "error": "boom"},
        ]
    )
    assert state.status == FAILED
    assert state.stages["tuner"].status == FAILED
    assert state.error == "boom"
    assert state.is_finished
    assert state.elapsed() == pytest.approx(8.0)


def test_todos_replace_rather_than_accumulate():
    state = replay(
        [
            {"ts": 1.0, "type": "todos", "items": [{"content": "profile", "status": "pending"}]},
            {
                "ts": 2.0,
                "type": "todos",
                "items": [
                    {"content": "profile", "status": "completed"},
                    {"content": "train", "status": "in_progress"},
                ],
            },
        ]
    )
    assert [todo["status"] for todo in state.todos] == ["completed", "in_progress"]


# --------------------------------------------------------------------------- #
# Recorder
# --------------------------------------------------------------------------- #


def test_events_are_appended_as_jsonl_and_replay_to_the_same_state(recorder):
    recorder.run_started(run_id="r1", goal="predict churn", dataset="churn.csv", model="m")
    recorder.stage_started("data-profiler", "Profile it.")
    recorder.stage_finished("data-profiler", ok=True, summary="Done.")
    recorder.run_finished(status=DONE)

    lines = (recorder.run_dir / EVENTS_FILE).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["type"] for line in lines] == [
        "run_started",
        "stage_started",
        "stage_finished",
        "run_finished",
    ]

    replayed = read_progress(recorder.run_dir)
    assert replayed.run_id == "r1"
    assert replayed.goal == "predict churn"
    assert replayed.status == DONE
    assert replayed.stages["data-profiler"].status == DONE


def test_the_state_snapshot_tracks_the_log(recorder):
    recorder.run_started(run_id="r1", goal="g", dataset="d", model="m")
    recorder.stage_started("modeler")

    snapshot = read_state_file(recorder.run_dir)
    assert snapshot["run_id"] == "r1"
    assert snapshot["completed_stages"] == 0
    assert snapshot["total_stages"] == len(STAGES)
    assert [stage["key"] for stage in snapshot["stages"]] == list(STAGE_KEYS)


def test_artifacts_are_discovered_by_scanning_the_run_directory(recorder):
    (recorder.run_dir / "profile_report.md").write_text("# Data profile", encoding="utf-8")
    (recorder.run_dir / "bundle").mkdir()
    (recorder.run_dir / "bundle" / "predict.py").write_text("print()", encoding="utf-8")

    found = recorder.scan_artifacts(stage="data-profiler")
    assert set(found) == {"profile_report.md", "bundle/predict.py"}
    # Nothing changed, so a second scan reports nothing.
    assert recorder.scan_artifacts() == []

    assert recorder.state.artifacts["profile_report.md"]["stage"] == "data-profiler"
    # The progress log itself is never reported as an artifact of the run.
    assert not any(path.startswith("progress/") for path in recorder.state.artifacts)


def test_a_recorder_never_lets_a_write_failure_reach_the_run(recorder, monkeypatch):
    def explode(*args: Any, **kwargs: Any):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "open", explode)
    recorder.emit("note", text="still fine")  # must not raise
    assert recorder.state.n_events == 1


# --------------------------------------------------------------------------- #
# Tools report themselves
# --------------------------------------------------------------------------- #


def test_running_a_tool_records_its_stage_timing_and_artifacts(
    recording, make_run, classification_csv
):
    make_run(classification_csv, goal="predict churn")
    # The recorder and the run share a directory, as they do in a real run.
    profile_dataset.invoke({})

    kinds = [event["type"] for event in _events(recording.run_dir)]
    assert "tool_started" in kinds and "tool_finished" in kinds

    state = read_progress(recording.run_dir)
    profiler = state.stages["data-profiler"]
    assert profiler.status == RUNNING, "a tool call alone does not finish the stage"
    assert [call.tool for call in profiler.tool_calls] == ["profile_dataset"]
    assert profiler.tool_calls[0].ok is True
    assert profiler.tool_calls[0].duration >= 0.0
    assert "Profiled" in profiler.tool_calls[0].summary


def test_a_tool_that_fails_is_recorded_as_a_failure_not_a_crash(
    recording, make_run, tmp_path
):
    missing = tmp_path / "not_here.csv"
    missing.write_text("a,b\n1,2\n", encoding="utf-8")
    make_run(missing)
    missing.unlink()

    result = preview_data.invoke({"n_rows": 5})
    assert result.startswith("ERROR")

    state = read_progress(recording.run_dir)
    call = state.stages["data-profiler"].tool_calls[-1]
    assert call.ok is False
    assert "dataset not found" in call.summary
    assert state.n_errors == 1


def test_tools_run_fine_with_nobody_recording(make_run, classification_csv):
    """Progress is optional — the tools must not require a recorder."""
    make_run(classification_csv, goal="predict churn")
    assert "Profiled" in profile_dataset.invoke({})
    tool_progress("tune_xgboost", "trial 1/2")  # must be a no-op, not a crash


def test_the_tuner_reports_every_trial_while_it_searches(
    recording, make_run, classification_csv
):
    """The longest stage in the pipeline must not look like a silent minute."""
    from tests.test_pipeline import DROPPED, _run_through_preprocessing

    ctx = make_run(
        classification_csv,
        goal="predict churn",
        max_tuning_trials=2,
        max_tuning_timeout_seconds=120,
    )
    _run_through_preprocessing(ctx, "churned", "binary_classification", DROPPED)
    tune_xgboost.invoke({"n_trials": 2, "timeout_seconds": 120})

    messages = [
        event["message"]
        for event in _events(recording.run_dir)
        if event["type"] == "tool_progress"
    ]
    assert [m.split(" — ")[0] for m in messages] == ["trial 1/2", "trial 2/2"]
    assert "roc_auc" in messages[-1], "the score so far is the point of the message"

    # And the call still ends cleanly once the search is done.
    call = read_progress(recording.run_dir).stages["tuner"].tool_calls[-1]
    assert call.ok is True
    assert call.progress == messages[-1]


# --------------------------------------------------------------------------- #
# The orchestrator's stream becomes stage events
# --------------------------------------------------------------------------- #


def _delegate(subagent: str, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "task",
                "args": {"description": f"Run {subagent}.", "subagent_type": subagent},
                "id": call_id,
            }
        ],
    )


def test_delegation_in_the_stream_opens_and_closes_a_stage(recorder):
    tracker = StreamTracker(recorder)

    tracker.consume({"messages": [_delegate("data-profiler", "t1")]})
    assert recorder.state.stages["data-profiler"].status == RUNNING
    assert recorder.state.stages["data-profiler"].instruction == "Run data-profiler."

    tracker.consume(
        {
            "messages": [
                ToolMessage(content="Profiled. churned is the target.", name="task", tool_call_id="t1")
            ]
        }
    )
    stage = recorder.state.stages["data-profiler"]
    assert stage.status == DONE
    assert "churned is the target" in stage.detail


def test_two_stages_in_flight_are_closed_by_their_own_call_ids(recorder):
    tracker = StreamTracker(recorder)
    tracker.consume({"messages": [_delegate("modeler", "a"), _delegate("tuner", "b")]})
    tracker.consume(
        {"messages": [ToolMessage(content="tuned", name="task", tool_call_id="b")]}
    )

    assert recorder.state.stages["tuner"].status == DONE
    assert recorder.state.stages["modeler"].status == RUNNING


def test_the_todo_list_and_narration_reach_the_log(recorder):
    tracker = StreamTracker(recorder)
    tracker.consume(
        {
            "messages": [
                AIMessage(
                    content="Planning the pipeline.",
                    tool_calls=[
                        {
                            "name": "write_todos",
                            "args": {
                                "todos": [
                                    {"content": "profile the data", "status": "in_progress"},
                                    {"content": "train", "status": "pending"},
                                ]
                            },
                            "id": "w1",
                        }
                    ],
                )
            ]
        }
    )

    assert [todo["content"] for todo in recorder.state.todos] == [
        "profile the data",
        "train",
    ]
    assert any(
        entry["text"] == "Planning the pipeline." for entry in recorder.state.activity
    )


def test_a_failed_delegation_marks_the_stage_failed(recorder):
    tracker = StreamTracker(recorder)
    tracker.consume({"messages": [_delegate("packager", "p1")]})
    tracker.consume(
        {
            "messages": [
                ToolMessage(
                    content="no such subagent", name="task", tool_call_id="p1", status="error"
                )
            ]
        }
    )
    assert recorder.state.stages["packager"].status == FAILED


# --------------------------------------------------------------------------- #
# Watching from another process
# --------------------------------------------------------------------------- #


def test_a_reader_only_ever_sees_new_events(recorder):
    reader = ProgressReader(recorder.run_dir)
    recorder.run_started(run_id="r", goal="g", dataset="d", model="m")

    assert [event["type"] for event in reader.poll()] == ["run_started"]
    assert reader.poll() == []

    recorder.stage_started("evaluator")
    assert [event["type"] for event in reader.poll()] == ["stage_started"]
    assert reader.state.stages["evaluator"].status == RUNNING


def test_a_half_written_line_is_skipped_until_it_is_complete(recorder):
    recorder.run_started(run_id="r", goal="g", dataset="d", model="m")
    with (recorder.run_dir / EVENTS_FILE).open("a", encoding="utf-8") as handle:
        handle.write('{"ts": 1.0, "type": "note", "text": "partial')

    state = read_progress(recorder.run_dir)
    assert state.run_id == "r", "a torn final line must not break the whole log"


# --------------------------------------------------------------------------- #
# The out-of-band approval handshake
# --------------------------------------------------------------------------- #


def test_an_approval_travels_through_the_run_directory(tmp_path):
    run_dir = tmp_path / "run"
    actions = [{"name": "set_task_plan", "args": {"plan": {"target": "churned"}}}]
    write_approval_request(run_dir, "req1", actions)

    published = read_approval_request(run_dir)
    assert published["id"] == "req1"
    assert published["actions"][0]["args"]["plan"]["target"] == "churned"

    write_approval_response(run_dir, "req1", [{"type": "approve"}])
    assert await_approval_response(run_dir, "req1", timeout=1.0) == [{"type": "approve"}]
    # Answering clears the handshake, so the next gate starts clean.
    assert read_approval_request(run_dir) is None


def test_a_stale_answer_for_a_different_request_is_ignored(tmp_path):
    run_dir = tmp_path / "run"
    write_approval_request(run_dir, "req2", [])
    write_approval_response(run_dir, "an-older-request", [{"type": "approve"}])

    with pytest.raises(ApprovalTimeout):
        await_approval_response(run_dir, "req2", timeout=0.0, sleep=lambda _: None)


def test_waiting_for_an_answer_that_never_comes_times_out(tmp_path):
    run_dir = tmp_path / "run"
    write_approval_request(run_dir, "req3", [])
    with pytest.raises(ApprovalTimeout, match="--yes"):
        await_approval_response(run_dir, "req3", timeout=0.0, sleep=lambda _: None)


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(None, "-"), (0.0, "0.0s"), (12.34, "12.3s"), (75, "1m15s"), (3725, "1h02m")],
)
def test_durations_read_the_way_a_human_would_say_them(seconds, expected):
    assert format_duration(seconds) == expected


def _events(run_dir: Path) -> list[dict[str, Any]]:
    from autoxgb_agent.progress import read_events

    return read_events(run_dir)
