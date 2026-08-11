"""The web UI's plumbing.

The UI itself is Streamlit and is exercised by hand, but everything it does that
can go wrong quietly — how it starts a run, where it finds them, what it shows —
lives in `ui/launcher.py` and is tested here.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from autoxgb_agent.progress import CONSOLE_LOG_FILE, PROGRESS_DIRNAME, ProgressRecorder
from autoxgb_agent.ui import launcher


def _spec(tmp_path: Path, **overrides) -> launcher.LaunchSpec:
    defaults = {
        "dataset": tmp_path / "churn.csv",
        "goal": "predict churn",
        "run_id": "r1",
        "output_dir": tmp_path / "outputs",
    }
    return launcher.LaunchSpec(**{**defaults, **overrides})


# --------------------------------------------------------------------------- #
# Launching
# --------------------------------------------------------------------------- #


def test_a_ui_run_publishes_its_approval_instead_of_prompting(tmp_path):
    """Nothing can answer a terminal prompt from a browser, so it must not use one."""
    command = _spec(tmp_path).command()
    assert command[1:3] == ["-m", "autoxgb_agent.cli"]
    assert "--approve-via" in command
    assert command[command.index("--approve-via") + 1] == "file"
    assert "--yes" not in command
    # A dashboard redrawing into a pipe would be noise in the console log.
    assert "--plain" in command


def test_auto_approve_skips_the_gate_entirely(tmp_path):
    command = _spec(tmp_path, auto_approve=True).command()
    assert "--yes" in command
    assert "--approve-via" not in command


def test_optional_settings_are_only_passed_when_set(tmp_path):
    bare = _spec(tmp_path).command()
    assert "--model" not in bare and "--tuning-trials" not in bare

    full = _spec(
        tmp_path, model="anthropic:claude-opus-5", tuning_trials=7, tuning_timeout=60, seed=1
    ).command()
    assert full[full.index("--model") + 1] == "anthropic:claude-opus-5"
    assert full[full.index("--tuning-trials") + 1] == "7"
    assert full[full.index("--tuning-timeout") + 1] == "60"
    assert full[full.index("--seed") + 1] == "1"


def test_launching_starts_a_real_process_and_captures_its_output(tmp_path, monkeypatch):
    """The whole path: spawn, run directory, console log.

    The run is made to stop immediately (no API key) so this stays fast and
    free — what is under test is the wiring, not the pipeline.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    dataset = tmp_path / "churn.csv"
    dataset.write_text("a,b\n1,2\n", encoding="utf-8")
    output_dir = tmp_path / "outputs"

    process = launcher.launch(
        _spec(tmp_path, dataset=dataset, output_dir=output_dir, run_id="spawned")
    )
    assert process.wait(timeout=120) == 2, "expected the missing-key exit"

    log = launcher.console_log(output_dir / "spawned")
    assert "ANTHROPIC_API_KEY" in log


def test_an_uploaded_dataset_is_kept_where_later_runs_can_find_it(tmp_path):
    saved = launcher.save_upload(tmp_path, "churn.csv", b"a,b\n1,2\n")
    assert saved.read_bytes() == b"a,b\n1,2\n"
    assert saved.parent.name == launcher.UPLOAD_DIRNAME
    # Uploads are not runs, so they must not show up in the run list.
    assert launcher.list_runs(tmp_path) == []


# --------------------------------------------------------------------------- #
# Finding and reading runs
# --------------------------------------------------------------------------- #


def test_runs_are_listed_most_recently_active_first(tmp_path):
    (tmp_path / "no-progress-yet").mkdir()
    for name in ("first", "second"):
        recorder = ProgressRecorder(tmp_path / name)
        recorder.run_started(run_id=name, goal="g", dataset="d", model="m")
        time.sleep(0.01)

    names = [path.name for path in launcher.list_runs(tmp_path)]
    assert names[:2] == ["second", "first"]
    assert set(names) == {"first", "second", "no-progress-yet"}


def test_the_artifact_list_is_what_the_run_produced_not_its_bookkeeping(tmp_path):
    run_dir = tmp_path / "run"
    recorder = ProgressRecorder(run_dir)
    recorder.run_started(run_id="r", goal="g", dataset="d", model="m")
    (run_dir / "profile_report.md").write_text("# Data profile", encoding="utf-8")
    (run_dir / "bundle").mkdir()
    (run_dir / "bundle" / "predict.py").write_text("print()", encoding="utf-8")

    names = [path.relative_to(run_dir).as_posix() for path in launcher.artifact_tree(run_dir)]
    assert names == ["bundle/predict.py", "profile_report.md"]
    assert not any(name.startswith(PROGRESS_DIRNAME) for name in names)


def test_a_long_console_log_is_tailed_rather_than_loaded_whole(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / PROGRESS_DIRNAME).mkdir(parents=True)
    (run_dir / CONSOLE_LOG_FILE).write_text("x" * 50_000 + "the end", encoding="utf-8")

    tail = launcher.console_log(run_dir, max_chars=1_000)
    assert tail.endswith("the end")
    assert len(tail) <= 1_002


def test_no_console_log_is_not_an_error(tmp_path):
    assert launcher.console_log(tmp_path / "nothing") == ""
    assert launcher.artifact_tree(tmp_path / "nothing") == []
    assert launcher.list_runs(tmp_path / "nothing") == []


# --------------------------------------------------------------------------- #
# The Streamlit page
# --------------------------------------------------------------------------- #


APP = Path(__file__).resolve().parents[1] / "autoxgb_agent" / "ui" / "app.py"


@pytest.fixture
def page(monkeypatch):
    """Render the real Streamlit page against a run directory."""
    testing = pytest.importorskip("streamlit.testing.v1")

    def _render(output_dir: Path):
        monkeypatch.setattr(sys, "argv", ["app.py", "--output-dir", str(output_dir)])
        # Left on its default auto-refresh, so an unfinished run renders through
        # the polling fragment — the page must not block the script to refresh.
        return testing.AppTest.from_file(str(APP), default_timeout=30).run()

    return _render


def _finished_run(output_dir: Path) -> ProgressRecorder:
    recorder = ProgressRecorder(output_dir / "run-1")
    recorder.run_started(
        run_id="run-1", goal="predict churn", dataset="churn.csv", model="m"
    )
    recorder.stage_started("data-profiler", "Profile it.")
    (recorder.run_dir / "profile_report.md").write_text("# Data profile", encoding="utf-8")
    handle = recorder.tool_started("profile_dataset")
    recorder.tool_finished(handle, ok=True, summary="Profiled 900 rows x 11 columns.")
    recorder.stage_finished("data-profiler", ok=True, summary="churned is the target.")
    recorder.run_finished(status="done")
    return recorder


def test_the_page_shows_where_a_run_got_to(page, tmp_path):
    _finished_run(tmp_path)
    app = page(tmp_path)

    assert not app.exception
    assert [metric.label for metric in app.metric] == [
        "Status",
        "Stages",
        "Elapsed",
        "Tool calls",
        "Errors",
    ]
    assert [metric.value for metric in app.metric][:2] == ["done", "1/7"]
    assert [tab.label for tab in app.tabs] == [
        "Stages",
        "Plan",
        "Activity",
        "Artifacts",
        "Console",
    ]


def test_tool_calls_render_without_a_dataframe(tmp_path):
    """Handing `st.dataframe` a list of dicts segfaults the server; markdown cannot."""
    pytest.importorskip("streamlit")
    from autoxgb_agent.progress import ToolCall
    from autoxgb_agent.ui.app import tool_call_table

    table = tool_call_table(
        [
            ToolCall(tool="train_xgboost", started_at=0.0, ended_at=2.5, ok=True, summary="Fitted."),
            ToolCall(tool="tune_xgboost", started_at=0.0, progress="trial 3/12 — best 0.81"),
            ToolCall(
                tool="evaluate_model",
                started_at=0.0,
                ended_at=1.0,
                ok=False,
                summary="ERROR: no model | yet",
            ),
        ]
    )
    lines = table.splitlines()
    assert lines[0].startswith("| tool |")
    assert "`train_xgboost` | ✅ ok | 2.50s | Fitted." in lines[2]
    assert "◐ running | — | trial 3/12" in lines[3]
    assert "❌ failed" in lines[4]
    assert "no model \\| yet" in lines[4], "an unescaped pipe would break the table"


def test_the_page_renders_with_no_runs_at_all(page, tmp_path):
    app = page(tmp_path)
    assert not app.exception
    assert any("No runs yet" in info.value for info in app.info)


def _paused_run(output_dir: Path) -> tuple[ProgressRecorder, str]:
    from autoxgb_agent.progress import write_approval_request

    recorder = ProgressRecorder(output_dir / "run-1")
    recorder.run_started(run_id="run-1", goal="predict churn", dataset="d", model="m")
    recorder.stage_started("target-detector", "Pick the target.")
    actions = [
        {
            "name": "set_task_plan",
            "args": {"plan": {"target": "churned", "reasoning": "the goal names churn"}},
        }
    ]
    request_id = recorder.approval_requested(actions)
    write_approval_request(recorder.run_dir, request_id, actions)
    return recorder, request_id


def test_approving_from_the_page_answers_the_waiting_run(page, tmp_path):
    from autoxgb_agent.progress import read_approval_response

    recorder, request_id = _paused_run(tmp_path)
    app = page(tmp_path)

    assert any("paused" in warning.value for warning in app.warning)
    _click(app, "Approve")
    assert read_approval_response(recorder.run_dir, request_id) == [{"type": "approve"}]


def test_rejecting_from_the_page_sends_the_feedback_back(page, tmp_path):
    from autoxgb_agent.progress import read_approval_response

    recorder, request_id = _paused_run(tmp_path)
    app = page(tmp_path)

    app.text_input(key="approval_feedback").input("use `renewed` instead").run()
    _click(app, "Reject")

    assert read_approval_response(recorder.run_dir, request_id) == [
        {"type": "reject", "message": "use `renewed` instead"}
    ]


def _click(app, label: str) -> None:
    button = next(button for button in app.button if button.label == label)
    button.click().run()
    assert not app.exception


def test_the_page_is_where_the_cli_tells_streamlit_to_look():
    from autoxgb_agent import cli

    assert (Path(cli.__file__).parent / "ui" / "app.py").exists()


def test_the_launched_run_inherits_the_environment(tmp_path, monkeypatch):
    """The API key comes from wherever the UI was started."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-inherited")
    dataset = tmp_path / "churn.csv"
    dataset.write_text("a,b\n1,2\n", encoding="utf-8")
    recorded: dict[str, object] = {}

    class FakePopen:
        def __init__(self, command, **kwargs):
            recorded["command"] = command
            recorded["env"] = kwargs["env"]

    monkeypatch.setattr(launcher.subprocess, "Popen", FakePopen)
    launcher.launch(_spec(tmp_path, dataset=dataset, output_dir=tmp_path / "outputs"))

    assert recorded["env"]["ANTHROPIC_API_KEY"] == "sk-ant-inherited"
    assert recorded["env"]["PYTHONUNBUFFERED"] == "1"
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-inherited"
