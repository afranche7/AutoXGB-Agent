"""Starting and inspecting runs from outside the CLI.

The UI does not build the agent in-process. It spawns `autoxgb run` and then
watches the same progress log everything else reads, which keeps the front-end a
viewer: a Streamlit rerun cannot disturb a run, and closing the browser does not
kill one.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from autoxgb_agent.progress import (
    CONSOLE_LOG_FILE,
    EVENTS_FILE,
    PROGRESS_DIRNAME,
    RunState,
    read_progress,
)

UPLOAD_DIRNAME = "_uploads"


@dataclass(frozen=True)
class LaunchSpec:
    """Everything `autoxgb run` needs, as the UI collects it."""

    dataset: Path
    goal: str
    run_id: str
    output_dir: Path
    model: str | None = None
    tuning_trials: int | None = None
    tuning_timeout: int | None = None
    seed: int = 42
    auto_approve: bool = False

    def command(self) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "autoxgb_agent.cli",
            "run",
            str(self.dataset),
            "--goal",
            self.goal,
            "--run-id",
            self.run_id,
            "--output-dir",
            str(self.output_dir),
            "--seed",
            str(self.seed),
            "--plain",
            "--quiet",
        ]
        # `file` publishes the approval for this UI to answer; `--yes` skips the
        # gate entirely, for people who would just click approve anyway.
        command += ["--yes"] if self.auto_approve else ["--approve-via", "file"]
        if self.model:
            command += ["--model", self.model]
        if self.tuning_trials:
            command += ["--tuning-trials", str(self.tuning_trials)]
        if self.tuning_timeout:
            command += ["--tuning-timeout", str(self.tuning_timeout)]
        return command


def launch(spec: LaunchSpec) -> subprocess.Popen[bytes]:
    """Start a run in its own process, teeing its console output into the run dir."""
    run_dir = spec.output_dir / spec.run_id
    (run_dir / PROGRESS_DIRNAME).mkdir(parents=True, exist_ok=True)
    log_path = run_dir / CONSOLE_LOG_FILE
    log = log_path.open("wb")
    environment = {**os.environ, "PYTHONUNBUFFERED": "1", "NO_COLOR": "1"}
    return subprocess.Popen(  # noqa: S603 - argv is built here, never a shell string
        spec.command(),
        stdout=log,
        stderr=subprocess.STDOUT,
        env=environment,
        cwd=str(Path.cwd()),
    )


def save_upload(output_dir: Path, name: str, data: bytes) -> Path:
    """Persist an uploaded dataset next to the runs that will use it."""
    target = output_dir / UPLOAD_DIRNAME / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


def list_runs(output_dir: Path) -> list[Path]:
    """Run directories, most recently touched first."""
    if not output_dir.exists():
        return []
    runs = [
        path
        for path in output_dir.iterdir()
        if path.is_dir() and path.name != UPLOAD_DIRNAME
    ]
    return sorted(runs, key=_last_touched, reverse=True)


def _last_touched(run_dir: Path) -> float:
    events = run_dir / EVENTS_FILE
    try:
        return (events if events.exists() else run_dir).stat().st_mtime
    except OSError:  # pragma: no cover - vanished between listing and stat
        return 0.0


def load_state(run_dir: Path) -> RunState:
    return read_progress(run_dir)


def console_log(run_dir: Path, max_chars: int = 20_000) -> str:
    """The tail of a UI-launched run's console output."""
    path = run_dir / CONSOLE_LOG_FILE
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text if len(text) <= max_chars else "…\n" + text[-max_chars:]


def artifact_tree(run_dir: Path) -> list[Path]:
    """Every artifact a run has written, progress bookkeeping excluded."""
    if not run_dir.exists():
        return []
    return [
        path
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and PROGRESS_DIRNAME not in path.relative_to(run_dir).parts
    ]
