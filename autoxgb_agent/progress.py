"""Run progress as an append-only event log.

Every surface that shows a run — the CLI dashboard, `autoxgb watch`, the
Streamlit UI — is a view over one file: `<run_dir>/progress/events.jsonl`. That
keeps the surfaces honest (they cannot disagree about what happened) and lets a
second process watch a run that is already going, without talking to the agent.

Two independent signals feed the log, which together cover the whole pipeline:

- **Delegation**, read off the orchestrator's message stream: a `task` tool call
  starts a stage, its result ends one. This is what the *orchestrator* thinks is
  happening.
- **Tool execution**, emitted by `@guard` inside every ML tool. Each tool belongs
  to exactly one specialist, so a tool call also identifies its stage. This is
  what is *actually* happening, including inside a subagent whose own messages
  never reach the parent stream.

Nothing here imports the rest of the package: the tools import this module, so it
has to stay at the bottom of the dependency graph. Messages are read with
`getattr`, so LangChain is not a dependency either.
"""

from __future__ import annotations

import contextvars
import json
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #

PROGRESS_DIRNAME = "progress"
EVENTS_FILE = f"{PROGRESS_DIRNAME}/events.jsonl"
STATE_FILE = f"{PROGRESS_DIRNAME}/state.json"
APPROVAL_REQUEST_FILE = f"{PROGRESS_DIRNAME}/approval_request.json"
APPROVAL_RESPONSE_FILE = f"{PROGRESS_DIRNAME}/approval_response.json"
CONSOLE_LOG_FILE = f"{PROGRESS_DIRNAME}/console.log"

# Where the UI parks uploaded datasets. It lives alongside the run directories,
# so anything listing runs has to know it is not one.
UPLOAD_DIRNAME = "_uploads"

# How many activity lines a live view keeps in memory.
ACTIVITY_LIMIT = 400


@dataclass(frozen=True)
class StageSpec:
    """One specialist in the pipeline, in the order it should run."""

    key: str
    label: str
    tools: tuple[str, ...]


# Keys are the subagent names from `subagents.py`; tools are that subagent's
# slice of the tool suite. `tests/test_progress.py` pins both against the real
# specs so this cannot drift silently.
STAGES: tuple[StageSpec, ...] = (
    StageSpec("data-profiler", "Profile data", ("preview_data", "profile_dataset")),
    StageSpec(
        "target-detector",
        "Target & task type",
        ("check_target_leakage", "set_task_plan"),
    ),
    StageSpec(
        "feature-engineer",
        "Engineer features",
        ("build_feature_spec", "apply_preprocessing"),
    ),
    StageSpec("modeler", "Train baseline", ("train_xgboost",)),
    StageSpec("tuner", "Tune hyperparameters", ("tune_xgboost",)),
    StageSpec("evaluator", "Evaluate & report", ("evaluate_model",)),
    StageSpec("packager", "Export bundle", ("export_bundle",)),
)

STAGE_KEYS: tuple[str, ...] = tuple(spec.key for spec in STAGES)
STAGE_BY_KEY: dict[str, StageSpec] = {spec.key: spec for spec in STAGES}
STAGE_BY_TOOL: dict[str, str] = {
    tool: spec.key for spec in STAGES for tool in spec.tools
}

PENDING = "pending"
RUNNING = "running"
BLOCKED = "blocked"
DONE = "done"
FAILED = "failed"

# Terminal run statuses — a watcher stops following once it sees one.
TERMINAL_STATUSES = frozenset({DONE, FAILED, "interrupted"})


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #


@dataclass
class ToolCall:
    tool: str
    started_at: float
    ended_at: float | None = None
    ok: bool | None = None
    summary: str = ""
    # Where a long call has got to, for tools that report as they go.
    progress: str = ""
    fraction: float | None = None

    @property
    def duration(self) -> float | None:
        if self.ended_at is None:
            return None
        return self.ended_at - self.started_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "ok": self.ok,
            "summary": self.summary,
            "duration": self.duration,
            "progress": self.progress,
            "fraction": self.fraction,
        }


@dataclass
class StageState:
    key: str
    label: str
    status: str = PENDING
    started_at: float | None = None
    ended_at: float | None = None
    instruction: str = ""
    detail: str = ""
    error: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    def elapsed(self, now: float | None = None) -> float | None:
        if self.started_at is None:
            return None
        end = self.ended_at if self.ended_at is not None else (now or time.time())
        return max(0.0, end - self.started_at)

    @property
    def active_tool(self) -> str | None:
        for call in reversed(self.tool_calls):
            if call.ended_at is None:
                return call.tool
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "elapsed": self.elapsed(),
            "instruction": self.instruction,
            "detail": self.detail,
            "error": self.error,
            "active_tool": self.active_tool,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
        }


@dataclass
class RunState:
    """The reduction of every event seen so far."""

    run_id: str = ""
    goal: str = ""
    dataset: str = ""
    model: str = ""
    run_dir: str = ""
    started_at: float | None = None
    ended_at: float | None = None
    status: str = PENDING
    error: str | None = None
    stages: dict[str, StageState] = field(
        default_factory=lambda: {
            spec.key: StageState(spec.key, spec.label) for spec in STAGES
        }
    )
    todos: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    activity: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=ACTIVITY_LIMIT)
    )
    approval: dict[str, Any] | None = None
    n_events: int = 0
    n_tool_calls: int = 0
    n_errors: int = 0

    @property
    def ordered_stages(self) -> list[StageState]:
        return [self.stages[key] for key in STAGE_KEYS]

    @property
    def is_finished(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def elapsed(self, now: float | None = None) -> float:
        if self.started_at is None:
            return 0.0
        end = self.ended_at if self.ended_at is not None else (now or time.time())
        return max(0.0, end - self.started_at)

    def completed_stages(self) -> int:
        return sum(1 for stage in self.ordered_stages if stage.status == DONE)

    def current_stage(self) -> StageState | None:
        for stage in self.ordered_stages:
            if stage.status in {RUNNING, BLOCKED}:
                return stage
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "dataset": self.dataset,
            "model": self.model,
            "run_dir": self.run_dir,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "elapsed": self.elapsed(),
            "status": self.status,
            "error": self.error,
            "stages": [stage.to_dict() for stage in self.ordered_stages],
            "todos": list(self.todos),
            "artifacts": dict(self.artifacts),
            "activity": list(self.activity),
            "approval": self.approval,
            "n_events": self.n_events,
            "n_tool_calls": self.n_tool_calls,
            "n_errors": self.n_errors,
            "completed_stages": self.completed_stages(),
            "total_stages": len(STAGES),
        }


# --------------------------------------------------------------------------- #
# Reducer
# --------------------------------------------------------------------------- #


def _activity(
    state: RunState, ts: float, text: str, *, stage: str = "", level: str = "info"
) -> None:
    state.activity.append({"ts": ts, "stage": stage, "text": text, "level": level})


def _stage(state: RunState, key: str | None) -> StageState | None:
    return state.stages.get(key) if key else None


def _open_call(stage: StageState, tool: str) -> ToolCall | None:
    """The most recent call of `tool` on `stage` that has not finished."""
    return next(
        (c for c in reversed(stage.tool_calls) if c.tool == tool and c.ended_at is None),
        None,
    )


def apply_event(state: RunState, event: dict[str, Any]) -> RunState:
    """Fold one event into `state`, in place. Unknown event types are ignored."""
    kind = event.get("type", "")
    ts = float(event.get("ts") or time.time())
    state.n_events += 1

    if kind == "run_started":
        state.run_id = event.get("run_id", state.run_id)
        state.goal = event.get("goal", state.goal)
        state.dataset = event.get("dataset", state.dataset)
        state.model = event.get("model", state.model)
        state.run_dir = event.get("run_dir", state.run_dir)
        state.started_at = ts
        state.status = RUNNING
        _activity(state, ts, f"Run started: {state.goal}")

    elif kind == "run_finished":
        state.ended_at = ts
        state.status = event.get("status") or (DONE if event.get("ok", True) else FAILED)
        state.error = event.get("error") or state.error
        for stage in state.ordered_stages:
            if stage.status in {RUNNING, BLOCKED}:
                stage.status = DONE if state.status == DONE else FAILED
                stage.ended_at = ts
        state.approval = None
        _activity(
            state,
            ts,
            f"Run {state.status}" + (f": {state.error}" if state.error else ""),
            level="error" if state.status == FAILED else "info",
        )

    elif kind == "stage_started":
        stage = _stage(state, event.get("stage"))
        if stage is not None:
            stage.status = RUNNING
            stage.started_at = stage.started_at or ts
            stage.ended_at = None
            stage.instruction = event.get("instruction", "") or stage.instruction
            stage.detail = "delegated"
            _activity(state, ts, f"→ {stage.label}", stage=stage.key)

    elif kind == "stage_finished":
        stage = _stage(state, event.get("stage"))
        if stage is not None:
            ok = bool(event.get("ok", True))
            stage.status = DONE if ok else FAILED
            stage.ended_at = ts
            stage.started_at = stage.started_at or ts
            summary = (event.get("summary") or "").strip()
            if summary:
                stage.detail = _first_line(summary)
            if not ok:
                stage.error = summary or "failed"
                state.n_errors += 1
            _activity(
                state,
                ts,
                f"✓ {stage.label}" if ok else f"✗ {stage.label}",
                stage=stage.key,
                level="info" if ok else "error",
            )

    elif kind == "tool_started":
        tool = event.get("tool", "tool")
        stage = _stage(state, event.get("stage"))
        if stage is not None:
            if stage.status == PENDING:
                stage.status = RUNNING
                stage.started_at = stage.started_at or ts
            stage.tool_calls.append(ToolCall(tool=tool, started_at=ts))
            stage.detail = f"{tool}…"
        _activity(state, ts, f"  {tool}…", stage=event.get("stage", ""), level="tool")

    elif kind == "tool_progress":
        tool = event.get("tool", "tool")
        message = str(event.get("message", "")).strip()
        stage = _stage(state, event.get("stage"))
        if stage is not None:
            call = _open_call(stage, tool)
            if call is not None:
                call.progress = message
                call.fraction = event.get("fraction")
            stage.detail = f"{tool}: {message}" if message else stage.detail
        _activity(state, ts, f"  {tool}: {message}", stage=event.get("stage", ""), level="tool")

    elif kind == "tool_finished":
        tool = event.get("tool", "tool")
        ok = bool(event.get("ok", True))
        summary = (event.get("summary") or "").strip()
        duration = event.get("duration")
        state.n_tool_calls += 1
        stage = _stage(state, event.get("stage"))
        if stage is not None:
            call = _open_call(stage, tool)
            if call is None:
                call = ToolCall(tool=tool, started_at=ts - float(duration or 0.0))
                stage.tool_calls.append(call)
            call.ended_at = ts
            call.ok = ok
            call.summary = summary
            stage.detail = _first_line(summary) or tool
            if not ok:
                stage.error = summary
        if not ok:
            state.n_errors += 1
        _activity(
            state,
            ts,
            f"  {tool} {'ok' if ok else 'failed'}"
            + (f" ({float(duration):.1f}s)" if duration else "")
            + (f" — {_first_line(summary)}" if summary else ""),
            stage=event.get("stage", ""),
            level="tool" if ok else "error",
        )

    elif kind == "todos":
        state.todos = list(event.get("items") or [])

    elif kind == "artifact":
        path = event.get("path", "")
        if path:
            state.artifacts[path] = {
                "path": path,
                "bytes": event.get("bytes", 0),
                "stage": event.get("stage", ""),
                "ts": ts,
            }
            _activity(state, ts, f"  wrote {path}", stage=event.get("stage", ""), level="artifact")

    elif kind == "note":
        text = (event.get("text") or "").strip()
        if text:
            _activity(state, ts, text, stage=event.get("stage", ""), level="note")

    elif kind == "approval_requested":
        state.approval = {
            "id": event.get("id", ""),
            "actions": event.get("actions") or [],
            "ts": ts,
        }
        stage = _stage(state, event.get("stage") or "target-detector")
        if stage is not None and stage.status in {PENDING, RUNNING}:
            stage.status = BLOCKED
            stage.started_at = stage.started_at or ts
            stage.detail = "waiting for your approval"
        _activity(state, ts, "⏸ waiting for approval", level="approval")

    elif kind == "approval_resolved":
        state.approval = None
        decision = event.get("decision", "approve")
        stage = _stage(state, event.get("stage") or "target-detector")
        if stage is not None and stage.status == BLOCKED:
            stage.status = RUNNING
            stage.detail = f"plan {decision}d"
        _activity(
            state,
            ts,
            f"▶ plan {decision}d"
            + (f": {event.get('message')}" if event.get("message") else ""),
            level="approval",
        )

    return state


def _first_line(text: str, limit: int = 120) -> str:
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    return line if len(line) <= limit else line[: limit - 1] + "…"


def replay(events: Iterator[dict[str, Any]] | list[dict[str, Any]]) -> RunState:
    """Fold a whole event stream into a fresh `RunState`."""
    state = RunState()
    for event in events:
        apply_event(state, event)
    return state


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


class ProgressRecorder:
    """Appends events to the run's log and keeps a reduced snapshot on disk.

    The JSONL log is the source of truth; `state.json` is a convenience for
    anything that wants the current picture without replaying (the `runs`
    listing, mostly). Writes are small and infrequent — a few dozen per run — so
    they happen synchronously and are flushed immediately, which is what lets
    another process tail them.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.events_path = self.run_dir / EVENTS_FILE
        self.state_path = self.run_dir / STATE_FILE
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = RunState(run_dir=str(self.run_dir))
        self._lock = threading.Lock()
        self._seen_files: dict[str, tuple[int, float]] = {}

    # -- emission ---------------------------------------------------------- #

    def emit(self, kind: str, **payload: Any) -> dict[str, Any]:
        """Record one event. Never raises: progress must not break a run."""
        event = {"ts": time.time(), "type": kind, **payload}
        with self._lock:
            apply_event(self.state, event)
            try:
                with self.events_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, default=str) + "\n")
                self._write_state()
            except OSError:  # pragma: no cover - disk trouble is not the run's problem
                pass
        return event

    def _write_state(self) -> None:
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state.to_dict(), default=str), encoding="utf-8")
        tmp.replace(self.state_path)

    # -- run lifecycle ----------------------------------------------------- #

    def run_started(self, *, run_id: str, goal: str, dataset: str, model: str) -> None:
        self.emit(
            "run_started",
            run_id=run_id,
            goal=goal,
            dataset=dataset,
            model=model,
            run_dir=str(self.run_dir),
        )

    def run_finished(self, *, status: str, error: str | None = None) -> None:
        self.scan_artifacts()
        self.emit("run_finished", status=status, ok=status == DONE, error=error)

    # -- stages ------------------------------------------------------------ #

    def stage_started(self, stage: str, instruction: str = "") -> None:
        self.emit("stage_started", stage=stage, instruction=instruction)

    def stage_finished(self, stage: str, *, ok: bool = True, summary: str = "") -> None:
        self.emit("stage_finished", stage=stage, ok=ok, summary=summary)

    # -- tools ------------------------------------------------------------- #

    def tool_started(self, tool: str) -> dict[str, Any]:
        stage = STAGE_BY_TOOL.get(tool)
        self.emit("tool_started", tool=tool, stage=stage)
        return {"tool": tool, "stage": stage, "t0": time.perf_counter()}

    def tool_report(self, tool: str, message: str, fraction: float | None = None) -> None:
        self.emit(
            "tool_progress",
            tool=tool,
            stage=STAGE_BY_TOOL.get(tool),
            message=message,
            fraction=fraction,
        )

    def tool_finished(self, handle: dict[str, Any], *, ok: bool, summary: str) -> None:
        duration = time.perf_counter() - float(handle.get("t0", time.perf_counter()))
        self.emit(
            "tool_finished",
            tool=handle.get("tool", "tool"),
            stage=handle.get("stage"),
            ok=ok,
            duration=round(duration, 3),
            summary=_first_line(summary, 200),
        )
        self.scan_artifacts(stage=handle.get("stage"))

    # -- artifacts --------------------------------------------------------- #

    def scan_artifacts(self, stage: str | None = None) -> list[str]:
        """Emit an event for every run-directory file that is new or changed.

        Diffing the directory rather than instrumenting each write catches
        everything a stage produces — reports, parquet splits, the saved booster,
        the PNGs, the whole bundle — without threading a recorder through code
        that has no other reason to know about progress.
        """
        found: list[str] = []
        if not self.run_dir.exists():
            return found
        for path in sorted(self.run_dir.rglob("*")):
            if not path.is_file():
                continue
            try:
                relative = path.relative_to(self.run_dir).as_posix()
            except ValueError:  # pragma: no cover - defensive
                continue
            if relative.startswith(f"{PROGRESS_DIRNAME}/"):
                continue
            try:
                stat = path.stat()
            except OSError:  # pragma: no cover - vanished mid-scan
                continue
            signature = (stat.st_size, stat.st_mtime)
            if self._seen_files.get(relative) == signature:
                continue
            self._seen_files[relative] = signature
            self.emit("artifact", path=relative, bytes=stat.st_size, stage=stage or "")
            found.append(relative)
        return found

    # -- narration & approval ---------------------------------------------- #

    def note(self, text: str, stage: str | None = None) -> None:
        self.emit("note", text=text, stage=stage or "")

    def approval_requested(self, actions: list[dict[str, Any]], stage: str | None = None) -> str:
        request_id = uuid.uuid4().hex[:12]
        self.emit(
            "approval_requested",
            id=request_id,
            actions=_jsonable(actions),
            stage=stage or _stage_for_actions(actions),
        )
        return request_id

    def approval_resolved(self, decision: str, message: str = "", stage: str | None = None) -> None:
        self.emit(
            "approval_resolved", decision=decision, message=message, stage=stage or ""
        )


def _stage_for_actions(actions: list[dict[str, Any]]) -> str:
    for action in actions:
        stage = STAGE_BY_TOOL.get(str(action.get("name", "")))
        if stage:
            return stage
    return "target-detector"


def _jsonable(value: Any) -> Any:
    """Round-trip through JSON so a payload from LangChain is safe to write."""
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return str(value)


# --------------------------------------------------------------------------- #
# The ambient recorder
# --------------------------------------------------------------------------- #

_recorder: contextvars.ContextVar[ProgressRecorder | None] = contextvars.ContextVar(
    "autoxgb_progress_recorder", default=None
)
# A run owns its process, so a module-level fallback is safe and covers tool
# calls that LangGraph dispatches onto a thread the context did not follow.
_fallback_recorder: ProgressRecorder | None = None


def set_recorder(recorder: ProgressRecorder | None) -> contextvars.Token[ProgressRecorder | None]:
    global _fallback_recorder  # noqa: PLW0603 - see the comment above
    _fallback_recorder = recorder
    return _recorder.set(recorder)


def reset_recorder(token: contextvars.Token[ProgressRecorder | None]) -> None:
    global _fallback_recorder  # noqa: PLW0603
    _recorder.reset(token)
    _fallback_recorder = _recorder.get()


def current_recorder() -> ProgressRecorder | None:
    return _recorder.get() or _fallback_recorder


def tool_started(tool: str) -> dict[str, Any] | None:
    """No-op when nothing is recording, so tools can call it unconditionally."""
    recorder = current_recorder()
    return recorder.tool_started(tool) if recorder is not None else None


def tool_finished(handle: dict[str, Any] | None, *, ok: bool, summary: str) -> None:
    recorder = current_recorder()
    if recorder is not None and handle is not None:
        recorder.tool_finished(handle, ok=ok, summary=summary)


def tool_progress(tool: str, message: str, fraction: float | None = None) -> None:
    """Report from inside a long tool call, so it is not a silent minute.

    Also a no-op when nothing is recording — a tool can call it freely.
    """
    recorder = current_recorder()
    if recorder is not None:
        recorder.tool_report(tool, message, fraction)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def read_events(run_dir: Path) -> list[dict[str, Any]]:
    """Every event written for a run, oldest first."""
    path = Path(run_dir) / EVENTS_FILE
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # A partial final line means a writer is mid-append; skip it and it
            # will be read whole on the next poll.
            continue
    return events


def read_progress(run_dir: Path) -> RunState:
    """Replay a run's log into a `RunState`."""
    state = replay(read_events(run_dir))
    if not state.run_dir:
        state.run_dir = str(run_dir)
    return state


def read_state_file(run_dir: Path) -> dict[str, Any] | None:
    """The recorder's last snapshot, without replaying the log."""
    path = Path(run_dir) / STATE_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


class ProgressReader:
    """Incrementally tails a run's event log.

    `poll()` returns only the events appended since the last call, so a watcher
    can both keep a `RunState` current and print what is new.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.state = RunState(run_dir=str(run_dir))
        self._consumed = 0

    def poll(self) -> list[dict[str, Any]]:
        events = read_events(self.run_dir)
        fresh = events[self._consumed :]
        self._consumed = len(events)
        for event in fresh:
            apply_event(self.state, event)
        return fresh


# --------------------------------------------------------------------------- #
# The out-of-band approval handshake
# --------------------------------------------------------------------------- #


def write_approval_request(
    run_dir: Path, request_id: str, actions: list[dict[str, Any]]
) -> Path:
    """Publish a pending approval for another process (the UI) to answer."""
    path = Path(run_dir) / APPROVAL_REQUEST_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    response = Path(run_dir) / APPROVAL_RESPONSE_FILE
    response.unlink(missing_ok=True)
    payload = {"id": request_id, "ts": time.time(), "actions": _jsonable(actions)}
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def read_approval_request(run_dir: Path) -> dict[str, Any] | None:
    path = Path(run_dir) / APPROVAL_REQUEST_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_approval_response(
    run_dir: Path, request_id: str, decisions: list[dict[str, Any]]
) -> Path:
    """Answer a pending approval. The id must match, so a stale file is ignored."""
    path = Path(run_dir) / APPROVAL_RESPONSE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps({"id": request_id, "ts": time.time(), "decisions": decisions}, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def read_approval_response(run_dir: Path, request_id: str) -> list[dict[str, Any]] | None:
    path = Path(run_dir) / APPROVAL_RESPONSE_FILE
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("id") != request_id:
        return None
    decisions = payload.get("decisions")
    return decisions if isinstance(decisions, list) else None


def clear_approval(run_dir: Path) -> None:
    for name in (APPROVAL_REQUEST_FILE, APPROVAL_RESPONSE_FILE):
        (Path(run_dir) / name).unlink(missing_ok=True)


class ApprovalTimeout(RuntimeError):
    """Nobody answered a file-based approval request in time."""


def await_approval_response(
    run_dir: Path,
    request_id: str,
    *,
    timeout: float = 1800.0,
    poll_seconds: float = 0.5,
    sleep: Callable[[float], Any] = time.sleep,
) -> list[dict[str, Any]]:
    """Block until the request is answered, or raise `ApprovalTimeout`."""
    deadline = time.time() + timeout
    while True:
        decisions = read_approval_response(run_dir, request_id)
        if decisions is not None:
            clear_approval(run_dir)
            return decisions
        if time.time() >= deadline:
            clear_approval(run_dir)
            msg = (
                f"no approval answer after {timeout:.0f}s. Answer it in the UI, with "
                "`autoxgb watch --approve`, or run with --yes."
            )
            raise ApprovalTimeout(msg)
        sleep(poll_seconds)


# --------------------------------------------------------------------------- #
# Formatting helpers shared by the CLI and the UI
# --------------------------------------------------------------------------- #

STATUS_ICONS = {
    PENDING: "○",
    RUNNING: "◐",
    BLOCKED: "⏸",
    DONE: "●",
    FAILED: "✗",
}


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{rest:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"
