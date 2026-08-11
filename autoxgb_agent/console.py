"""Terminal rendering of a run: the pinned stage dashboard and the narration.

Two things happen while a run streams. The *narration* — what the agent said,
which tool it called, what came back — scrolls past as it always did. The
*dashboard* — seven stages, their status, elapsed time and current tool — stays
pinned at the bottom, so at any moment you can see where the pipeline is without
reading back.

`StreamTracker` is the other half: it turns the orchestrator's message stream
into progress events. It reads messages with `getattr`, so it works on anything
message-shaped and does not care which LangChain version produced it.
"""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from autoxgb_agent.progress import (
    BLOCKED,
    DONE,
    FAILED,
    PENDING,
    RUNNING,
    STATUS_ICONS,
    ProgressRecorder,
    RunState,
    format_duration,
)

# The delegation tool DeepAgents gives the orchestrator, and its argument
# carrying the subagent name.
TASK_TOOL = "task"
SUBAGENT_ARG = "subagent_type"
TODO_TOOL = "write_todos"

_STATUS_STYLES = {
    PENDING: "dim",
    RUNNING: "bold cyan",
    BLOCKED: "bold yellow",
    DONE: "green",
    FAILED: "bold red",
}

_LEVEL_STYLES = {
    "info": "white",
    "note": "white",
    "tool": "dim",
    "artifact": "dim green",
    "error": "red",
    "approval": "yellow",
}


# --------------------------------------------------------------------------- #
# Message → events
# --------------------------------------------------------------------------- #


class StreamTracker:
    """Translates the orchestrator's stream into stage-level progress events.

    Only the orchestrator's own messages come through the stream — a subagent
    runs inside the `task` tool, so its turns never appear. That is exactly the
    split this class relies on: delegation boundaries come from here, and what
    happens *inside* a stage is reported by the tools themselves.
    """

    def __init__(self, recorder: ProgressRecorder) -> None:
        self.recorder = recorder
        self._stage_by_call: dict[str, str] = {}

    def consume(self, update: Any) -> None:
        """Feed one node update from `stream_mode="updates"`."""
        if not isinstance(update, dict):
            return
        for message in update.get("messages") or []:
            self.consume_message(message)

    def consume_message(self, message: Any) -> None:
        kind = getattr(message, "type", None)
        if kind == "ai":
            self._consume_ai(message)
        elif kind == "tool":
            self._consume_tool_result(message)

    def _consume_ai(self, message: Any) -> None:
        text = message_text(message).strip()
        if text:
            self.recorder.note(text)
        for call in getattr(message, "tool_calls", None) or []:
            name = call.get("name")
            args = call.get("args") or {}
            if name == TASK_TOOL:
                stage = str(args.get(SUBAGENT_ARG) or "")
                if stage:
                    call_id = str(call.get("id") or stage)
                    self._stage_by_call[call_id] = stage
                    self.recorder.stage_started(stage, str(args.get("description") or ""))
            elif name == TODO_TOOL:
                todos = args.get("todos")
                if isinstance(todos, list):
                    self.recorder.emit("todos", items=_normalise_todos(todos))

    def _consume_tool_result(self, message: Any) -> None:
        if getattr(message, "name", None) != TASK_TOOL:
            return
        call_id = str(getattr(message, "tool_call_id", "") or "")
        stage = self._stage_by_call.pop(call_id, None)
        if stage is None:
            # No matching call id (a resumed thread, say) — close whichever
            # stage is still open.
            open_stages = [
                key
                for key, value in self.recorder.state.stages.items()
                if value.status in {RUNNING, BLOCKED}
            ]
            stage = open_stages[-1] if open_stages else None
        if stage is None:
            return
        summary = message_text(message)
        ok = getattr(message, "status", "success") != "error"
        self.recorder.stage_finished(stage, ok=ok, summary=summary)


def _normalise_todos(todos: list[Any]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for todo in todos:
        if isinstance(todo, dict):
            items.append(
                {
                    "content": str(todo.get("content", "")),
                    "status": str(todo.get("status", "pending")),
                }
            )
        else:
            items.append({"content": str(todo), "status": "pending"})
    return items


def message_text(message: Any) -> str:
    """Flatten a message's content to text, whichever block shape it uses."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Narration
# --------------------------------------------------------------------------- #


class MessageRenderer:
    """Prints the agent's turns as they stream past."""

    def __init__(self, console: Console, *, quiet: bool = False) -> None:
        self.console = console
        self.quiet = quiet

    def render(self, update: Any) -> None:
        if not isinstance(update, dict):
            return
        for message in update.get("messages") or []:
            kind = getattr(message, "type", None)
            if kind == "ai":
                self._render_ai(message)
            elif kind == "tool" and not self.quiet:
                self._render_tool(message)

    def _render_ai(self, message: Any) -> None:
        text = message_text(message).strip()
        if text:
            self.console.print(text, style="white")
        for call in getattr(message, "tool_calls", None) or []:
            label = call["name"]
            args = call.get("args") or {}
            if label == TASK_TOOL and args.get(SUBAGENT_ARG):
                self.console.print(f"  → delegate to {args[SUBAGENT_ARG]}", style="bold cyan")
            else:
                self.console.print(f"  → {label}", style="cyan")
            if not self.quiet and args:
                self.console.print(indent(short_json(args)), style="dim cyan")

    def _render_tool(self, message: Any) -> None:
        name = getattr(message, "name", "tool")
        body = message_text(message).strip()
        if body:
            self.console.print(f"  ← {name}", style="green")
            self.console.print(indent(truncate(body)), style="dim")


def short_json(payload: Any, limit: int = 400) -> str:
    text = json.dumps(payload, default=str)
    return text if len(text) <= limit else text[:limit] + " …"


def truncate(text: str, max_lines: int = 12) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    hidden = len(lines) - max_lines
    return "\n".join([*lines[:max_lines], f"… {hidden} more line(s)"])


def indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


# --------------------------------------------------------------------------- #
# The dashboard
# --------------------------------------------------------------------------- #


def stage_table(state: RunState) -> Table:
    table = Table(
        show_header=True,
        header_style="dim",
        expand=True,
        box=None,
        pad_edge=False,
    )
    table.add_column("", width=2, no_wrap=True)
    table.add_column("stage", width=20, no_wrap=True)
    table.add_column("status", width=9, no_wrap=True)
    table.add_column("time", width=8, justify="right", no_wrap=True)
    table.add_column("tools", width=5, justify="right", no_wrap=True)
    table.add_column("detail", overflow="ellipsis", no_wrap=True)

    for stage in state.ordered_stages:
        style = _STATUS_STYLES.get(stage.status, "white")
        detail = stage.detail or ("" if stage.status == PENDING else "…")
        open_call = next(
            (call for call in reversed(stage.tool_calls) if call.ended_at is None), None
        )
        if stage.status == RUNNING and open_call is not None:
            detail = (
                f"{open_call.tool}: {open_call.progress}"
                if open_call.progress
                else f"{open_call.tool}…"
            )
        done_calls = sum(1 for call in stage.tool_calls if call.ended_at is not None)
        table.add_row(
            Text(STATUS_ICONS.get(stage.status, "?"), style=style),
            Text(stage.label, style=style),
            Text(stage.status, style=style),
            Text(format_duration(stage.elapsed()) if stage.started_at else "-", style="dim"),
            Text(str(done_calls) if stage.tool_calls else "-", style="dim"),
            Text(detail, style="dim" if stage.status != FAILED else "red"),
        )
    return table


def todo_panel(state: RunState) -> Panel | None:
    if not state.todos:
        return None
    marks = {"completed": ("[green]✓[/green]", "dim"), "in_progress": ("[cyan]▸[/cyan]", "bold")}
    lines = []
    for todo in state.todos:
        mark, style = marks.get(str(todo.get("status")), ("[dim]○[/dim]", "dim"))
        lines.append(f"{mark} [{style}]{todo.get('content', '')}[/{style}]")
    return Panel(
        "\n".join(lines), title="[dim]plan[/dim]", border_style="dim", padding=(0, 1)
    )


def header_text(state: RunState) -> Text:
    done = state.completed_stages()
    total = len(state.ordered_stages)
    status_style = _STATUS_STYLES.get(
        RUNNING if state.status == RUNNING else DONE if state.status == DONE else FAILED,
        "white",
    )
    header = Text()
    header.append(f"{done}/{total} stages", style="bold")
    header.append("  •  ", style="dim")
    header.append(format_duration(state.elapsed()), style="dim")
    header.append("  •  ", style="dim")
    header.append(f"{state.n_tool_calls} tool calls", style="dim")
    if state.n_errors:
        header.append(f"  •  {state.n_errors} error(s)", style="red")
    header.append("  •  ", style="dim")
    header.append(state.status, style=status_style)
    return header


def dashboard(state: RunState) -> RenderableType:
    parts: list[RenderableType] = [stage_table(state)]
    todos = todo_panel(state)
    if todos is not None:
        parts.append(todos)
    if state.approval is not None:
        parts.append(
            Text("⏸ waiting for your approval of the task plan", style="bold yellow")
        )
    return Panel(
        Group(*parts),
        title="[bold]pipeline[/bold]",
        subtitle=header_text(state),
        border_style="blue" if not state.is_finished else "green",
        padding=(0, 1),
    )


class RunDashboard:
    """A pinned dashboard that coexists with scrolling narration.

    Falls back to periodic one-line summaries when the output is not a terminal,
    so piping to a file or a CI log stays readable instead of filling with
    redraws.
    """

    def __init__(self, console: Console, *, live: bool = True) -> None:
        self.console = console
        self.enabled = live and console.is_terminal
        self._live: Live | None = None
        self._last_summary = ""

    def __enter__(self) -> RunDashboard:
        if self.enabled:
            self._live = Live(
                console=self.console,
                refresh_per_second=4,
                transient=False,
                vertical_overflow="visible",
            )
            self._live.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def stop(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def update(self, state: RunState) -> None:
        if self._live is not None:
            self._live.update(dashboard(state))
            return
        # Non-terminal: emit a line only when the picture actually changes.
        summary = f"{state.completed_stages()}/{len(state.ordered_stages)} " + " ".join(
            f"{stage.key}={stage.status}" for stage in state.ordered_stages
        )
        if summary != self._last_summary:
            self._last_summary = summary
            current = state.current_stage()
            self.console.print(
                f"progress: {state.completed_stages()}/{len(state.ordered_stages)} stages"
                f" • {format_duration(state.elapsed())}"
                + (f" • {current.label}: {current.status}" if current else ""),
                style="dim",
                highlight=False,
            )

    def pause(self) -> None:
        """Release the terminal so a prompt can be shown."""
        if self._live is not None:
            self._live.stop()

    def resume(self, state: RunState) -> None:
        if self.enabled and self._live is not None:
            self._live.start()
            self._live.update(dashboard(state))


def activity_line(event: dict[str, Any]) -> Text | None:
    """Render one logged event for `autoxgb watch`."""
    kind = event.get("type", "")
    if kind == "note":
        text = str(event.get("text", "")).strip()
        return Text(text, style="white") if text else None
    if kind == "stage_started":
        return Text(f"→ {event.get('stage')}", style="bold cyan")
    if kind == "stage_finished":
        ok = event.get("ok", True)
        return Text(
            f"{'✓' if ok else '✗'} {event.get('stage')}", style="green" if ok else "red"
        )
    if kind == "tool_started":
        return Text(f"  {event.get('tool')}…", style="dim")
    if kind == "tool_progress":
        return Text(f"  {event.get('tool')}: {event.get('message', '')}", style="dim")
    if kind == "tool_finished":
        ok = event.get("ok", True)
        duration = event.get("duration")
        suffix = f" ({float(duration):.1f}s)" if duration else ""
        return Text(
            f"  {event.get('tool')} {'ok' if ok else 'failed'}{suffix} — "
            f"{event.get('summary', '')}",
            style="dim" if ok else "red",
        )
    if kind == "artifact":
        return Text(f"  wrote {event.get('path')}", style="dim green")
    if kind == "approval_requested":
        return Text("⏸ waiting for approval", style="yellow")
    if kind == "approval_resolved":
        return Text(f"▶ plan {event.get('decision')}d", style="yellow")
    if kind == "run_finished":
        status = event.get("status", DONE)
        return Text(
            f"run {status}" + (f": {event.get('error')}" if event.get("error") else ""),
            style="green" if status == DONE else "red",
        )
    return None


def level_style(level: str) -> str:
    return _LEVEL_STYLES.get(level, "white")
