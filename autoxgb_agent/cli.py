"""`autoxgb` command line interface."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import typer
from langgraph.types import Command
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from autoxgb_agent.console import (
    MessageRenderer,
    RunDashboard,
    StreamTracker,
    activity_line,
)
from autoxgb_agent.context import (
    MAX_TUNING_TIMEOUT_SECONDS,
    MAX_TUNING_TRIALS,
    RunContext,
    new_run_id,
    reset_run_context,
    set_run_context,
)
from autoxgb_agent.orchestrator import build_agent, initial_message, resolve_model
from autoxgb_agent.progress import (
    DONE,
    EVENTS_FILE,
    FAILED,
    ApprovalTimeout,
    ProgressReader,
    ProgressRecorder,
    await_approval_response,
    clear_approval,
    format_duration,
    read_approval_request,
    read_state_file,
    reset_recorder,
    set_recorder,
    write_approval_request,
    write_approval_response,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Point AutoXGB at a dataset and a plain-English goal; get a tuned "
    "XGBoost pipeline, an evaluation report and a runnable inference bundle.",
)
console = Console()

# LangGraph's default (25) is far too low for a seven-stage delegated pipeline.
RECURSION_LIMIT = 400

# How a run gets its one human decision answered.
APPROVE_PROMPT = "prompt"  # ask on this terminal
APPROVE_FILE = "file"  # publish it for the UI (or `autoxgb watch --approve`)
APPROVE_AUTO = "auto"  # approve whatever the agent proposes
APPROVAL_MODES = (APPROVE_PROMPT, APPROVE_FILE, APPROVE_AUTO)


# --------------------------------------------------------------------------- #
# Approval
# --------------------------------------------------------------------------- #


def _action_requests(request: Any) -> list[dict[str, Any]]:
    value = getattr(request, "value", request)
    if isinstance(value, dict) and "action_requests" in value:
        return list(value["action_requests"])
    if isinstance(value, list):
        return [a for item in value for a in _action_requests(item)]
    return [value] if isinstance(value, dict) else []


def _decide_one(action: dict[str, Any]) -> dict[str, Any]:
    """Show one pending action and collect the user's decision."""
    args = action.get("args", {})
    plan = args.get("plan", args)
    body = json.dumps(plan, indent=2, default=str)

    console.print()
    console.print(
        Panel(
            Syntax(body, "json", theme="ansi_dark", word_wrap=True),
            title=f"[bold]Approval needed — {action.get('name', 'action')}[/bold]",
            subtitle="every later stage depends on this",
            border_style="yellow",
        )
    )
    if isinstance(plan, dict) and plan.get("reasoning"):
        console.print(f"[bold]Why:[/bold] {plan['reasoning']}")

    console.print()
    choice = typer.prompt(
        "Approve this plan? [y]es / [n]o (give feedback)", default="y"
    ).strip().lower()
    if choice.startswith("y"):
        console.print("[green]Approved.[/green]")
        return {"type": "approve"}

    feedback = typer.prompt(
        "What should change? (e.g. 'the target should be `renewed`, not `churned`')"
    ).strip()
    console.print("[yellow]Rejected — sending your feedback back to the agent.[/yellow]")
    return {
        "type": "reject",
        "message": feedback or "The plan was rejected. Reconsider the target and task type.",
    }


def _decide(
    requests: list[Any],
    *,
    mode: str,
    recorder: ProgressRecorder,
    dashboard: RunDashboard | None = None,
) -> dict[str, Any]:
    """Resolve every pending action, however this run collects approvals."""
    actions = [action for request in requests for action in _action_requests(request)]
    request_id = recorder.approval_requested(actions)
    if dashboard is not None:
        dashboard.update(recorder.state)

    if mode == APPROVE_AUTO:
        decisions = [{"type": "approve"} for _ in actions]
    elif mode == APPROVE_FILE:
        decisions = _decide_via_file(actions, request_id, recorder)
    else:
        if dashboard is not None:
            dashboard.pause()
        try:
            decisions = [_decide_one(action) for action in actions]
        finally:
            if dashboard is not None:
                dashboard.resume(recorder.state)

    first = decisions[0] if decisions else {"type": "approve"}
    recorder.approval_resolved(str(first.get("type", "approve")), str(first.get("message", "")))
    return {"decisions": decisions}


def _decide_via_file(
    actions: list[dict[str, Any]], request_id: str, recorder: ProgressRecorder
) -> list[dict[str, Any]]:
    write_approval_request(recorder.run_dir, request_id, actions)
    console.print(
        "[yellow]Waiting for approval[/yellow] — answer it in the UI, or run "
        f"`autoxgb watch {recorder.run_dir.name} -o {recorder.run_dir.parent} "
        "--approve` in another terminal."
    )
    return await_approval_response(recorder.run_dir, request_id)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


@app.command()
def run(  # noqa: PLR0913, PLR0912, PLR0915 - a CLI entry point; each flag is user-facing
    dataset: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True, help="CSV, TSV or Parquet file."
    ),
    goal: str = typer.Option(
        ..., "--goal", "-g", help="What to predict, in plain English."
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Name this run."),
    output_dir: Path = typer.Option(
        Path("outputs"), "--output-dir", "-o", help="Where run directories are created."
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", help="provider:model string. Defaults to AUTOXGB_MODEL."
    ),
    tuning_trials: int = typer.Option(
        MAX_TUNING_TRIALS, "--tuning-trials", min=1, help="Cap on Optuna trials."
    ),
    tuning_timeout: int = typer.Option(
        MAX_TUNING_TIMEOUT_SECONDS,
        "--tuning-timeout",
        min=10,
        help="Cap on tuning wall-clock seconds.",
    ),
    seed: int = typer.Option(42, "--seed", help="Random state for splits and training."),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the task-plan approval prompt (unattended runs)."
    ),
    approve_via: str = typer.Option(
        APPROVE_PROMPT,
        "--approve-via",
        help="How to answer the approval gate: prompt (this terminal), file "
        "(the UI or `autoxgb watch --approve`), or auto.",
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Hide tool arguments and results; show narration only."
    ),
    plain: bool = typer.Option(
        False, "--plain", help="Disable the live dashboard; log progress line by line."
    ),
) -> None:
    """Build an XGBoost pipeline for DATASET."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print(
            "[red]ANTHROPIC_API_KEY is not set.[/red] Export it before running, or "
            "point --model at a provider you are authenticated for.",
        )
        raise typer.Exit(code=2)

    if approve_via not in APPROVAL_MODES:
        console.print(
            f"[red]--approve-via must be one of {', '.join(APPROVAL_MODES)}.[/red]"
        )
        raise typer.Exit(code=2)

    mode = APPROVE_AUTO if yes else approve_via
    if mode == APPROVE_PROMPT and not sys.stdin.isatty():
        console.print(
            "[red]Nothing can answer the approval prompt[/red] — stdin is not a "
            "terminal. Use --yes for an unattended run, or --approve-via file to "
            "answer it from the UI or `autoxgb watch --approve`."
        )
        raise typer.Exit(code=2)

    dataset = dataset.resolve()
    resolved_run_id = run_id or new_run_id()
    run_dir = (output_dir / resolved_run_id).resolve()

    context = RunContext(
        dataset_path=dataset,
        run_dir=run_dir,
        goal=goal,
        run_id=resolved_run_id,
        max_tuning_trials=min(tuning_trials, MAX_TUNING_TRIALS),
        max_tuning_timeout_seconds=min(tuning_timeout, MAX_TUNING_TIMEOUT_SECONDS),
        random_state=seed,
    )
    token = set_run_context(context)

    resolved_model = resolve_model(model)
    console.print(
        Panel(
            f"[bold]{goal}[/bold]\n"
            f"dataset : {dataset}\n"
            f"run dir : {run_dir}\n"
            f"model   : {resolved_model}\n"
            f"budget  : {context.max_tuning_trials} trials / "
            f"{context.max_tuning_timeout_seconds}s tuning\n"
            f"approval: {mode}\n"
            f"watch   : autoxgb watch {resolved_run_id} -o {output_dir}",
            title="[bold]AutoXGB[/bold]",
            border_style="blue",
        )
    )

    recorder, progress_token = start_recording(
        run_dir,
        run_id=resolved_run_id,
        goal=goal,
        dataset=str(dataset),
        model=str(resolved_model),
    )
    tracker = StreamTracker(recorder)
    renderer = MessageRenderer(console, quiet=quiet)
    status, error = DONE, None

    try:
        with RunDashboard(console, live=not plain) as dashboard:
            agent = build_agent(run_dir, model=model, require_approval=mode != APPROVE_AUTO)
            config = {
                "configurable": {"thread_id": resolved_run_id},
                "recursion_limit": RECURSION_LIMIT,
            }
            payload: Any = {
                "messages": [
                    {
                        "role": "user",
                        "content": initial_message(dataset, goal, resolved_run_id),
                    }
                ]
            }

            while True:
                pending: list[Any] | None = None
                for chunk in agent.stream(payload, config, stream_mode="updates"):
                    if not isinstance(chunk, dict):
                        continue
                    if "__interrupt__" in chunk:
                        pending = list(chunk["__interrupt__"])
                        break
                    for update in chunk.values():
                        if isinstance(update, dict):
                            renderer.render(update)
                            tracker.consume(update)
                            dashboard.update(recorder.state)
                if pending is None:
                    break
                payload = Command(
                    resume=_decide(
                        pending, mode=mode, recorder=recorder, dashboard=dashboard
                    )
                )
            dashboard.update(recorder.state)

    except KeyboardInterrupt:
        status, error = "interrupted", "interrupted by the user"
        console.print(f"\n[yellow]Interrupted. Partial artifacts are in {run_dir}[/yellow]")
        raise typer.Exit(code=130) from None
    except ApprovalTimeout as exc:
        status, error = FAILED, str(exc)
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    except Exception as exc:
        status, error = FAILED, f"{type(exc).__name__}: {exc}"
        raise
    finally:
        recorder.run_finished(status=status, error=error)
        clear_approval(run_dir)
        finish_recording(progress_token)
        reset_run_context(token)

    _print_outcome(run_dir)


def start_recording(
    run_dir: Path, *, run_id: str, goal: str, dataset: str, model: str
) -> tuple[ProgressRecorder, Any]:
    """Open a run's progress log and install it for the tools to report into."""
    recorder = ProgressRecorder(run_dir)
    clear_approval(run_dir)
    progress_token = set_recorder(recorder)
    recorder.run_started(run_id=run_id, goal=goal, dataset=dataset, model=model)
    return recorder, progress_token


def finish_recording(progress_token: Any) -> None:
    reset_recorder(progress_token)


def _print_outcome(run_dir: Path) -> None:
    bundle = run_dir / "bundle"
    lines = [f"run directory : {run_dir}"]
    for label, name in (
        ("profile      ", "profile_report.md"),
        ("model report ", "model_report.md"),
        ("metrics      ", "metrics.json"),
        ("progress log ", EVENTS_FILE),
    ):
        if (run_dir / name).exists():
            lines.append(f"{label}: {run_dir / name}")

    if (bundle / "predict.py").exists():
        lines.append("")
        lines.append("score new data with:")
        lines.append(f"  cd {bundle}")
        lines.append("  uv sync && uv run predict.py /path/to/new_data.csv -o preds.csv")
        style = "green"
        title = "[bold]Done[/bold]"
    else:
        lines.append("")
        lines.append("[yellow]No inference bundle was produced — the run did not "
                     "reach the packaging stage.[/yellow]")
        style = "yellow"
        title = "[bold]Incomplete[/bold]"

    console.print(Panel("\n".join(lines), title=title, border_style=style))


@app.command("runs")
def list_runs(
    output_dir: Path = typer.Option(
        Path("outputs"), "--output-dir", "-o", help="Where run directories live."
    ),
) -> None:
    """List previous runs, how far each got and what it produced."""
    if not output_dir.exists():
        console.print(f"No runs yet — {output_dir} does not exist.")
        raise typer.Exit()

    rows = []
    for directory in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        plan_path = directory / "task_plan.json"
        selected_path = directory / "selected_model.json"
        target = task_type = score = "-"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            target, task_type = plan.get("target", "-"), plan.get("task_type", "-")
        if selected_path.exists():
            selected = json.loads(selected_path.read_text(encoding="utf-8"))
            value = selected.get("test_primary_metric")
            score = f"{value:.4f}" if isinstance(value, (int, float)) else "-"
        packaged = "yes" if (directory / "bundle" / "predict.py").exists() else "no"

        state = read_state_file(directory) or {}
        status = state.get("status", "-")
        stages = (
            f"{state.get('completed_stages', 0)}/{state.get('total_stages', 7)}"
            if state
            else "-"
        )
        elapsed = format_duration(state.get("elapsed")) if state.get("elapsed") else "-"
        rows.append(
            (directory.name, status, stages, elapsed, target, task_type, score, packaged)
        )

    if not rows:
        console.print(f"No runs found under {output_dir}.")
        raise typer.Exit()

    from rich.table import Table

    table = Table(title=str(output_dir))
    for column in (
        "run",
        "status",
        "stages",
        "time",
        "target",
        "task",
        "test metric",
        "bundle",
    ):
        table.add_column(column)
    for row in rows:
        table.add_row(*row)
    console.print(table)


@app.command()
def watch(
    run_id: str | None = typer.Argument(
        None, help="Which run to watch. Defaults to the most recent one."
    ),
    output_dir: Path = typer.Option(
        Path("outputs"), "--output-dir", "-o", help="Where run directories live."
    ),
    approve: bool = typer.Option(
        False, "--approve", help="Answer approval requests from this terminal."
    ),
    follow: bool = typer.Option(
        True, "--follow/--no-follow", help="Keep watching until the run finishes."
    ),
    refresh: float = typer.Option(
        0.5, "--refresh", min=0.05, help="Seconds between polls of the progress log."
    ),
) -> None:
    """Watch a run's progress — live, or replayed after the fact.

    Reads the run's progress log, so it works while another terminal (or the UI)
    is driving the run, and equally on a run that finished yesterday.
    """
    run_dir = _resolve_run_dir(output_dir, run_id)
    state_source = ProgressReader(run_dir)
    # The run clears the handshake once it reads our answer; until then the
    # request file is still there, and must not be prompted for twice.
    answered: set[str] = set()

    console.print(
        Panel(
            f"watching [bold]{run_dir.name}[/bold]\n{run_dir}",
            border_style="blue",
            title="[bold]AutoXGB[/bold]",
        )
    )

    try:
        with RunDashboard(console, live=follow) as board:
            while True:
                for event in state_source.poll():
                    line = activity_line(event)
                    if line is not None:
                        console.print(line)
                board.update(state_source.state)

                if (
                    approve
                    and state_source.state.approval is not None
                    and _answer_pending_approval(run_dir, board, state_source, answered)
                ):
                    continue

                if not follow or state_source.state.is_finished:
                    break
                time.sleep(refresh)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped watching. The run itself is untouched.[/yellow]")
        raise typer.Exit(code=130) from None

    state = state_source.state
    console.print(
        f"[dim]{state.completed_stages()}/{len(state.ordered_stages)} stages • "
        f"{format_duration(state.elapsed())} • {state.status}[/dim]"
    )


def _answer_pending_approval(
    run_dir: Path, board: RunDashboard, reader: ProgressReader, answered: set[str]
) -> bool:
    """Answer the approval the watched run is blocked on.

    Returns False when there is nothing to answer from here — the run is asking
    on its own terminal, somebody else got there first, or we have already
    answered this one and the run has yet to pick it up.
    """
    request = read_approval_request(run_dir)
    if request is None or str(request.get("id", "")) in answered:
        return False
    answered.add(str(request.get("id", "")))
    board.pause()
    try:
        decisions = [_decide_one(action) for action in request.get("actions") or []]
    finally:
        board.resume(reader.state)
    write_approval_response(run_dir, str(request.get("id", "")), decisions)
    return True


def _resolve_run_dir(output_dir: Path, run_id: str | None) -> Path:
    if run_id:
        run_dir = output_dir / run_id
        if not run_dir.is_dir():
            console.print(f"[red]No run directory at {run_dir}.[/red]")
            raise typer.Exit(code=2)
        return run_dir

    if not output_dir.exists():
        console.print(f"[red]No runs yet — {output_dir} does not exist.[/red]")
        raise typer.Exit(code=2)
    candidates = [p for p in output_dir.iterdir() if (p / EVENTS_FILE).exists()]
    if not candidates:
        console.print(
            f"[red]No run under {output_dir} has a progress log to watch.[/red]"
        )
        raise typer.Exit(code=2)
    return max(candidates, key=lambda p: (p / EVENTS_FILE).stat().st_mtime)


@app.command()
def ui(
    output_dir: Path = typer.Option(
        Path("outputs"), "--output-dir", "-o", help="Where run directories live."
    ),
    port: int = typer.Option(8501, "--port", help="Port to serve the UI on."),
    headless: bool = typer.Option(
        False, "--headless", help="Do not open a browser window."
    ),
) -> None:
    """Launch the web UI: start runs, watch every stage, approve the plan."""
    if shutil.which("streamlit") is None:
        try:
            import streamlit  # noqa: F401
        except ImportError:
            console.print(
                "[red]Streamlit is not installed.[/red] Install the UI extra:\n"
                "  uv sync --extra ui"
            )
            raise typer.Exit(code=2) from None

    app_path = Path(__file__).resolve().parent / "ui" / "app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        "--server.headless",
        "true" if headless else "false",
        "--",
        "--output-dir",
        str(output_dir.resolve()),
    ]
    console.print(f"[dim]{' '.join(command)}[/dim]")
    raise typer.Exit(code=subprocess.call(command))


def main() -> None:  # pragma: no cover - console-script shim
    sys.exit(app())


if __name__ == "__main__":  # pragma: no cover
    main()
