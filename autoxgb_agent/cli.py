"""`autoxgb` command line interface."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import typer
from langgraph.types import Command
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from autoxgb_agent.context import (
    MAX_TUNING_TIMEOUT_SECONDS,
    MAX_TUNING_TRIALS,
    RunContext,
    new_run_id,
    reset_run_context,
    set_run_context,
)
from autoxgb_agent.orchestrator import build_agent, initial_message, resolve_model

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Point AutoXGB at a dataset and a plain-English goal; get a tuned "
    "XGBoost pipeline, an evaluation report and a runnable inference bundle.",
)
console = Console()

# LangGraph's default (25) is far too low for a seven-stage delegated pipeline.
RECURSION_LIMIT = 400


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _render_messages(update: dict[str, Any], quiet: bool) -> None:
    for message in update.get("messages", []) or []:
        kind = getattr(message, "type", None)
        if kind == "ai":
            text = _text_of(message)
            if text.strip():
                console.print(text.strip(), style="white")
            for call in getattr(message, "tool_calls", []) or []:
                console.print(f"  → {call['name']}", style="cyan")
                if not quiet and call.get("args"):
                    console.print(_indent(_short_json(call["args"])), style="dim cyan")
        elif kind == "tool" and not quiet:
            name = getattr(message, "name", "tool")
            body = _text_of(message).strip()
            if body:
                console.print(f"  ← {name}", style="green")
                console.print(_indent(_truncate(body)), style="dim")


def _text_of(message: Any) -> str:
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


def _short_json(payload: Any, limit: int = 400) -> str:
    text = json.dumps(payload, default=str)
    return text if len(text) <= limit else text[:limit] + " …"


def _truncate(text: str, max_lines: int = 12) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    hidden = len(lines) - max_lines
    return "\n".join([*lines[:max_lines], f"… {hidden} more line(s)"])


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


# --------------------------------------------------------------------------- #
# Approval
# --------------------------------------------------------------------------- #


def _decide(requests: list[Any]) -> dict[str, Any]:
    """Show each pending action and collect the user's decisions."""
    decisions: list[dict[str, Any]] = []
    for request in requests:
        action_requests = _action_requests(request)
        for action in action_requests:
            decisions.append(_decide_one(action))
    return {"decisions": decisions}


def _action_requests(request: Any) -> list[dict[str, Any]]:
    value = getattr(request, "value", request)
    if isinstance(value, dict) and "action_requests" in value:
        return list(value["action_requests"])
    if isinstance(value, list):
        return [a for item in value for a in _action_requests(item)]
    return [value] if isinstance(value, dict) else []


def _decide_one(action: dict[str, Any]) -> dict[str, Any]:
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


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


@app.command()
def run(  # noqa: PLR0913 - a CLI entry point; each flag is user-facing
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
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Hide tool arguments and results; show narration only."
    ),
) -> None:
    """Build an XGBoost pipeline for DATASET."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print(
            "[red]ANTHROPIC_API_KEY is not set.[/red] Export it before running, or "
            "point --model at a provider you are authenticated for.",
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

    console.print(
        Panel(
            f"[bold]{goal}[/bold]\n"
            f"dataset : {dataset}\n"
            f"run dir : {run_dir}\n"
            f"model   : {resolve_model(model)}\n"
            f"budget  : {context.max_tuning_trials} trials / "
            f"{context.max_tuning_timeout_seconds}s tuning"
            + ("\napproval: skipped (--yes)" if yes else ""),
            title="[bold]AutoXGB[/bold]",
            border_style="blue",
        )
    )

    try:
        agent = build_agent(run_dir, model=model, require_approval=not yes)
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
                        _render_messages(update, quiet)
            if pending is None:
                break
            payload = Command(resume=_decide(pending))

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Partial artifacts are in "
                      f"{run_dir}[/yellow]")
        raise typer.Exit(code=130) from None
    finally:
        reset_run_context(token)

    _print_outcome(run_dir)


def _print_outcome(run_dir: Path) -> None:
    bundle = run_dir / "bundle"
    lines = [f"run directory : {run_dir}"]
    for label, name in (
        ("profile      ", "profile_report.md"),
        ("model report ", "model_report.md"),
        ("metrics      ", "metrics.json"),
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
    """List previous runs and what each produced."""
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
        rows.append((directory.name, target, task_type, score, packaged))

    if not rows:
        console.print(f"No runs found under {output_dir}.")
        raise typer.Exit()

    from rich.table import Table

    table = Table(title=str(output_dir))
    for column in ("run", "target", "task", "test metric", "bundle"):
        table.add_column(column)
    for row in rows:
        table.add_row(*row)
    console.print(table)


def main() -> None:  # pragma: no cover - console-script shim
    sys.exit(app())


if __name__ == "__main__":  # pragma: no cover
    main()
