"""Per-run context shared by every tool.

The agent decides *what* to do; it never supplies file paths. The dataset
location, the run directory and the tuning budget are set once by the CLI and
read by the tools through this module, so a hallucinated path can't reach the
filesystem and tool signatures stay small enough for the model to fill in
reliably.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from pathlib import Path

# Hard ceilings. The agent can ask for less; it can never get more than the CLI
# allows, and the CLI can never exceed these.
MAX_TUNING_TRIALS = 50
MAX_TUNING_TIMEOUT_SECONDS = 600


@dataclass(frozen=True)
class RunContext:
    """Everything a tool needs to know about the current run."""

    dataset_path: Path
    run_dir: Path
    goal: str
    run_id: str
    max_tuning_trials: int = MAX_TUNING_TRIALS
    max_tuning_timeout_seconds: int = MAX_TUNING_TIMEOUT_SECONDS
    random_state: int = 42

    def path(self, *parts: str) -> Path:
        """Resolve a path inside the run directory, creating parent dirs."""
        target = self.run_dir.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target


_run_context: contextvars.ContextVar[RunContext | None] = contextvars.ContextVar(
    "autoxgb_run_context", default=None
)


class RunContextError(RuntimeError):
    """Raised when a tool runs outside an active run."""


def set_run_context(ctx: RunContext) -> contextvars.Token[RunContext | None]:
    """Install `ctx` as the active run context. Returns a reset token."""
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    return _run_context.set(ctx)


def reset_run_context(token: contextvars.Token[RunContext | None]) -> None:
    """Restore the previous run context."""
    _run_context.reset(token)


def get_run_context() -> RunContext:
    """Return the active run context, or raise if there isn't one."""
    ctx = _run_context.get()
    if ctx is None:
        msg = (
            "No active AutoXGB run. Tools must be called inside a run started by "
            "the CLI (or a test that calls set_run_context)."
        )
        raise RunContextError(msg)
    return ctx


def new_run_id() -> str:
    """Timestamp-based run id, sortable and filesystem-safe."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")
