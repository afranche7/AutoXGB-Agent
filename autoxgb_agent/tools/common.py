"""Shared helpers for the ML tools."""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
from pydantic import BaseModel, ValidationError

from autoxgb_agent.context import RunContext, get_run_context

T = TypeVar("T", bound=BaseModel)


class ToolFailure(Exception):
    """A failure the agent is expected to read and recover from."""


def format_validation_error(exc: ValidationError, tool_name: str = "this tool") -> str:
    """Render a pydantic error as instructions the agent can act on."""
    lines = [
        f"- {'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
        for err in exc.errors()
    ]
    return (
        f"INVALID CONFIG for {tool_name}. Fix these fields and call it again:\n"
        + "\n".join(lines)
    )


def harden(tools: list[Any]) -> list[Any]:
    """Make argument-validation failures readable instead of fatal.

    LangChain validates tool arguments *before* the function body runs, so
    `guard` never sees those errors. Without this, a mis-filled config surfaces
    as a raw pydantic traceback (or, depending on the tool node, kills the run).
    """
    for tool in tools:
        tool.handle_validation_error = functools.partial(
            format_validation_error, tool_name=tool.name
        )
    return tools


def guard(fn: Callable[..., str]) -> Callable[..., str]:
    """Turn exceptions raised inside a tool into tool text.

    A tool that raises kills the run; a tool that *reports* lets the agent try
    something else. Argument validation is handled separately by `harden`; this
    covers everything that goes wrong once the body is running.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        try:
            return fn(*args, **kwargs)
        except ValidationError as exc:
            return format_validation_error(exc, fn.__name__)
        except ToolFailure as exc:
            return f"ERROR in {fn.__name__}: {exc}"
        except FileNotFoundError as exc:
            return (
                f"ERROR in {fn.__name__}: required artifact missing ({exc}). "
                "An earlier pipeline stage has not run yet."
            )
        except Exception as exc:  # noqa: BLE001 - deliberately broad; see docstring
            return f"ERROR in {fn.__name__}: {type(exc).__name__}: {exc}"

    return wrapper


def validate(model: type[T], value: Any) -> T:
    """Accept either a dict from the model or an already-built instance."""
    if isinstance(value, model):
        return value
    return model.model_validate(value)


def ctx() -> RunContext:
    return get_run_context()


def load_raw_dataset() -> pd.DataFrame:
    """Load the run's dataset from CSV or Parquet."""
    path = ctx().dataset_path
    if not path.exists():
        msg = f"dataset not found at {path}"
        raise ToolFailure(msg)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt", ""}:
        return pd.read_csv(path)
    if suffix in {".tsv"}:
        return pd.read_csv(path, sep="\t")
    msg = f"unsupported dataset format {suffix!r}; use .csv, .tsv or .parquet"
    raise ToolFailure(msg)


def write_json(relpath: str, payload: Any) -> Path:
    target = ctx().path(relpath)
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return target


def read_json(relpath: str) -> Any:
    target = ctx().run_dir / relpath
    if not target.exists():
        raise FileNotFoundError(target)
    return json.loads(target.read_text(encoding="utf-8"))


def write_text(relpath: str, text: str) -> Path:
    target = ctx().path(relpath)
    target.write_text(text, encoding="utf-8")
    return target


def fmt(value: Any, digits: int = 4) -> str:
    """Compact float formatting for report tables."""
    if isinstance(value, float):
        if value != value:  # NaN
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)
