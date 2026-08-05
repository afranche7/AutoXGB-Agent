"""The seven pipeline specialists.

Each one owns a narrow slice of tools, so the model choosing what to do next only
ever sees the handful of options that make sense at that stage. Filesystem tools
(`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`) are supplied to
every subagent by DeepAgents' `FilesystemMiddleware` and are not listed here.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from deepagents import SubAgent

from autoxgb_agent.tools import (
    DATA_TOOLS,
    EVALUATE_TOOLS,
    FEATURE_TOOLS,
    PACKAGE_TOOLS,
    PLAN_TOOLS,
    TRAIN_TOOLS,
    TUNE_TOOLS,
)

_PROMPTS = Path(__file__).resolve().parent / "prompts"

# The tool whose call pauses for human approval. It is the target/task decision:
# every later stage is built on it, and a wrong answer here fails silently rather
# than loudly.
APPROVAL_TOOL = "set_task_plan"


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    path = _PROMPTS / f"{name}.md"
    if not path.exists():
        msg = f"prompt file not found: {path}"
        raise FileNotFoundError(msg)
    return path.read_text(encoding="utf-8")


def _spec(name: str, description: str, prompt: str, tools: list[Any]) -> SubAgent:
    return {
        "name": name,
        "description": description,
        "system_prompt": load_prompt(prompt),
        "tools": tools,
    }


def build_subagents() -> list[SubAgent]:
    """The pipeline, in the order the orchestrator should run it."""
    return [
        _spec(
            "data-profiler",
            "Profile the raw dataset: dtypes, missingness, cardinality, class "
            "balance, candidate targets and near-duplicate columns. Run first, "
            "before any decision about the data.",
            "profiler",
            DATA_TOOLS,
        ),
        _spec(
            "target-detector",
            "Decide the target column and task type from the user's goal, screen "
            "the target for leakage, and get the plan approved by the user. Run "
            "after profiling and before any feature work.",
            "target_detector",
            PLAN_TOOLS,
        ),
        _spec(
            "feature-engineer",
            "Assign columns to numeric/categorical/datetime features, choose "
            "encodings and split sizes, and materialise the train/val/test sets. "
            "Run after the task plan is approved.",
            "feature_engineer",
            FEATURE_TOOLS,
        ),
        _spec(
            "modeler",
            "Train the baseline XGBoost model with sensible hyperparameters and "
            "early stopping. Run after preprocessing.",
            "modeler",
            TRAIN_TOOLS,
        ),
        _spec(
            "tuner",
            "Run a budget-capped Optuna hyperparameter search and refit the best "
            "configuration. Run after the baseline exists.",
            "tuner",
            TUNE_TOOLS,
        ),
        _spec(
            "evaluator",
            "Pick the better of the baseline and tuned models, compute metrics, "
            "feature importance and SHAP, and write the model report. Run after "
            "training and tuning.",
            "evaluator",
            EVALUATE_TOOLS,
        ),
        _spec(
            "packager",
            "Export the standalone inference bundle: model, preprocessor, "
            "predict.py, pyproject.toml and README. Run last.",
            "packager",
            PACKAGE_TOOLS,
        ),
    ]
