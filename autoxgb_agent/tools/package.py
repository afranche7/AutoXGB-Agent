"""Export a standalone inference bundle."""

from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import Any

from langchain_core.tools import tool

from autoxgb_agent import preprocess as preprocess_module
from autoxgb_agent.schemas import FeatureSpec, primary_metric
from autoxgb_agent.tools.common import (
    ToolFailure,
    ctx,
    fmt,
    guard,
    read_json,
    validate,
    write_json,
)

_TEMPLATES = Path(__file__).resolve().parent.parent / "templates"


def _render(name: str, mapping: dict[str, str]) -> str:
    template = Template((_TEMPLATES / name).read_text(encoding="utf-8"))
    return template.safe_substitute(mapping)


def _bundle_name(target: str, run_id: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", f"{target}-{run_id}".lower()).strip("-")
    return f"autoxgb-{slug}" if slug else "autoxgb-model"


def _resolve_selection(task_type: str) -> dict[str, Any]:
    try:
        return read_json("selected_model.json")
    except FileNotFoundError:
        pass
    # Evaluation is the normal path here, but don't hard-fail a run that skipped
    # it — fall back to whatever model exists, preferring the tuned one.
    for stage in ("tuned", "baseline"):
        if (ctx().run_dir / f"model_{stage}.json").exists():
            return {"stage": stage, "model_file": f"model_{stage}.json", "reason": "fallback"}
    msg = "no trained model found. Run train_xgboost (and evaluate_model) first."
    raise ToolFailure(msg)


@tool
@guard
def export_bundle() -> str:
    """Export a self-contained inference bundle into `bundle/`.

    Copies the selected model, the fitted preprocessor state and the exact
    preprocessing code used at training time, then generates `predict.py`, a
    `pyproject.toml` pinning only the runtime dependencies, and a README. The
    result runs with `uv run predict.py input.csv` and imports nothing from
    `autoxgb_agent`. Run this last.
    """
    run = ctx()
    meta = read_json("processed/metadata.json")
    plan = read_json("task_plan.json")
    spec = validate(FeatureSpec, read_json("feature_spec.json"))
    task_type = meta["task_type"]
    selection = _resolve_selection(task_type)
    stage = selection["stage"]

    bundle = run.run_dir / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)

    model_src = run.run_dir / f"model_{stage}.json"
    if not model_src.exists():
        msg = f"{model_src.name} is missing; cannot package the {stage} model."
        raise ToolFailure(msg)
    shutil.copyfile(model_src, bundle / "model.json")
    shutil.copyfile(run.run_dir / "preprocessor.json", bundle / "preprocessor.json")
    # The bundle carries its own copy of the transform, so it never imports us
    # and never breaks when this package changes.
    shutil.copyfile(Path(preprocess_module.__file__), bundle / "preprocess.py")

    required_columns = spec.feature_names()
    inference_metadata: dict[str, Any] = {
        "run_id": run.run_id,
        "goal": run.goal,
        "stage": stage,
        "target": plan["target"],
        "task_type": task_type,
        "required_input_columns": required_columns,
        "model_feature_columns": meta["feature_columns"],
    }
    if meta.get("classes"):
        inference_metadata["classes"] = meta["classes"]
    write_json("bundle/inference_metadata.json", inference_metadata)

    metric = primary_metric(task_type)
    score = selection.get("test_primary_metric")
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    mapping = {
        "BUNDLE_NAME": _bundle_name(plan["target"], run.run_id),
        "RUN_ID": run.run_id,
        "TARGET": plan["target"],
        "TASK_TYPE": task_type,
        "GENERATED_AT": generated_at,
        "STAGE": stage,
        "PRIMARY_METRIC": metric,
        "PRIMARY_SCORE": fmt(score) if score is not None else "not evaluated",
        "REQUIRED_COLUMNS": "\n".join(f"- `{c}`" for c in required_columns),
    }
    (bundle / "predict.py").write_text(_render("predict.py.tmpl", mapping), encoding="utf-8")
    (bundle / "pyproject.toml").write_text(
        _render("pyproject.toml.tmpl", mapping), encoding="utf-8"
    )
    (bundle / "README.md").write_text(_render("README.md.tmpl", mapping), encoding="utf-8")

    files = sorted(p.name for p in bundle.iterdir())
    write_json(
        "bundle_manifest.json",
        {"stage": stage, "files": files, "generated_at": generated_at, **inference_metadata},
    )

    return (
        f"Bundle exported to bundle/ ({stage} model).\n"
        f"  files: {', '.join(files)}\n"
        f"  needs {len(required_columns)} raw input column(s): "
        f"{', '.join(required_columns)}\n"
        f"  run it with: cd {bundle} && uv sync && "
        f"uv run predict.py /path/to/new_data.csv -o preds.csv"
    )


PACKAGE_TOOLS = [export_bundle]
