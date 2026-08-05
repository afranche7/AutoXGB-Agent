"""Evaluation, explainability and the final model report."""

from __future__ import annotations

from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")  # No display in a CLI run; must precede pyplot.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from autoxgb_agent.modeling import (  # noqa: E402
    compute_metrics,
    format_metrics,
    load_processed_metadata,
    load_split,
    predict_frames,
)
from autoxgb_agent.schemas import (  # noqa: E402
    greater_is_better,
    is_classification,
    primary_metric,
)
from autoxgb_agent.tools.common import (  # noqa: E402
    ToolFailure,
    ctx,
    fmt,
    guard,
    read_json,
    write_json,
    write_text,
)

# SHAP is exact for trees but still O(rows); cap it so a large test set can't
# turn the report step into the slowest part of the run.
_MAX_SHAP_ROWS = 2000
_TOP_FEATURES = 20

ModelStage = Literal["baseline", "tuned", "best"]


def _load_booster(stage: str, task_type: str):
    from xgboost import XGBClassifier, XGBRegressor

    path = ctx().run_dir / f"model_{stage}.json"
    if not path.exists():
        msg = f"model_{stage}.json does not exist. Train (or tune) before evaluating."
        raise ToolFailure(msg)
    model = XGBClassifier() if is_classification(task_type) else XGBRegressor()
    model.load_model(str(path))
    return model


def _stage_test_score(relpath: str, task_type: str) -> float | None:
    try:
        payload = read_json(relpath)
    except FileNotFoundError:
        return None
    return payload.get("metrics", {}).get("test", {}).get(primary_metric(task_type))


def _resolve_stage(stage: ModelStage, task_type: str) -> tuple[str, str]:
    """Return (stage_name, why) for the model to report on."""
    if stage != "best":
        return stage, f"explicitly requested ({stage})"

    baseline = _stage_test_score("metrics_baseline.json", task_type)
    tuned = _stage_test_score("metrics_tuned.json", task_type)
    metric = primary_metric(task_type)

    if tuned is None and baseline is None:
        msg = "no trained model found. Run train_xgboost first."
        raise ToolFailure(msg)
    if tuned is None:
        return "baseline", "only the baseline has been trained"
    if baseline is None:
        return "tuned", "only the tuned model has been trained"

    tuned_wins = tuned > baseline if greater_is_better(task_type) else tuned < baseline
    if tuned_wins:
        return "tuned", f"tuned test {metric} {tuned:.4f} beats baseline {baseline:.4f}"
    return (
        "baseline",
        f"tuned test {metric} {tuned:.4f} did not beat baseline {baseline:.4f}; "
        "keeping the simpler model",
    )


def _importance_plot(model, feature_names: list[str]) -> list[dict[str, Any]]:
    booster = model.get_booster()
    booster.feature_names = list(feature_names)
    gains = booster.get_score(importance_type="gain")
    ranked = sorted(gains.items(), key=lambda kv: -kv[1])
    top = ranked[:_TOP_FEATURES]

    if top:
        labels = [name for name, _ in reversed(top)]
        values = [value for _, value in reversed(top)]
        height = max(3.0, 0.32 * len(top) + 1.2)
        fig, ax = plt.subplots(figsize=(9, height))
        ax.barh(labels, values, color="#2b6cb0")
        ax.set_xlabel("Gain")
        ax.set_title(f"Top {len(top)} features by gain")
        fig.tight_layout()
        fig.savefig(ctx().path("feature_importance.png"), dpi=140)
        plt.close(fig)

    return [{"feature": name, "gain": round(float(value), 6)} for name, value in ranked]


def _shap_plot(model, x_test: pd.DataFrame) -> str:
    """Write shap_summary.png. Returns a status line for the report."""
    try:
        import shap
    except ImportError:  # pragma: no cover - shap is a hard dependency
        return "SHAP unavailable (package not installed)."

    sample = x_test
    if len(sample) > _MAX_SHAP_ROWS:
        sample = sample.sample(_MAX_SHAP_ROWS, random_state=ctx().random_state)
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(sample)
        # Multiclass returns one matrix per class; average magnitude across them.
        if isinstance(values, list):
            values = np.mean([np.abs(v) for v in values], axis=0)
        elif getattr(values, "ndim", 2) == 3:
            values = np.mean(np.abs(values), axis=2)
        plt.figure()
        shap.summary_plot(values, sample, show=False, max_display=_TOP_FEATURES)
        fig = plt.gcf()
        fig.set_size_inches(9, max(4.0, 0.3 * min(len(sample.columns), _TOP_FEATURES) + 2))
        fig.tight_layout()
        fig.savefig(ctx().path("shap_summary.png"), dpi=140)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001 - explainability is best-effort
        return f"SHAP plot skipped ({type(exc).__name__}: {exc})."
    return f"SHAP summary over {len(sample):,} test rows: `shap_summary.png`."


def _render_report(
    *,
    stage: str,
    why: str,
    task_type: str,
    meta: dict[str, Any],
    plan: dict[str, Any],
    scores: dict[str, dict[str, float]],
    importance: list[dict[str, Any]],
    confusion: dict[str, Any] | None,
    shap_note: str,
    params: dict[str, Any],
) -> str:
    run = ctx()
    metric = primary_metric(task_type)
    lines = [
        "# Model report",
        "",
        f"- **Run**: `{run.run_id}`",
        f"- **Goal**: {run.goal}",
        f"- **Target**: `{plan['target']}` ({task_type})",
        f"- **Selected model**: `{stage}` — {why}",
        f"- **Primary metric**: {metric} = **{fmt(scores['test'].get(metric))}** (test)",
        f"- **Features**: {meta['n_features']} model columns",
        f"- **Rows**: train {meta['split_sizes']['train']:,} / "
        f"val {meta['split_sizes']['val']:,} / test {meta['split_sizes']['test']:,}",
        "",
        "## Metrics",
        "",
    ]
    all_keys = sorted({k for split in scores.values() for k in split})
    lines.append("| metric | train | val | test |")
    lines.append("| --- | --- | --- | --- |")
    for key in all_keys:
        lines.append(
            f"| {key} | {fmt(scores['train'].get(key))} | "
            f"{fmt(scores['val'].get(key))} | {fmt(scores['test'].get(key))} |"
        )

    if confusion:
        lines += ["", "## Confusion matrix (test)", ""]
        labels = confusion["labels"]
        lines.append("| actual \\ predicted | " + " | ".join(labels) + " |")
        lines.append("| --- |" + " --- |" * len(labels))
        for label, row in zip(labels, confusion["matrix"], strict=True):
            lines.append(f"| **{label}** | " + " | ".join(str(v) for v in row) + " |")

    lines += ["", "## Feature importance", ""]
    if importance:
        lines.append("| rank | feature | gain |")
        lines.append("| --- | --- | --- |")
        for i, entry in enumerate(importance[:_TOP_FEATURES], start=1):
            lines.append(f"| {i} | `{entry['feature']}` | {fmt(entry['gain'], 2)} |")
        lines += ["", "Plot: `feature_importance.png`."]
    else:
        lines.append("_The model made no splits — importance is undefined._")

    lines += ["", "## Explainability", "", shap_note]

    lines += ["", "## Hyperparameters", "", "```json", _pretty(params), "```"]

    lines += [
        "",
        "## Caveats",
        "",
        f"- Metrics are on a held-out test split of the same dataset; they say nothing "
        f"about a different data source. See the transfer-check note in the bundle README.",
        f"- Dropped columns (`{', '.join(plan.get('dropped_columns', [])) or 'none'}`) "
        "were excluded on the task plan's reasoning; if any was excluded wrongly, "
        "re-run with a corrected plan.",
        "",
    ]
    return "\n".join(lines)


def _pretty(payload: Any) -> str:
    import json

    return json.dumps(payload, indent=2, default=str)


@tool
@guard
def evaluate_model(model_stage: ModelStage = "best") -> str:
    """Score a trained model on the test split and write the evaluation report.

    Computes task-appropriate metrics on train/val/test, a confusion matrix for
    classification, gain-based feature importance and a SHAP summary plot, then
    writes `metrics.json`, `selected_model.json`, `model_report.md`,
    `feature_importance.png` and `shap_summary.png`.

    Args:
        model_stage: Which model to report on. "best" compares the baseline and
            tuned test scores on the primary metric and keeps the simpler model
            on a tie.
    """
    meta = load_processed_metadata()
    plan = read_json("task_plan.json")
    task_type = meta["task_type"]

    stage, why = _resolve_stage(model_stage, task_type)
    model = _load_booster(stage, task_type)

    scores: dict[str, dict[str, float]] = {}
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split_name in ("train", "val", "test"):
        x_part, y_part = load_split(split_name)
        y_pred, y_proba = predict_frames(model, x_part, task_type)
        scores[split_name] = compute_metrics(task_type, y_part, y_pred, y_proba)
        predictions[split_name] = (y_part, y_pred)

    x_test, _ = load_split("test")
    importance = _importance_plot(model, list(x_test.columns))
    shap_note = _shap_plot(model, x_test)

    confusion: dict[str, Any] | None = None
    if is_classification(task_type):
        y_true, y_pred = predictions["test"]
        classes = meta.get("classes") or [str(i) for i in sorted(set(y_true.tolist()))]
        labels = list(range(len(classes)))
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        confusion = {"labels": classes, "matrix": matrix.tolist()}

    params_source = "metrics_tuned.json" if stage == "tuned" else "metrics_baseline.json"
    try:
        params = read_json(params_source).get("params", {})
    except FileNotFoundError:
        params = {}

    write_json(
        "metrics.json",
        {
            "selected_stage": stage,
            "selection_reason": why,
            "task_type": task_type,
            "primary_metric": primary_metric(task_type),
            "metrics": scores,
            "confusion_matrix": confusion,
            "feature_importance": importance,
            "params": params,
        },
    )
    write_json(
        "selected_model.json",
        {
            "stage": stage,
            "model_file": f"model_{stage}.json",
            "reason": why,
            "test_primary_metric": scores["test"].get(primary_metric(task_type)),
        },
    )
    write_text(
        "model_report.md",
        _render_report(
            stage=stage,
            why=why,
            task_type=task_type,
            meta=meta,
            plan=plan,
            scores=scores,
            importance=importance,
            confusion=confusion,
            shap_note=shap_note,
            params=params,
        ),
    )

    return (
        f"Evaluated the {stage} model ({why}).\n"
        f"  test : {format_metrics(scores['test'])}\n"
        f"  wrote model_report.md, metrics.json, selected_model.json, "
        f"feature_importance.png\n"
        f"  {shap_note}"
    )


EVALUATE_TOOLS = [evaluate_model]
