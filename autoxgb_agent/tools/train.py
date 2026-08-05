"""Baseline XGBoost training."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from autoxgb_agent.modeling import (
    build_model,
    compute_metrics,
    fit_model,
    format_metrics,
    load_processed_metadata,
    load_split,
    predict_frames,
    xgb_params,
)
from autoxgb_agent.schemas import (
    TrainConfig,
    is_classification,
    xgb_eval_metric,
    xgb_objective,
)
from autoxgb_agent.tools.common import ctx, guard, validate, write_json


@tool
@guard
def train_xgboost(training_config: TrainConfig) -> str:
    """Train a baseline XGBoost model on the processed training split.

    You supply hyperparameters only. The objective, eval metric and class count
    are derived from the approved task type, so a classification/regression
    mismatch is impossible. Early stopping uses the validation split. Writes
    `model_baseline.json` and `metrics_baseline.json`.

    Args:
        training_config: XGBoost hyperparameters and early-stopping/class-balance
            settings.
    """
    config = validate(TrainConfig, training_config)
    run = ctx()
    meta = load_processed_metadata()
    task_type = meta["task_type"]
    classification = is_classification(task_type)

    x_train, y_train = load_split("train")
    x_val, y_val = load_split("val")
    x_test, y_test = load_split("test")

    model = build_model(
        task_type,
        xgb_params(config),
        n_classes=len(meta.get("classes", [])) or None,
        early_stopping_rounds=config.early_stopping_rounds,
        random_state=run.random_state,
    )
    fit_model(
        model,
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        balance_classes=config.balance_classes,
        classification=classification,
    )

    scores: dict[str, dict[str, float]] = {}
    for split_name, (x_part, y_part) in {
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "test": (x_test, y_test),
    }.items():
        y_pred, y_proba = predict_frames(model, x_part, task_type)
        scores[split_name] = compute_metrics(task_type, y_part, y_pred, y_proba)

    model_path = run.path("model_baseline.json")
    model.save_model(str(model_path))

    best_iteration = getattr(model, "best_iteration", None)
    payload: dict[str, Any] = {
        "stage": "baseline",
        "task_type": task_type,
        "objective": xgb_objective(task_type),
        "eval_metric": xgb_eval_metric(task_type),
        "params": xgb_params(config),
        "balance_classes": config.balance_classes,
        "early_stopping_rounds": config.early_stopping_rounds,
        "best_iteration": int(best_iteration) if best_iteration is not None else None,
        "n_features": meta["n_features"],
        "split_sizes": meta["split_sizes"],
        "metrics": scores,
    }
    write_json("metrics_baseline.json", payload)

    overfit = ""
    if classification and "roc_auc" in scores["train"] and "roc_auc" in scores["val"]:
        gap = scores["train"]["roc_auc"] - scores["val"]["roc_auc"]
        if gap > 0.15:
            overfit = (
                f"\n  WARNING: train AUC exceeds validation AUC by {gap:.3f} — "
                "consider lower max_depth, more regularisation, or lower learning_rate."
            )

    return (
        f"Baseline trained ({xgb_objective(task_type)}), saved to model_baseline.json.\n"
        f"  best_iteration: {best_iteration}\n"
        f"  train : {format_metrics(scores['train'])}\n"
        f"  val   : {format_metrics(scores['val'])}\n"
        f"  test  : {format_metrics(scores['test'])}\n"
        f"  full metrics written to metrics_baseline.json{overfit}"
    )


TRAIN_TOOLS = [train_xgboost]
