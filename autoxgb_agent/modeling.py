"""Shared model construction, fitting and metric helpers.

Used by the training, tuning and evaluation tools so all three agree on how a
model is built and scored. The `objective`/`eval_metric`/`num_class` triple is
derived here from the approved task type and is never supplied by the agent.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics as skm
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from autoxgb_agent.schemas import (
    TaskType,
    TrainConfig,
    is_classification,
    xgb_eval_metric,
    xgb_objective,
)
from autoxgb_agent.tools.common import ToolFailure, ctx, read_json

TARGET_COLUMN = "__target__"

# Hyperparameters that go straight to XGBoost; everything else in TrainConfig is
# ours (early stopping, class balancing) and is handled explicitly.
_XGB_PARAM_FIELDS = (
    "n_estimators",
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "gamma",
    "reg_alpha",
    "reg_lambda",
)


def load_processed_metadata() -> dict[str, Any]:
    try:
        return read_json("processed/metadata.json")
    except FileNotFoundError:
        msg = (
            "processed/metadata.json does not exist. Run apply_preprocessing before "
            "training, tuning or evaluating."
        )
        raise ToolFailure(msg) from None


def load_split(name: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load one processed split as (features, target)."""
    path = ctx().run_dir / "processed" / f"{name}.parquet"
    if not path.exists():
        msg = f"processed/{name}.parquet is missing. Run apply_preprocessing first."
        raise ToolFailure(msg)
    frame = pd.read_parquet(path)
    y = frame[TARGET_COLUMN].to_numpy()
    x = frame.drop(columns=[TARGET_COLUMN])
    return x, y


def build_model(
    task_type: TaskType,
    params: dict[str, Any],
    *,
    n_classes: int | None = None,
    early_stopping_rounds: int | None = None,
    random_state: int = 42,
) -> XGBClassifier | XGBRegressor:
    """Construct an XGBoost estimator for the approved task type."""
    common: dict[str, Any] = {
        **params,
        "objective": xgb_objective(task_type),
        "eval_metric": xgb_eval_metric(task_type),
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    if early_stopping_rounds is not None:
        common["early_stopping_rounds"] = early_stopping_rounds

    if is_classification(task_type):
        if task_type == "multiclass_classification":
            common["num_class"] = n_classes
        return XGBClassifier(**common)
    return XGBRegressor(**common)


def xgb_params(config: TrainConfig) -> dict[str, Any]:
    return {field: getattr(config, field) for field in _XGB_PARAM_FIELDS}


def fit_model(
    model: XGBClassifier | XGBRegressor,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    eval_set: list[tuple[pd.DataFrame, np.ndarray]] | None = None,
    balance_classes: bool = False,
    classification: bool = False,
) -> XGBClassifier | XGBRegressor:
    """Fit with optional early stopping and inverse-frequency class weights."""
    kwargs: dict[str, Any] = {"verbose": False}
    if eval_set:
        kwargs["eval_set"] = eval_set
    if balance_classes and classification:
        kwargs["sample_weight"] = compute_sample_weight("balanced", y_train)
    model.fit(x_train, y_train, **kwargs)
    return model


def predict_frames(
    model: XGBClassifier | XGBRegressor, x: pd.DataFrame, task_type: TaskType
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (hard predictions, probabilities-or-None)."""
    if is_classification(task_type):
        proba = model.predict_proba(x)
        return proba.argmax(axis=1), proba
    return model.predict(x), None


def compute_metrics(
    task_type: TaskType,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
) -> dict[str, float]:
    """Task-appropriate metrics. Anything undefined for the data is omitted."""
    out: dict[str, float] = {}
    if task_type == "regression":
        out["rmse"] = float(np.sqrt(skm.mean_squared_error(y_true, y_pred)))
        out["mae"] = float(skm.mean_absolute_error(y_true, y_pred))
        out["r2"] = float(skm.r2_score(y_true, y_pred))
        nonzero = np.abs(y_true) > 1e-9
        if nonzero.any():
            out["mape"] = float(
                np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100.0
            )
        return out

    out["accuracy"] = float(skm.accuracy_score(y_true, y_pred))
    average = "binary" if task_type == "binary_classification" else "macro"
    out["f1"] = float(skm.f1_score(y_true, y_pred, average=average, zero_division=0))
    out["f1_macro"] = float(skm.f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["precision"] = float(
        skm.precision_score(y_true, y_pred, average=average, zero_division=0)
    )
    out["recall"] = float(skm.recall_score(y_true, y_pred, average=average, zero_division=0))
    out["balanced_accuracy"] = float(skm.balanced_accuracy_score(y_true, y_pred))

    if y_proba is None:
        return out
    present = np.unique(y_true)
    try:
        if task_type == "binary_classification" and len(present) == 2:
            scores = y_proba[:, 1]
            out["roc_auc"] = float(skm.roc_auc_score(y_true, scores))
            out["pr_auc"] = float(skm.average_precision_score(y_true, scores))
            out["log_loss"] = float(skm.log_loss(y_true, scores, labels=[0, 1]))
        elif len(present) > 2:
            labels = list(range(y_proba.shape[1]))
            out["roc_auc_ovr"] = float(
                skm.roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            )
            out["log_loss"] = float(skm.log_loss(y_true, y_proba, labels=labels))
    except ValueError:
        # A split that happens to contain a single class makes these undefined.
        pass
    return out


def score_for_tuning(task_type: TaskType, metrics: dict[str, float]) -> float:
    """The single number Optuna optimises (always maximised)."""
    if task_type == "binary_classification":
        return metrics.get("roc_auc", metrics.get("f1", 0.0))
    if task_type == "multiclass_classification":
        return metrics.get("f1_macro", metrics.get("accuracy", 0.0))
    return -metrics.get("rmse", float("inf"))


def format_metrics(metrics: dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
