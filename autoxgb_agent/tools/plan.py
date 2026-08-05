"""Target selection, leakage checking and the approved task plan."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from autoxgb_agent.schemas import TaskPlan
from autoxgb_agent.tools.common import (
    ToolFailure,
    ctx,
    guard,
    load_raw_dataset,
    validate,
    write_json,
)

# A single feature scoring above this against the target is almost certainly
# leakage — no legitimate predictor is this good on its own.
_LEAKAGE_THRESHOLD = 0.99
# Below this it is merely strong; worth a look but not damning.
_SUSPICIOUS_THRESHOLD = 0.90
# Leakage screening is a sanity check, not a model — cap the work it does.
_MAX_SCREEN_ROWS = 20_000


def _encode_for_screening(series: pd.Series) -> np.ndarray | None:
    """Turn one column into a numeric array a shallow tree can split on."""
    if pd.api.types.is_datetime64_any_dtype(series):
        values = series.astype("int64").astype(float)
        values[series.isna()] = np.nan
    elif pd.api.types.is_bool_dtype(series):
        values = series.astype(float)
    elif pd.api.types.is_numeric_dtype(series):
        values = series.astype(float)
    else:
        codes = series.astype("category").cat.codes.astype(float)
        codes[codes < 0] = np.nan
        values = codes
    array = values.to_numpy(dtype=float).reshape(-1, 1)
    if np.all(np.isnan(array)):
        return None
    # Trees need finite values; median-fill is fine for a screening pass.
    median = np.nanmedian(array)
    return np.nan_to_num(array, nan=0.0 if np.isnan(median) else median)


def _screen_feature(
    x: np.ndarray, y: np.ndarray, *, classification: bool, binary: bool, seed: int
) -> float | None:
    """Score how well one feature alone predicts the target, on held-out rows."""
    try:
        stratify = y if classification and len(np.unique(y)) > 1 else None
        x_tr, x_te, y_tr, y_te = train_test_split(
            x, y, test_size=0.3, random_state=seed, stratify=stratify
        )
    except ValueError:
        return None
    if len(np.unique(y_tr)) < 2 and classification:
        return None

    if classification:
        model = DecisionTreeClassifier(max_depth=3, random_state=seed).fit(x_tr, y_tr)
        if binary and len(np.unique(y_te)) == 2:
            proba = model.predict_proba(x_te)[:, 1]
            return float(roc_auc_score(y_te, proba))
        # Multiclass: accuracy relative to the majority-class baseline, rescaled
        # onto 0-1 so it is comparable with AUC.
        accuracy = float(model.score(x_te, y_te))
        baseline = float(pd.Series(y_te).value_counts(normalize=True).max())
        if baseline >= 1.0:
            return None
        return max(0.0, (accuracy - baseline) / (1.0 - baseline)) * 0.5 + 0.5

    model = DecisionTreeRegressor(max_depth=3, random_state=seed).fit(x_tr, y_tr)
    return float(r2_score(y_te, model.predict(x_te)))


@tool
@guard
def check_target_leakage(target: str) -> str:
    """Screen every column for leakage against a proposed target column.

    Fits a depth-3 decision tree on each feature alone and scores it on held-out
    rows (ROC AUC for binary targets, a baseline-adjusted accuracy for multiclass,
    R^2 for regression). A single feature scoring near 1.0 is almost always
    leakage — a value that would not be available at prediction time. Run this
    before `set_task_plan` for any target you are considering.

    Args:
        target: The column you intend to predict.
    """
    df = load_raw_dataset()
    if target not in df.columns:
        msg = f"column {target!r} is not in the dataset. Available: {', '.join(df.columns)}"
        raise ToolFailure(msg)

    seed = ctx().random_state
    frame = df.dropna(subset=[target])
    if len(frame) > _MAX_SCREEN_ROWS:
        frame = frame.sample(_MAX_SCREEN_ROWS, random_state=seed)
    if frame.empty:
        msg = f"every row has a missing {target!r}; it cannot be the target"
        raise ToolFailure(msg)

    y_series = frame[target]
    classification = not (
        pd.api.types.is_float_dtype(y_series) and y_series.nunique() > 20
    ) and y_series.nunique() <= 50
    if classification:
        y = y_series.astype("category").cat.codes.to_numpy()
        binary = int(pd.Series(y).nunique()) == 2
    else:
        y = pd.to_numeric(y_series, errors="coerce").to_numpy(dtype=float)
        binary = False
        keep = ~np.isnan(y)
        frame, y = frame[keep], y[keep]

    scored: list[dict[str, Any]] = []
    for name in frame.columns:
        if name == target:
            continue
        x = _encode_for_screening(frame[name])
        if x is None:
            continue
        score = _screen_feature(x, y, classification=classification, binary=binary, seed=seed)
        if score is None:
            continue
        scored.append({"column": name, "score": round(float(score), 4)})

    scored.sort(key=lambda item: -item["score"])
    leaks = [s for s in scored if s["score"] >= _LEAKAGE_THRESHOLD]
    suspicious = [
        s for s in scored if _SUSPICIOUS_THRESHOLD <= s["score"] < _LEAKAGE_THRESHOLD
    ]

    write_json(
        f"leakage_check_{target}.json",
        {
            "target": target,
            "metric": "roc_auc"
            if binary
            else ("adjusted_accuracy" if classification else "r2"),
            "rows_screened": int(len(frame)),
            "scores": scored,
            "leakage": [s["column"] for s in leaks],
            "suspicious": [s["column"] for s in suspicious],
        },
    )

    metric = "AUC" if binary else ("adj. accuracy" if classification else "R^2")
    lines = [
        f"Leakage screen for target {target!r} ({metric}, {len(frame):,} rows):",
        "",
        "Top single-feature predictors:",
    ]
    lines += [f"  {s['column']}: {s['score']}" for s in scored[:10]] or ["  (none scoreable)"]
    lines += [""]
    if leaks:
        lines.append(
            f"LEAKAGE (>= {_LEAKAGE_THRESHOLD}): {', '.join(s['column'] for s in leaks)} "
            "— drop these unless you can justify them being known at prediction time."
        )
    else:
        lines.append(f"No feature scores >= {_LEAKAGE_THRESHOLD}. No obvious leakage.")
    if suspicious:
        lines.append(
            f"Suspicious ({_SUSPICIOUS_THRESHOLD}-{_LEAKAGE_THRESHOLD}): "
            f"{', '.join(s['column'] for s in suspicious)} — check whether these are "
            "genuinely available before the target is known."
        )
    return "\n".join(lines)


def _check_task_type_fits(series: pd.Series, task_type: str, target: str) -> None:
    n_unique = int(series.nunique(dropna=True))
    if task_type == "binary_classification" and n_unique != 2:
        msg = (
            f"{target!r} has {n_unique} distinct values, so it is not binary. "
            f"Use {'multiclass_classification' if n_unique <= 50 else 'regression'} instead."
        )
        raise ToolFailure(msg)
    if task_type == "multiclass_classification":
        if n_unique < 3:
            msg = f"{target!r} has only {n_unique} distinct values; use binary_classification."
            raise ToolFailure(msg)
        if n_unique > 50:
            msg = (
                f"{target!r} has {n_unique} distinct values — too many for multiclass. "
                "Use regression, or pick a different target."
            )
            raise ToolFailure(msg)
    if task_type == "regression":
        if not pd.api.types.is_numeric_dtype(series):
            msg = (
                f"{target!r} is {series.dtype}, not numeric, so regression is impossible. "
                "Use a classification task type."
            )
            raise ToolFailure(msg)
        if n_unique <= 2:
            msg = (
                f"{target!r} takes only {n_unique} distinct values — that is a "
                "classification target, not a regression one."
            )
            raise ToolFailure(msg)


@tool
@guard
def set_task_plan(plan: TaskPlan) -> str:
    """Record the approved target column, task type and dropped columns.

    This is the decision every later stage depends on, so it is gated on human
    approval. Run `profile_dataset` and `check_target_leakage` first, and put
    identifiers, leakage candidates, constants and free-text columns in
    `dropped_columns`. Writes `task_plan.json`.

    Args:
        plan: The target column, task type, columns to drop, and your reasoning.
    """
    plan = validate(TaskPlan, plan)
    df = load_raw_dataset()

    if plan.target not in df.columns:
        msg = (
            f"target {plan.target!r} is not in the dataset. "
            f"Available columns: {', '.join(df.columns)}"
        )
        raise ToolFailure(msg)

    unknown = [c for c in plan.dropped_columns if c not in df.columns]
    if unknown:
        msg = f"dropped_columns contains columns not in the dataset: {', '.join(unknown)}"
        raise ToolFailure(msg)

    remaining = [c for c in df.columns if c != plan.target and c not in plan.dropped_columns]
    if not remaining:
        msg = "dropping those columns leaves no features. Keep at least one predictor."
        raise ToolFailure(msg)

    target_series = df[plan.target].dropna()
    _check_task_type_fits(target_series, plan.task_type, plan.target)

    payload: dict[str, Any] = {
        **plan.model_dump(),
        "n_rows": int(len(df)),
        "n_rows_with_target": int(len(target_series)),
        "feature_columns": remaining,
    }
    if plan.task_type != "regression":
        counts = target_series.value_counts()
        payload["classes"] = [str(c) for c in counts.index.tolist()]
        payload["class_counts"] = {str(k): int(v) for k, v in counts.items()}
        payload["n_classes"] = int(len(counts))
        payload["imbalance_ratio"] = round(
            float(counts.max()) / max(float(counts.min()), 1.0), 2
        )
    write_json("task_plan.json", payload)

    summary = [
        "Task plan approved and written to task_plan.json.",
        f"  target      : {plan.target}",
        f"  task_type   : {plan.task_type}",
        f"  features    : {len(remaining)} columns",
        f"  dropped     : {', '.join(plan.dropped_columns) or 'none'}",
        f"  usable rows : {len(target_series):,} of {len(df):,}",
    ]
    if plan.task_type != "regression":
        summary.append(
            f"  classes     : {payload['n_classes']} "
            f"(imbalance {payload['imbalance_ratio']}:1)"
        )
    return "\n".join(summary)


PLAN_TOOLS = [check_target_leakage, set_task_plan]
