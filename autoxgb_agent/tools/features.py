"""Feature specification and preprocessing."""

from __future__ import annotations

from typing import Any

import pandas as pd
from langchain_core.tools import tool
from sklearn.model_selection import train_test_split

from autoxgb_agent.preprocess import fit_preprocessor, transform
from autoxgb_agent.schemas import FeatureSpec, is_classification
from autoxgb_agent.tools.common import (
    ToolFailure,
    ctx,
    guard,
    load_raw_dataset,
    read_json,
    validate,
    write_json,
)

TARGET_COLUMN = "__target__"


def _require_task_plan() -> dict[str, Any]:
    try:
        return read_json("task_plan.json")
    except FileNotFoundError:
        msg = (
            "task_plan.json does not exist yet. The target and task type must be "
            "approved via set_task_plan before features can be built."
        )
        raise ToolFailure(msg) from None


@tool
@guard
def build_feature_spec(spec: FeatureSpec) -> str:
    """Validate and record how raw columns become model features.

    Assign each usable column to exactly one of `numeric`, `categorical` or
    `datetime`, using the kinds reported in `profile_report.md`. Prefer `one_hot`
    for categoricals under ~20 levels and `ordinal` above that. Do not include
    the target or any column dropped in the task plan. Writes `feature_spec.json`.

    Args:
        spec: The feature assignment plus the train/validation/test split sizes.
    """
    spec = validate(FeatureSpec, spec)
    plan = _require_task_plan()
    df = load_raw_dataset()

    names = spec.feature_names()
    unknown = [n for n in names if n not in df.columns]
    if unknown:
        msg = f"these columns are not in the dataset: {', '.join(unknown)}"
        raise ToolFailure(msg)

    if plan["target"] in names:
        msg = f"the target {plan['target']!r} cannot also be a feature"
        raise ToolFailure(msg)

    dropped = set(plan.get("dropped_columns", []))
    reused = [n for n in names if n in dropped]
    if reused:
        msg = (
            f"these columns were dropped in the task plan and must not be features: "
            f"{', '.join(reused)}"
        )
        raise ToolFailure(msg)

    unassigned = [
        c for c in df.columns if c != plan["target"] and c not in dropped and c not in names
    ]

    write_json("feature_spec.json", spec.model_dump())

    lines = [
        "Feature spec written to feature_spec.json.",
        f"  numeric     : {len(spec.numeric)} ({', '.join(f.name for f in spec.numeric) or '-'})",
        f"  categorical : {len(spec.categorical)} "
        f"({', '.join(f'{f.name}:{f.encoding}' for f in spec.categorical) or '-'})",
        f"  datetime    : {len(spec.datetime)} ({', '.join(f.name for f in spec.datetime) or '-'})",
        f"  split       : test={spec.test_size}, val={spec.val_size} (of the remainder)",
    ]
    if unassigned:
        lines.append(
            f"  NOTE: {len(unassigned)} column(s) are neither features nor dropped and "
            f"will be ignored: {', '.join(unassigned)}. If that is wrong, call this "
            "tool again with them assigned."
        )
    return "\n".join(lines)


@tool
@guard
def apply_preprocessing() -> str:
    """Fit the preprocessor on the training split and materialise train/val/test sets.

    Reads `task_plan.json` and `feature_spec.json`, drops rows with a missing
    target, splits the data (stratified for classification), fits imputation and
    encoding on the training rows only, and writes
    `processed/{train,val,test}.parquet` plus `preprocessor.json`. Run this once,
    after `build_feature_spec`.
    """
    run = ctx()
    plan = _require_task_plan()
    try:
        spec_dict = read_json("feature_spec.json")
    except FileNotFoundError:
        msg = "feature_spec.json does not exist yet. Call build_feature_spec first."
        raise ToolFailure(msg) from None
    spec = validate(FeatureSpec, spec_dict)

    df = load_raw_dataset()
    target = plan["target"]
    task_type = plan["task_type"]

    frame = df.dropna(subset=[target]).reset_index(drop=True)
    if len(frame) < 50:
        msg = f"only {len(frame)} rows have a non-missing target; too few to train on"
        raise ToolFailure(msg)

    features = frame[spec.feature_names()]
    classes: list[str] | None = None
    if is_classification(task_type):
        classes = sorted(str(v) for v in frame[target].dropna().unique())
        codes = {label: i for i, label in enumerate(classes)}
        y = frame[target].astype(str).map(codes)
    else:
        y = pd.to_numeric(frame[target], errors="coerce")
        if y.isna().any():
            msg = f"target {target!r} has non-numeric values; it cannot be a regression target"
            raise ToolFailure(msg)

    stratify = y if is_classification(task_type) else None
    x_rest, x_test, y_rest, y_test = train_test_split(
        features,
        y,
        test_size=spec.test_size,
        random_state=run.random_state,
        stratify=stratify,
    )
    stratify_rest = y_rest if is_classification(task_type) else None
    x_train, x_val, y_train, y_val = train_test_split(
        x_rest,
        y_rest,
        test_size=spec.val_size,
        random_state=run.random_state,
        stratify=stratify_rest,
    )

    state = fit_preprocessor(x_train, spec.model_dump())
    write_json("preprocessor.json", state)

    splits = {
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "test": (x_test, y_test),
    }
    sizes: dict[str, int] = {}
    for split_name, (x_part, y_part) in splits.items():
        matrix = transform(x_part, state)
        matrix[TARGET_COLUMN] = y_part.to_numpy()
        path = run.path("processed", f"{split_name}.parquet")
        matrix.to_parquet(path, index=False)
        sizes[split_name] = len(matrix)

    metadata: dict[str, Any] = {
        "target": target,
        "task_type": task_type,
        "n_features": len(state["output_columns"]),
        "feature_columns": state["output_columns"],
        "split_sizes": sizes,
        "random_state": run.random_state,
    }
    if classes is not None:
        metadata["classes"] = classes
    write_json("processed/metadata.json", metadata)

    return (
        f"Preprocessing complete. {len(state['output_columns'])} model features from "
        f"{len(spec.feature_names())} raw columns.\n"
        f"  train / val / test rows: {sizes['train']:,} / {sizes['val']:,} / {sizes['test']:,}\n"
        f"  wrote processed/{{train,val,test}}.parquet, processed/metadata.json, "
        f"preprocessor.json\n"
        + (f"  classes: {', '.join(classes)}\n" if classes else "")
    )


FEATURE_TOOLS = [build_feature_spec, apply_preprocessing]
