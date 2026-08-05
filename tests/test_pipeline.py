"""End-to-end exercise of the tool chain with no LLM in the loop.

Tools are invoked exactly as the agent would (`.invoke({...})`), so these tests
also cover the LangChain arg-schema plumbing, not just the Python functions.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys

import pandas as pd
import pytest

from autoxgb_agent.tools import (
    apply_preprocessing,
    build_feature_spec,
    check_target_leakage,
    evaluate_model,
    export_bundle,
    preview_data,
    profile_dataset,
    set_task_plan,
    train_xgboost,
    tune_xgboost,
)

FEATURE_SPEC = {
    "numeric": [
        {"name": "tenure_months", "impute": "median"},
        {"name": "monthly_charges", "impute": "median"},
        {"name": "support_tickets", "impute": "median"},
    ],
    "categorical": [
        {"name": "region", "encoding": "one_hot", "max_categories": 10},
        {"name": "plan", "encoding": "ordinal"},
    ],
    "datetime": [{"name": "signup_date", "parts": ["year", "month", "dayofweek"]}],
    "test_size": 0.2,
    "val_size": 0.2,
}

DROPPED = ["customer_id", "internal_note", "always_one", "cancellation_reason_code"]


def _run_through_preprocessing(ctx, target: str, task_type: str, dropped: list[str]) -> None:
    assert "Profiled" in profile_dataset.invoke({})
    plan_result = set_task_plan.invoke(
        {
            "plan": {
                "target": target,
                "task_type": task_type,
                "dropped_columns": dropped,
                "reasoning": "Chosen for the test fixture.",
            }
        }
    )
    assert "approved" in plan_result, plan_result
    spec_result = build_feature_spec.invoke({"spec": FEATURE_SPEC})
    assert "written" in spec_result, spec_result
    prep_result = apply_preprocessing.invoke({})
    assert "Preprocessing complete" in prep_result, prep_result


# --------------------------------------------------------------------------- #
# Profiling and target selection
# --------------------------------------------------------------------------- #


def test_profile_classifies_columns_and_flags_drops(make_run, classification_csv):
    ctx = make_run(classification_csv, goal="predict churn")
    summary = profile_dataset.invoke({})

    assert "Profiled" in summary
    stats = json.loads((ctx.run_dir / "column_stats.json").read_text())
    kinds = {c["name"]: c["kind"] for c in stats["columns"]}

    assert kinds["customer_id"] == "identifier"
    assert kinds["always_one"] == "constant"
    assert kinds["internal_note"] == "text"
    assert kinds["signup_date"] == "datetime_string"
    assert kinds["region"] == "categorical"
    assert kinds["churned"] == "binary"
    assert kinds["monthly_charges"] == "numeric"

    candidates = {c["name"] for c in stats["candidate_targets"]}
    assert "churned" in candidates
    assert "customer_id" not in candidates

    report = (ctx.run_dir / "profile_report.md").read_text()
    assert "# Data profile" in report
    assert "customer_id" in report


def test_preview_shows_head_and_dtypes(make_run, classification_csv):
    make_run(classification_csv)
    out = preview_data.invoke({"n_rows": 5})
    assert "--- head(5) ---" in out
    assert "--- dtypes ---" in out
    assert "churned" in out


def test_leakage_screen_catches_the_planted_leak(make_run, classification_csv):
    ctx = make_run(classification_csv, goal="predict churn")
    out = check_target_leakage.invoke({"target": "churned"})

    payload = json.loads((ctx.run_dir / "leakage_check_churned.json").read_text())
    assert "cancellation_reason_code" in payload["leakage"], payload["scores"][:5]
    assert "tenure_months" not in payload["leakage"]
    assert "LEAKAGE" in out


def test_leakage_screen_rejects_unknown_target(make_run, classification_csv):
    make_run(classification_csv)
    assert "not in the dataset" in check_target_leakage.invoke({"target": "nope"})


# --------------------------------------------------------------------------- #
# Task plan guards
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("target", "task_type", "expected"),
    [
        ("churned", "regression", "classification target"),
        ("churned", "multiclass_classification", "binary_classification"),
        ("region", "regression", "not numeric"),
        ("customer_id", "multiclass_classification", "too many for multiclass"),
    ],
)
def test_set_task_plan_rejects_mismatched_task_types(
    make_run, classification_csv, target, task_type, expected
):
    make_run(classification_csv)
    out = set_task_plan.invoke(
        {"plan": {"target": target, "task_type": task_type, "reasoning": "test"}}
    )
    assert expected in out, out


def test_set_task_plan_rejects_unknown_columns(make_run, classification_csv):
    make_run(classification_csv)
    assert "not in the dataset" in set_task_plan.invoke(
        {"plan": {"target": "ghost", "task_type": "binary_classification", "reasoning": "t"}}
    )
    assert "not in the dataset" in set_task_plan.invoke(
        {
            "plan": {
                "target": "churned",
                "task_type": "binary_classification",
                "dropped_columns": ["ghost"],
                "reasoning": "t",
            }
        }
    )


def test_set_task_plan_surfaces_validation_errors_as_text(make_run, classification_csv):
    make_run(classification_csv)
    out = set_task_plan.invoke(
        {
            "plan": {
                "target": "churned",
                "task_type": "binary_classification",
                "dropped_columns": ["churned"],
                "reasoning": "t",
            }
        }
    )
    assert "INVALID CONFIG" in out


# --------------------------------------------------------------------------- #
# Feature spec guards
# --------------------------------------------------------------------------- #


def test_build_feature_spec_requires_a_task_plan(make_run, classification_csv):
    make_run(classification_csv)
    assert "task_plan.json does not exist" in build_feature_spec.invoke({"spec": FEATURE_SPEC})


def test_build_feature_spec_rejects_target_and_dropped_columns(make_run, classification_csv):
    make_run(classification_csv, goal="predict churn")
    set_task_plan.invoke(
        {
            "plan": {
                "target": "churned",
                "task_type": "binary_classification",
                "dropped_columns": DROPPED,
                "reasoning": "t",
            }
        }
    )
    as_target = {**FEATURE_SPEC, "numeric": [*FEATURE_SPEC["numeric"], {"name": "churned"}]}
    assert "cannot also be a feature" in build_feature_spec.invoke({"spec": as_target})

    as_dropped = {
        **FEATURE_SPEC,
        "numeric": [*FEATURE_SPEC["numeric"], {"name": "cancellation_reason_code"}],
    }
    assert "dropped in the task plan" in build_feature_spec.invoke({"spec": as_dropped})


def test_apply_preprocessing_requires_a_feature_spec(make_run, classification_csv):
    make_run(classification_csv)
    set_task_plan.invoke(
        {
            "plan": {
                "target": "churned",
                "task_type": "binary_classification",
                "dropped_columns": DROPPED,
                "reasoning": "t",
            }
        }
    )
    assert "feature_spec.json does not exist" in apply_preprocessing.invoke({})


def test_preprocessing_encodes_and_splits(make_run, classification_csv):
    ctx = make_run(classification_csv, goal="predict churn")
    _run_through_preprocessing(ctx, "churned", "binary_classification", DROPPED)

    meta = json.loads((ctx.run_dir / "processed" / "metadata.json").read_text())
    assert meta["classes"] == ["0", "1"]
    assert meta["split_sizes"]["train"] > meta["split_sizes"]["test"]

    train = pd.read_parquet(ctx.run_dir / "processed" / "train.parquet")
    assert "__target__" in train.columns
    assert train.notna().all().all(), "preprocessing must leave no NaNs"
    # region is one-hot, plan is ordinal.
    assert any(c.startswith("region__") for c in train.columns)
    assert "plan" in train.columns
    assert "signup_date__year" in train.columns


# --------------------------------------------------------------------------- #
# Training, tuning, evaluation
# --------------------------------------------------------------------------- #


def test_baseline_derives_objective_from_task_type(make_run, classification_csv):
    ctx = make_run(classification_csv, goal="predict churn")
    _run_through_preprocessing(ctx, "churned", "binary_classification", DROPPED)

    out = train_xgboost.invoke({"training_config": {"n_estimators": 120, "max_depth": 4}})
    assert "Baseline trained" in out, out

    payload = json.loads((ctx.run_dir / "metrics_baseline.json").read_text())
    assert payload["objective"] == "binary:logistic"
    assert payload["eval_metric"] == "auc"
    # The signal is real, so a working pipeline must beat chance comfortably.
    assert payload["metrics"]["test"]["roc_auc"] > 0.7


def test_regression_pipeline_uses_squared_error(make_run, regression_csv):
    ctx = make_run(regression_csv, goal="predict price")
    _run_through_preprocessing(
        ctx, "price", "regression", ["customer_id", "internal_note", "always_one"]
    )

    out = train_xgboost.invoke({"training_config": {"n_estimators": 150, "max_depth": 5}})
    assert "Baseline trained" in out, out

    payload = json.loads((ctx.run_dir / "metrics_baseline.json").read_text())
    assert payload["objective"] == "reg:squarederror"
    assert payload["metrics"]["test"]["r2"] > 0.8


def test_train_rejects_out_of_range_hyperparameters(make_run, classification_csv):
    ctx = make_run(classification_csv, goal="predict churn")
    _run_through_preprocessing(ctx, "churned", "binary_classification", DROPPED)
    out = train_xgboost.invoke({"training_config": {"max_depth": 99, "learning_rate": 5.0}})
    assert "INVALID CONFIG" in out
    assert "max_depth" in out


def test_tuning_clamps_the_budget(make_run, classification_csv):
    ctx = make_run(
        classification_csv,
        goal="predict churn",
        max_tuning_trials=3,
        max_tuning_timeout_seconds=120,
    )
    _run_through_preprocessing(ctx, "churned", "binary_classification", DROPPED)

    out = tune_xgboost.invoke(
        {
            "search_space": {
                "n_estimators": {"low": 60, "high": 140},
                "max_depth": {"low": 2, "high": 5},
            },
            "n_trials": 500,
            "timeout_seconds": 9999,
        }
    )
    assert "budget clamped" in out, out

    best = json.loads((ctx.run_dir / "best_params.json").read_text())
    assert best["budget"]["applied_trials"] == 3
    assert best["budget"]["applied_timeout_seconds"] == 120

    lines = (ctx.run_dir / "trials.jsonl").read_text().strip().splitlines()
    assert len(lines) <= 3
    assert (ctx.run_dir / "model_tuned.json").exists()


def test_tuning_rejects_an_inverted_range(make_run, classification_csv):
    ctx = make_run(classification_csv, goal="predict churn")
    _run_through_preprocessing(ctx, "churned", "binary_classification", DROPPED)
    out = tune_xgboost.invoke(
        {"search_space": {"max_depth": {"low": 9, "high": 2}}, "n_trials": 1}
    )
    assert "INVALID CONFIG" in out


def test_evaluate_selects_the_better_model_and_writes_the_report(
    make_run, classification_csv
):
    ctx = make_run(
        classification_csv,
        goal="predict churn",
        max_tuning_trials=3,
        max_tuning_timeout_seconds=120,
    )
    _run_through_preprocessing(ctx, "churned", "binary_classification", DROPPED)
    train_xgboost.invoke({"training_config": {"n_estimators": 120, "max_depth": 4}})
    tune_xgboost.invoke({"n_trials": 3, "timeout_seconds": 120})

    out = evaluate_model.invoke({"model_stage": "best"})
    assert "Evaluated the" in out, out

    metrics = json.loads((ctx.run_dir / "metrics.json").read_text())
    assert metrics["selected_stage"] in {"baseline", "tuned"}
    assert metrics["confusion_matrix"]["labels"] == ["0", "1"]
    assert metrics["feature_importance"]

    report = (ctx.run_dir / "model_report.md").read_text()
    assert "# Model report" in report
    assert "Confusion matrix" in report
    assert (ctx.run_dir / "feature_importance.png").stat().st_size > 0
    assert (ctx.run_dir / "shap_summary.png").stat().st_size > 0


def test_evaluate_without_a_model_reports_instead_of_raising(make_run, classification_csv):
    ctx = make_run(classification_csv, goal="predict churn")
    _run_through_preprocessing(ctx, "churned", "binary_classification", DROPPED)
    assert "no trained model" in evaluate_model.invoke({"model_stage": "best"})


# --------------------------------------------------------------------------- #
# Bundle
# --------------------------------------------------------------------------- #


def test_bundle_is_standalone_and_scores_new_rows(make_run, classification_csv, tmp_path):
    ctx = make_run(classification_csv, goal="predict churn")
    _run_through_preprocessing(ctx, "churned", "binary_classification", DROPPED)
    train_xgboost.invoke({"training_config": {"n_estimators": 120, "max_depth": 4}})
    evaluate_model.invoke({"model_stage": "baseline"})

    out = export_bundle.invoke({})
    assert "Bundle exported" in out, out

    bundle = ctx.run_dir / "bundle"
    for name in (
        "predict.py",
        "preprocess.py",
        "preprocessor.json",
        "model.json",
        "inference_metadata.json",
        "pyproject.toml",
        "README.md",
    ):
        assert (bundle / name).exists(), f"{name} missing from bundle"

    # The bundle must not import anything from this project (prose mentions in
    # docstrings are fine — only real imports would break standalone use).
    for name in ("preprocess.py", "predict.py"):
        source = (bundle / name).read_text()
        assert "import autoxgb_agent" not in source
        assert "from autoxgb_agent" not in source

    # Score a holdout that includes an unseen category and a missing value.
    holdout = pd.read_csv(classification_csv).head(25).drop(columns=["churned"])
    holdout.loc[0, "region"] = "atlantis"
    holdout.loc[1, "monthly_charges"] = None
    holdout_path = tmp_path / "holdout.csv"
    holdout.to_csv(holdout_path, index=False)

    output_path = tmp_path / "preds.csv"
    result = subprocess.run(
        [sys.executable, str(bundle / "predict.py"), str(holdout_path), "-o", str(output_path)],
        capture_output=True,
        text=True,
        cwd=bundle,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    predictions = pd.read_csv(output_path)
    assert len(predictions) == len(holdout)
    assert set(predictions.columns) == {"prediction", "proba_0", "proba_1"}
    assert predictions["proba_0"].between(0, 1).all()


def test_bundle_declares_every_third_party_module_it_imports(
    make_run, classification_csv
):
    """The bundle runs in its own environment, not ours.

    A subprocess test alone cannot catch this: it inherits the project venv,
    where extra packages happen to be installed. `xgboost`'s sklearn wrappers in
    particular pull in scikit-learn, which the bundle does not ship.
    """
    ctx = make_run(classification_csv, goal="predict churn")
    _run_through_preprocessing(ctx, "churned", "binary_classification", DROPPED)
    train_xgboost.invoke({"training_config": {"n_estimators": 60, "max_depth": 3}})
    export_bundle.invoke({})

    bundle = ctx.run_dir / "bundle"
    declared = set(
        re.findall(r'"([A-Za-z0-9_.\-]+)\s*[><=]', (bundle / "pyproject.toml").read_text())
    )
    local = {path.stem for path in bundle.glob("*.py")}

    imported: set[str] = set()
    for path in bundle.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])

    external = imported - set(sys.stdlib_module_names) - local
    assert external <= declared, f"undeclared bundle dependencies: {external - declared}"
    assert "sklearn" not in imported and "scikit_learn" not in imported

    # The sklearn wrappers import scikit-learn when constructed, so their absence
    # matters in code — a docstring explaining why we avoid them does not count.
    tree = ast.parse((bundle / "predict.py").read_text(encoding="utf-8"))
    referenced = {
        node.attr if isinstance(node, ast.Attribute) else node.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Attribute, ast.Name))
    }
    assert not referenced & {"XGBClassifier", "XGBRegressor"}


def test_bundle_predict_reports_missing_columns(make_run, classification_csv):
    ctx = make_run(classification_csv, goal="predict churn")
    _run_through_preprocessing(ctx, "churned", "binary_classification", DROPPED)
    train_xgboost.invoke({"training_config": {"n_estimators": 80, "max_depth": 3}})
    export_bundle.invoke({})

    bundle = ctx.run_dir / "bundle"
    sys.path.insert(0, str(bundle))
    try:
        for module in ("predict", "preprocess"):
            sys.modules.pop(module, None)
        import predict as bundle_predict  # type: ignore[import-not-found]

        frame = pd.read_csv(classification_csv).head(5).drop(columns=["tenure_months"])
        with pytest.raises(KeyError, match="tenure_months"):
            bundle_predict.predict(frame)
    finally:
        for module in ("predict", "preprocess"):
            sys.modules.pop(module, None)
        sys.path.remove(str(bundle))


def test_export_requires_a_model(make_run, classification_csv):
    ctx = make_run(classification_csv, goal="predict churn")
    _run_through_preprocessing(ctx, "churned", "binary_classification", DROPPED)
    assert "no trained model" in export_bundle.invoke({})
