"""Budget-capped hyperparameter search with Optuna."""

from __future__ import annotations

import json
from typing import Any

import optuna
import pandas as pd
from langchain_core.tools import tool
from optuna.samplers import TPESampler

from autoxgb_agent.modeling import (
    build_model,
    compute_metrics,
    fit_model,
    format_metrics,
    load_processed_metadata,
    load_split,
    predict_frames,
    score_for_tuning,
)
from autoxgb_agent.schemas import (
    FloatRange,
    IntRange,
    SearchSpace,
    is_classification,
    primary_metric,
)
from autoxgb_agent.tools.common import ctx, guard, validate, write_json

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest(trial: optuna.Trial, space: SearchSpace) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in space.model_dump().items():
        field = getattr(space, name)
        if isinstance(field, IntRange):
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif isinstance(field, FloatRange):
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec["log"]
            )
    return params


@tool
@guard
def tune_xgboost(
    search_space: SearchSpace | None = None,
    n_trials: int = 25,
    timeout_seconds: int = 300,
    balance_classes: bool = False,
    early_stopping_rounds: int = 50,
) -> str:
    """Search hyperparameters with Optuna, then refit the best on train+validation.

    Optimises ROC AUC (binary), macro F1 (multiclass) or negative RMSE
    (regression) on the validation split. Both `n_trials` and `timeout_seconds`
    are clamped to the run's budget, so an over-ambitious request is silently
    reduced rather than allowed to run away. Writes `best_params.json`,
    `trials.jsonl`, `model_tuned.json` and `metrics_tuned.json`.

    Args:
        search_space: Per-hyperparameter ranges. Omit for sensible defaults.
        n_trials: Number of trials to attempt.
        timeout_seconds: Wall-clock cap for the whole search.
        balance_classes: Classification only — weight samples by inverse class frequency.
        early_stopping_rounds: Validation rounds without improvement before a trial stops.
    """
    space = validate(SearchSpace, search_space or {})
    run = ctx()
    meta = load_processed_metadata()
    task_type = meta["task_type"]
    classification = is_classification(task_type)
    n_classes = len(meta.get("classes", [])) or None

    requested_trials, requested_timeout = int(n_trials), int(timeout_seconds)
    n_trials = max(1, min(requested_trials, run.max_tuning_trials))
    timeout_seconds = max(10, min(requested_timeout, run.max_tuning_timeout_seconds))
    early_stopping_rounds = max(1, min(int(early_stopping_rounds), 500))
    clamped = (n_trials != requested_trials) or (timeout_seconds != requested_timeout)

    x_train, y_train = load_split("train")
    x_val, y_val = load_split("val")
    x_test, y_test = load_split("test")

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial, space)
        model = build_model(
            task_type,
            params,
            n_classes=n_classes,
            early_stopping_rounds=early_stopping_rounds,
            random_state=run.random_state,
        )
        fit_model(
            model,
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            balance_classes=balance_classes,
            classification=classification,
        )
        best_iteration = getattr(model, "best_iteration", None)
        trial.set_user_attr(
            "best_iteration", int(best_iteration) if best_iteration is not None else None
        )
        y_pred, y_proba = predict_frames(model, x_val, task_type)
        return score_for_tuning(task_type, compute_metrics(task_type, y_val, y_pred, y_proba))

    study = optuna.create_study(
        direction="maximize", sampler=TPESampler(seed=run.random_state)
    )
    study.optimize(
        objective, n_trials=n_trials, timeout=timeout_seconds, catch=(Exception,)
    )

    completed = [t for t in study.trials if t.value is not None]
    if not completed:
        msg = (
            "every tuning trial failed. Check that preprocessing produced usable "
            "splits, then retry with a narrower search space."
        )
        raise RuntimeError(msg)

    trials_path = run.path("trials.jsonl")
    with trials_path.open("w", encoding="utf-8") as handle:
        for trial in study.trials:
            handle.write(
                json.dumps(
                    {
                        "number": trial.number,
                        "state": trial.state.name,
                        "value": trial.value,
                        "params": trial.params,
                        "best_iteration": trial.user_attrs.get("best_iteration"),
                    }
                )
                + "\n"
            )

    best = study.best_trial
    best_params = dict(best.params)
    best_iteration = best.user_attrs.get("best_iteration")
    # Refit on train+val: early stopping needs a holdout we no longer have, so
    # pin the tree count to what the winning trial actually used.
    if best_iteration is not None:
        best_params["n_estimators"] = int(best_iteration) + 1

    x_full = pd.concat([x_train, x_val], axis=0, ignore_index=True)
    y_full = pd.concat(
        [pd.Series(y_train), pd.Series(y_val)], axis=0, ignore_index=True
    ).to_numpy()

    final = build_model(
        task_type,
        best_params,
        n_classes=n_classes,
        random_state=run.random_state,
    )
    fit_model(
        final,
        x_full,
        y_full,
        balance_classes=balance_classes,
        classification=classification,
    )
    final.save_model(str(run.path("model_tuned.json")))

    scores: dict[str, dict[str, float]] = {}
    for split_name, (x_part, y_part) in {
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "test": (x_test, y_test),
    }.items():
        y_pred, y_proba = predict_frames(final, x_part, task_type)
        scores[split_name] = compute_metrics(task_type, y_part, y_pred, y_proba)

    write_json(
        "best_params.json",
        {
            "params": best_params,
            "balance_classes": balance_classes,
            "best_trial_number": best.number,
            "best_validation_score": best.value,
            "optimised_metric": primary_metric(task_type),
            "trials_run": len(study.trials),
            "trials_completed": len(completed),
            "budget": {
                "requested_trials": requested_trials,
                "applied_trials": n_trials,
                "requested_timeout_seconds": requested_timeout,
                "applied_timeout_seconds": timeout_seconds,
            },
        },
    )
    write_json(
        "metrics_tuned.json",
        {
            "stage": "tuned",
            "task_type": task_type,
            "params": best_params,
            "balance_classes": balance_classes,
            "refit_on": "train+val",
            "metrics": scores,
        },
    )

    note = ""
    if clamped:
        note = (
            f"\n  NOTE: budget clamped — asked for {requested_trials} trials / "
            f"{requested_timeout}s, ran {n_trials} / {timeout_seconds}s."
        )
    return (
        f"Tuning complete: {len(completed)}/{len(study.trials)} trials succeeded, "
        f"optimising {primary_metric(task_type)}.\n"
        f"  best trial   : #{best.number} (validation score {best.value:.4f})\n"
        f"  best params  : {json.dumps(best_params)}\n"
        f"  refit on train+val, saved to model_tuned.json\n"
        f"  test         : {format_metrics(scores['test'])}{note}"
    )


TUNE_TOOLS = [tune_xgboost]
