# AutoXGB Agent

A DeepAgents-powered ML bootstrapper. Point it at a raw dataset and a plain-English
goal, and it autonomously builds a working, tuned XGBoost pipeline — EDA report,
feature engineering code, trained model, evaluation report, and a reusable inference
script — with no ML expertise required from the user.

## Core concept

A multi-agent system built on LangChain's **DeepAgents** library that takes:

- an input dataset (CSV/parquet)
- a plain-English goal (e.g. "predict churn", "predict price")

...and produces:

- a data profiling report
- cleaned data + feature engineering code
- a trained XGBoost model (baseline, then tuned)
- an evaluation report (metrics + feature importance)
- a standalone inference script

Essentially a quick bootstrap for an ML project: pass in the data, the agent does the work.

## Why DeepAgents fits this problem

- **Built-in planning/todo tool** — the agent lays out a checklist (profile data →
  clean → engineer features → detect target/task type → train baseline → tune →
  evaluate → export) and tracks progress across a long session.
- **Virtual filesystem** — intermediate artifacts (profiling report, cleaned dataset,
  feature spec, trained model file, metrics.json) persist as "files" the agent
  reads/writes between steps, since this is a multi-stage pipeline, not one-shot.
- **Sub-agents** — the workflow decomposes into specialists instead of one agent
  holding the entire pipeline in context at once.
- **Human-in-the-loop / interrupt support** — lets the user approve things like
  "I inferred this is a classification problem" or "I'm dropping this leaky column"
  before the agent proceeds.

## Suggested architecture

**Orchestrator agent** — owns the todo list, decides task type
(classification/regression/multiclass), delegates to sub-agents, assembles the
final report.

**Sub-agents:**

1. **Data Profiler** — runs pandas/ydata-profiling-style checks: dtypes,
   missingness, cardinality, target leakage candidates, class imbalance,
   correlations. Writes `profile_report.md`.
2. **Target & Task Detector** — given the user's goal string + column stats,
   decides the target column and task type (binary/multiclass/regression), flags
   ambiguity for user confirmation.
3. **Feature Engineer** — writes `preprocessing.py`: handles missing values,
   encodes categoricals (target/one-hot depending on cardinality), datetime
   decomposition, generates `feature_spec.json`.
4. **Modeler** — writes `train.py` using XGBoost, sets up train/val/test split
   (or CV), picks the right objective/eval_metric based on task type, runs
   baseline fit.
5. **Tuner** — runs hyperparameter search (Optuna or `RandomizedSearchCV`) within
   a time/trial budget, logs trials.
6. **Evaluator** — computes metrics (AUC/F1/accuracy or RMSE/MAE/R²), generates
   SHAP feature-importance plots, writes `model_report.md`.
7. **Packager** — exports `model.json`, a `predict.py` inference script, and a
   `pyproject.toml` for the bundle. This project uses **uv + pyproject.toml**
   throughout; do not generate `requirements.txt` anywhere.

## Tools the agents actually have

There is **no code execution tool**. Every ML operation is a hand-written
function with a Pydantic argument schema; the agent fills in decisions, never
code. Consequently there is no sandbox to secure and no `exec`.

- `preview_data`, `profile_dataset` — data profiler
- `check_target_leakage`, `set_task_plan` — target & task detector
- `build_feature_spec`, `apply_preprocessing` — feature engineer
- `train_xgboost` — modeler
- `tune_xgboost` — tuner
- `evaluate_model` — evaluator
- `export_bundle` — packager

Plus DeepAgents' built-ins: `write_todos`, `task`, and the filesystem tools
rooted at the run directory. `execute` (shell) and `delete` are deliberately
withheld.

## Repo structure

```
autoxgb_agent/
  cli.py            # `autoxgb run data.csv --goal "predict churn"`, `autoxgb runs`
  orchestrator.py   # create_deep_agent wiring, permissions, approval gate
  subagents.py      # the 7 SubAgent specs
  context.py        # RunContext (dataset path, run dir, budget) via ContextVar
  schemas.py        # TaskPlan, FeatureSpec, TrainConfig, SearchSpace
  modeling.py       # shared model build / fit / metric helpers
  preprocess.py     # fit + transform; copied verbatim into every bundle
  prompts/*.md      # one per agent
  tools/*.py        # the tool suite above
  templates/        # predict.py, pyproject.toml, README for the bundle
outputs/<run_id>/   # artifacts, gitignored
tests/
```

## Conventions

- **Tools never take file paths.** The dataset location and run directory come
  from `RunContext`; the agent supplies decisions only.
- **Tools report, they don't raise.** `@guard` turns exceptions into readable
  tool text so the agent can correct itself; `harden()` does the same for
  LangChain's pre-call argument validation.
- **Artifacts are the handoff.** Every tool writes its full output to the run
  directory and returns a short summary, so large tables never enter context.
- **Never name a tool parameter `config`** — LangChain's `StructuredTool._run`
  takes `config` as a keyword-only argument and would swallow it.
- **`objective`/`eval_metric`/`num_class` are derived** from the approved task
  type in `modeling.py`, never supplied by the model.

## Still to build

- Streamlit front-end for non-technical users to upload data and watch the run.
- The transfer check described below.
- Featurising text columns instead of dropping them.

## Design decisions worth thinking about early

- **Guardrails on code execution** — since the agent writes/runs Python, sandbox
  it (Docker or restricted subprocess) rather than raw `exec`.
- **Constrained vs. free-form codegen** — fully free-form code generation for
  training gives flexibility but is fragile; a parameterized template (agent
  fills in a config dict, a fixed script runs it) is far more robust for a
  target user with little ML knowledge, who needs reliability more than
  flexibility.
- **Task-type ambiguity** — surface this back to the user rather than silently
  guessing; a classification/regression mismatch quietly breaks everything
  downstream.
- **Budget control** — cap tuning trials/time so the agent doesn't spiral into
  an expensive Optuna run on a large dataset.

## Possible extension: cross-dataset generalization check

XGBoost models don't generalize across datasets with the same schema but
different underlying distributions (e.g. the same pipeline applied to a
different factory's data) — they need retraining or fine-tuning per source.
A natural extension of this agent: given a new dataset with the same schema as
a previously-trained model, add a **transfer-check step** where the agent first
evaluates the existing model on the new data, and only triggers a full retrain
if performance drops below a threshold.
