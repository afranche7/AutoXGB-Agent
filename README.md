# AutoXGB-Agent

A DeepAgents-powered ML bootstrapper. Point it at a raw dataset and a
plain-English goal; it profiles the data, picks the target, engineers features,
trains and tunes an XGBoost model, evaluates it, and exports a standalone
inference bundle you can run anywhere.

```bash
uv sync
export ANTHROPIC_API_KEY=...
uv run autoxgb run data.csv --goal "predict customer churn"
```

## What you get

Everything lands in `outputs/<run_id>/`:

| file | what it is |
| --- | --- |
| `profile_report.md` | Dtypes, missingness, cardinality, class balance, candidate targets, near-duplicate columns. |
| `task_plan.json` | The approved target, task type and dropped columns, with the reasoning. |
| `feature_spec.json` | How each raw column becomes a model feature. |
| `model_report.md` | Metrics, confusion matrix, feature importance, SHAP, caveats. |
| `metrics.json` | The same numbers, machine-readable. |
| `feature_importance.png`, `shap_summary.png` | The plots. |
| `bundle/` | A standalone inference package — see below. |

The bundle runs on its own, in its own environment, with no dependency on this
project — just pandas, numpy, pyarrow and xgboost:

```bash
cd outputs/<run_id>/bundle
uv sync
uv run predict.py /path/to/new_data.csv -o preds.csv
```

## How it works

An orchestrator owns a todo list and delegates to seven specialists, each with a
narrow slice of tools:

```
data-profiler → target-detector → feature-engineer → modeler
              → tuner → evaluator → packager
```

Artifacts on disk are the handoff between stages — the orchestrator's filesystem
is rooted at the run directory, so each specialist reads what the last one wrote
instead of carrying it in context.

### The agent writes no code

Every ML operation is a hand-written Python function with a Pydantic argument
schema. The model chooses the target, the encodings, the hyperparameter ranges —
it never emits code, and there is no `exec` and no sandbox to secure. A malformed
config comes back as a readable validation error the agent corrects on the next
turn.

Two things follow from this that are worth knowing:

- `objective`, `eval_metric` and `num_class` are **derived from the approved task
  type**, never supplied by the model, so a classification/regression mismatch is
  structurally impossible rather than merely unlikely.
- The tuning budget is clamped inside the tool. Asking for 500 trials gets you
  the cap, plus a note in the output saying so.

### One human-in-the-loop gate

The run pauses once, before the target and task type are recorded — the decision
every later stage is built on, and the one that fails silently if it is wrong.
You see the proposed plan and the agent's reasoning, and either approve it or
reject it with feedback the agent acts on.

```
Approve this plan? [y]es / [n]o (give feedback) [y]:
```

Use `--yes` for unattended runs.

### Leakage screening

Before proposing a plan, the target-detector fits a depth-3 tree on each column
*alone* against the proposed target and scores it on held-out rows. Anything
scoring near-perfect is almost certainly a value that would not exist at
prediction time, and gets flagged for dropping.

## Usage

```
autoxgb run DATASET --goal TEXT [options]
autoxgb runs                       # list previous runs and what they produced
```

| option | default | |
| --- | --- | --- |
| `--goal`, `-g` | required | What to predict, in plain English. |
| `--run-id` | timestamp | Name the run. |
| `--output-dir`, `-o` | `outputs` | Where run directories are created. |
| `--model`, `-m` | `anthropic:claude-opus-5` | Any LangChain `provider:model` string; also reads `AUTOXGB_MODEL`. |
| `--tuning-trials` | 50 | Cap on Optuna trials (50 is also the hard ceiling). |
| `--tuning-timeout` | 600 | Cap on tuning wall-clock seconds. |
| `--seed` | 42 | Random state for splits and training. |
| `--yes`, `-y` | off | Skip the approval prompt. |
| `--quiet`, `-q` | off | Narration only; hide tool arguments and results. |

Datasets may be CSV, TSV or Parquet.

## Development

```bash
uv sync
uv run pytest
```

44 tests, no network and no model calls, so `uv run pytest` is fast and free. They
cover the whole tool chain on synthetic classification and regression data
(profiling through to executing the exported `predict.py` in a subprocess), and
drive the approval gate and orchestrator→subagent delegation through real
LangGraph graphs backed by a scripted model.

Binary, multiclass and regression bundles have each been verified by hand in a
fresh `uv sync`'d environment, which is how the missing `scikit-learn` dependency
was found; the suite now guards that statically instead.

## Known gaps

- **The full agent loop has not been run against a live model.** Every tool, the
  graph wiring, delegation and the approval round-trip are covered by tests, but
  nobody has yet watched Claude drive all seven stages end to end. Expect the
  prompts to need tuning on first contact.
- No transfer check yet: a model trained here is scored only on a held-out split
  of the same dataset. See the note in each bundle's README.
- Text columns are dropped rather than featurised.
