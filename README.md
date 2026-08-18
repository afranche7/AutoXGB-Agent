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

## Watching it work

A run is seven specialists and a few minutes of tuning, so it tells you where it
is the whole way through. The terminal keeps a live dashboard pinned below the
narration:

```
╭──────────────────────────────── pipeline ─────────────────────────────────╮
│    stage                  status      time   tools  detail                │
│ ●  Profile data           done       19.0s       1  churned looks like …  │
│ ⏸  Target & task type     blocked    1m38s       1  waiting for your app… │
│ ○  Engineer features      pending        -       -                        │
│ ○  Train baseline         pending        -       -                        │
│ ╭─────────────────────────────── plan ────────────────────────────────╮   │
│ │ ✓ profile the data                                                  │   │
│ │ ▸ pick the target                                                   │   │
│ ╰─────────────────────────────────────────────────────────────────────╯   │
╰──────── 1/7 stages • 2m00s • 2 tool calls • 1 error(s) • running ─────────╯
```

Every run also writes an append-only event log to
`outputs/<run_id>/progress/events.jsonl` — each delegation, tool call, artifact
and error, with timings. Everything else is a view over that file:

```bash
autoxgb watch                 # follow the most recent run, from any terminal
autoxgb watch <run_id>        # follow (or replay) a specific one
autoxgb ui                    # the web UI: start runs, watch them, approve plans
```

Because the log is a file, `watch` works on a run someone else started, and on a
run that finished last week. `--plain` swaps the dashboard for one progress line
per change, which is what you want in CI.

### The web UI

`autoxgb ui` (needs `uv sync --extra ui`) serves a Streamlit front-end for people
who would rather not use a terminal: upload a dataset, type a goal, and watch each
stage light up — with the tool calls, their durations, the todo list, the activity
log, and every artifact browsable and downloadable as it appears. The task-plan
approval appears there too, so a non-technical user can run the whole thing.

The UI never runs the agent in-process: it spawns `autoxgb run` and reads the same
progress log. Closing the browser does not stop a run.

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
| `progress/events.jsonl` | Every stage, tool call, artifact and error, with timings. |

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

Use `--yes` for unattended runs. Where the question gets asked is up to you:

| `--approve-via` | who answers |
| --- | --- |
| `prompt` (default) | this terminal |
| `file` | the UI, or `autoxgb watch <run_id> --approve` from anywhere |
| `auto` | nobody — same as `--yes` |

In `file` mode the run publishes the pending plan into its own directory and
blocks until something answers, which is how the UI can approve a run it started
in another process.

### Leakage screening

Before proposing a plan, the target-detector fits a depth-3 tree on each column
*alone* against the proposed target and scores it on held-out rows. Anything
scoring near-perfect is almost certainly a value that would not exist at
prediction time, and gets flagged for dropping.

## Usage

```
autoxgb run DATASET --goal TEXT [options]
autoxgb runs                       # previous runs: how far each got, what it produced
autoxgb watch [RUN_ID]             # follow or replay a run's progress
autoxgb ui                         # the web UI
```

| `run` option | default | |
| --- | --- | --- |
| `--goal`, `-g` | required | What to predict, in plain English. |
| `--run-id` | timestamp | Name the run. |
| `--output-dir`, `-o` | `outputs` | Where run directories are created. |
| `--model`, `-m` | `anthropic:claude-opus-5` | Any LangChain `provider:model` string; also reads `AUTOXGB_MODEL`. |
| `--tuning-trials` | 50 | Cap on Optuna trials (50 is also the hard ceiling). |
| `--tuning-timeout` | 600 | Cap on tuning wall-clock seconds. |
| `--seed` | 42 | Random state for splits and training. |
| `--yes`, `-y` | off | Skip the approval prompt. |
| `--approve-via` | `prompt` | Who answers the approval gate: `prompt`, `file` or `auto`. |
| `--quiet`, `-q` | off | Narration only; hide tool arguments and results. |
| `--plain` | off | No live dashboard; one progress line per change. |

| `watch` option | default | |
| --- | --- | --- |
| `--output-dir`, `-o` | `outputs` | Where run directories live. |
| `--approve` | off | Answer this run's approval gate from here. |
| `--follow/--no-follow` | follow | Keep watching, or render once and exit. |
| `--refresh` | 0.5 | Seconds between polls of the progress log. |

Datasets may be CSV, TSV or Parquet.

## Development

```bash
uv sync
uv run pytest
```

103 tests, no network and no model calls, so `uv run pytest` is fast and free. They
cover the whole tool chain on synthetic classification and regression data
(profiling through to executing the exported `predict.py` in a subprocess), and
drive the approval gate and orchestrator→subagent delegation through real
LangGraph graphs backed by a scripted model.

Progress tracking is covered end to end: the event log and its reducer, the tools
reporting themselves, `autoxgb run`/`watch`/`runs` driven through the real
commands, and the Streamlit page rendered headlessly with Streamlit's own test
harness — including approving and rejecting a plan from it.

Binary, multiclass and regression bundles have each been verified by hand in a
fresh `uv sync`'d environment, which is how the missing `scikit-learn` dependency
was found; the suite now guards that statically instead.

## UI
