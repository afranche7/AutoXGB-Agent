"""The AutoXGB web UI: start a run, watch every stage, approve the plan.

Run it with `autoxgb ui` (which is `streamlit run` on this file). Every panel is
rendered from the run's progress log, so the UI shows exactly what the CLI shows,
including for runs it did not start itself.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st

from autoxgb_agent.context import (
    MAX_TUNING_TIMEOUT_SECONDS,
    MAX_TUNING_TRIALS,
    new_run_id,
)
from autoxgb_agent.progress import (
    BLOCKED,
    DONE,
    FAILED,
    PENDING,
    RUNNING,
    STAGES,
    RunState,
    format_duration,
    read_approval_request,
    write_approval_response,
)
from autoxgb_agent.ui import launcher

REFRESH_SECONDS = 2.0
ACTIVITY_LINES = 60

STATUS_BADGE = {
    PENDING: ("○", "gray"),
    RUNNING: ("◐", "blue"),
    BLOCKED: ("⏸", "orange"),
    DONE: ("●", "green"),
    FAILED: ("✗", "red"),
}


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", default="outputs")
    known, _ = parser.parse_known_args(sys.argv[1:])
    return known


@st.cache_data(show_spinner=False)
def _cached_output_dir(raw: str) -> str:
    return str(Path(raw).expanduser().resolve())


def main() -> None:
    st.set_page_config(page_title="AutoXGB", page_icon="🌲", layout="wide")
    args = parse_args()
    output_dir = Path(_cached_output_dir(args.output_dir))

    st.title("AutoXGB")
    st.caption(
        "Point it at a dataset and a plain-English goal. Seven specialists profile "
        "the data, pick the target, engineer features, train, tune, evaluate and "
        "package it — and you watch each one work."
    )

    run_dir = sidebar(output_dir)
    if run_dir is None:
        st.info("No runs yet. Start one from the sidebar.")
        return

    state = launcher.load_state(run_dir)
    render_run(run_dir, state)

    if st.session_state.get("auto_refresh", True) and not state.is_finished:
        time.sleep(REFRESH_SECONDS)
        st.rerun()


# --------------------------------------------------------------------------- #
# Sidebar: pick a run, or start one
# --------------------------------------------------------------------------- #


def sidebar(output_dir: Path) -> Path | None:
    with st.sidebar:
        st.subheader("Runs")
        st.caption(f"`{output_dir}`")

        runs = launcher.list_runs(output_dir)
        selected: Path | None = None
        if runs:
            names = [path.name for path in runs]
            default = st.session_state.get("selected_run")
            index = names.index(default) if default in names else 0
            choice = st.selectbox("Run", names, index=index, key="run_selector")
            st.session_state["selected_run"] = choice
            selected = output_dir / choice
        else:
            st.caption("No runs yet.")

        st.checkbox("Auto-refresh", value=True, key="auto_refresh")
        if st.button("Refresh now"):
            st.rerun()

        st.divider()
        started = new_run_form(output_dir)
        if started is not None:
            selected = started

    return selected


def new_run_form(output_dir: Path) -> Path | None:
    st.subheader("New run")
    with st.form("new_run", clear_on_submit=False):
        upload = st.file_uploader("Dataset", type=["csv", "tsv", "parquet"])
        dataset_path = st.text_input(
            "…or a path on this machine", placeholder="/data/churn.csv"
        )
        goal = st.text_input("Goal", placeholder="predict customer churn")
        with st.expander("Options"):
            model = st.text_input("Model", placeholder="anthropic:claude-opus-5")
            trials = st.slider("Tuning trials", 1, MAX_TUNING_TRIALS, MAX_TUNING_TRIALS)
            timeout = st.slider(
                "Tuning seconds", 10, MAX_TUNING_TIMEOUT_SECONDS, MAX_TUNING_TIMEOUT_SECONDS
            )
            seed = st.number_input("Seed", value=42, step=1)
            auto_approve = st.checkbox(
                "Approve the task plan automatically",
                help="Otherwise the run pauses here and asks you.",
            )
        submitted = st.form_submit_button("Start run")

    if not submitted:
        return None

    dataset = _resolve_dataset(output_dir, upload, dataset_path)
    if dataset is None:
        return None
    if not goal.strip():
        st.error("Give the run a goal — what should it predict?")
        return None

    run_id = _unique_run_id(output_dir)
    spec = launcher.LaunchSpec(
        dataset=dataset,
        goal=goal.strip(),
        run_id=run_id,
        output_dir=output_dir,
        model=model.strip() or None,
        tuning_trials=int(trials),
        tuning_timeout=int(timeout),
        seed=int(seed),
        auto_approve=bool(auto_approve),
    )
    launcher.launch(spec)
    st.session_state["selected_run"] = run_id
    st.success(f"Started `{run_id}`.")
    return output_dir / run_id


def _resolve_dataset(output_dir: Path, upload: Any, typed: str) -> Path | None:
    if upload is not None:
        return launcher.save_upload(output_dir, upload.name, upload.getvalue())
    if typed.strip():
        path = Path(typed.strip()).expanduser()
        if not path.is_file():
            st.error(f"No file at `{path}`.")
            return None
        return path.resolve()
    st.error("Upload a dataset, or give a path to one.")
    return None


def _unique_run_id(output_dir: Path) -> str:
    run_id = new_run_id()
    candidate, suffix = run_id, 1
    while (output_dir / candidate).exists():
        suffix += 1
        candidate = f"{run_id}-{suffix}"
    return candidate


# --------------------------------------------------------------------------- #
# The run view
# --------------------------------------------------------------------------- #


def render_run(run_dir: Path, state: RunState) -> None:
    header(run_dir, state)
    approval_gate(run_dir, state)

    stages_tab, plan_tab, activity_tab, artifacts_tab, log_tab = st.tabs(
        ["Stages", "Plan", "Activity", "Artifacts", "Console"]
    )
    with stages_tab:
        render_stages(state)
    with plan_tab:
        render_todos(state)
    with activity_tab:
        render_activity(state)
    with artifacts_tab:
        render_artifacts(run_dir, state)
    with log_tab:
        render_console(run_dir)


def header(run_dir: Path, state: RunState) -> None:
    st.subheader(state.goal or run_dir.name)
    total = len(STAGES)
    done = state.completed_stages()

    columns = st.columns(5)
    columns[0].metric("Status", state.status)
    columns[1].metric("Stages", f"{done}/{total}")
    columns[2].metric("Elapsed", format_duration(state.elapsed()))
    columns[3].metric("Tool calls", state.n_tool_calls)
    columns[4].metric("Errors", state.n_errors)

    current = state.current_stage()
    label = current.label if current else ("finished" if state.is_finished else "waiting")
    st.progress(done / total, text=f"{label} — {done}/{total} stages complete")

    if state.error:
        st.error(state.error)
    st.caption(f"`{run_dir}` • dataset `{state.dataset or '?'}` • model `{state.model or '?'}`")


def approval_gate(run_dir: Path, state: RunState) -> None:
    """The one human decision, surfaced wherever you are watching from."""
    if state.approval is None:
        return
    request = read_approval_request(run_dir)
    if request is None:
        st.warning(
            "This run is waiting for approval, but it was started with the prompt "
            "on a terminal — answer it there."
        )
        return

    st.warning("This run is paused. Approve the task plan to let it continue.")
    for action in request.get("actions") or []:
        plan = (action.get("args") or {}).get("plan", action.get("args", {}))
        with st.container(border=True):
            st.markdown(f"**{action.get('name', 'action')}**")
            if isinstance(plan, dict) and plan.get("reasoning"):
                st.markdown(f"_{plan['reasoning']}_")
            st.json(plan)

    feedback = st.text_input(
        "If this is wrong, say what should change",
        placeholder="the target should be `renewed`, not `churned`",
        key="approval_feedback",
    )
    approve, reject = st.columns(2)
    request_id = str(request.get("id", ""))
    if approve.button("Approve", type="primary"):
        write_approval_response(run_dir, request_id, [{"type": "approve"}])
        st.rerun()
    if reject.button("Reject"):
        message = feedback.strip() or (
            "The plan was rejected. Reconsider the target and task type."
        )
        write_approval_response(
            run_dir, request_id, [{"type": "reject", "message": message}]
        )
        st.rerun()


def render_stages(state: RunState) -> None:
    for stage in state.ordered_stages:
        icon, colour = STATUS_BADGE.get(stage.status, ("?", "gray"))
        elapsed = format_duration(stage.elapsed()) if stage.started_at else "—"
        title = f"{icon} **{stage.label}** · :{colour}[{stage.status}] · {elapsed}"
        with st.expander(title, expanded=stage.status in {RUNNING, BLOCKED, FAILED}):
            if stage.instruction:
                st.caption(stage.instruction)
            if stage.detail:
                st.markdown(f"`{stage.detail}`")
            if stage.error:
                st.error(stage.error)
            if stage.tool_calls:
                st.dataframe(
                    [
                        {
                            "tool": call.tool,
                            "outcome": "running"
                            if call.ended_at is None
                            else ("ok" if call.ok else "failed"),
                            "seconds": None
                            if call.duration is None
                            else round(call.duration, 2),
                            "result": call.summary or call.progress,
                        }
                        for call in stage.tool_calls
                    ],
                    hide_index=True,
                )
            elif stage.status == PENDING:
                st.caption("Not started.")


def render_todos(state: RunState) -> None:
    if not state.todos:
        st.caption("The orchestrator has not written its plan yet.")
        return
    marks = {"completed": "✅", "in_progress": "🔵"}
    for todo in state.todos:
        st.markdown(
            f"{marks.get(str(todo.get('status')), '⚪')} {todo.get('content', '')}"
        )


def render_activity(state: RunState) -> None:
    if not state.activity:
        st.caption("Nothing yet.")
        return
    lines = []
    for entry in list(state.activity)[-ACTIVITY_LINES:]:
        stamp = time.strftime("%H:%M:%S", time.localtime(entry.get("ts", 0)))
        lines.append(f"{stamp}  {entry.get('text', '')}")
    st.code("\n".join(lines), language="text")


def render_artifacts(run_dir: Path, state: RunState) -> None:
    files = launcher.artifact_tree(run_dir)
    if not files:
        st.caption("No artifacts yet.")
        return

    names = [path.relative_to(run_dir).as_posix() for path in files]
    default = next(
        (name for name in ("model_report.md", "profile_report.md") if name in names),
        names[0],
    )
    choice = st.selectbox("File", names, index=names.index(default))
    path = run_dir / choice
    written_by = state.artifacts.get(choice, {}).get("stage") or "the run"
    st.caption(f"{path.stat().st_size:,} bytes · written by {written_by}")
    st.download_button("Download", path.read_bytes(), file_name=path.name)
    _preview(path)


def _preview(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".png":
        st.image(str(path))
    elif suffix == ".json":
        try:
            st.json(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            st.code(path.read_text(encoding="utf-8", errors="replace"))
    elif suffix == ".md":
        st.markdown(path.read_text(encoding="utf-8", errors="replace"))
    elif suffix in {".py", ".toml", ".txt", ".csv"}:
        text = path.read_text(encoding="utf-8", errors="replace")
        st.code(text[:20_000], language="python" if suffix == ".py" else "text")
    else:
        st.caption("No preview for this file type — download it instead.")


def render_console(run_dir: Path) -> None:
    text = launcher.console_log(run_dir)
    if not text:
        st.caption(
            "No console output. This run was started from a terminal, not this UI."
        )
        return
    st.code(text, language="text")


if __name__ == "__main__":  # Streamlit executes this file as `__main__`.
    main()
