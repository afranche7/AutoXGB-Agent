"""Synthetic datasets, run-context fixtures and a scripted model.

No test in this suite calls a real model, so the whole suite is free to run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from autoxgb_agent.context import RunContext, reset_run_context, set_run_context


class ScriptedChatModel(BaseChatModel):
    """Replays a fixed list of assistant turns.

    LangChain's bundled fakes don't implement `bind_tools`, which `create_agent`
    requires, so this is the smallest thing that can drive a real graph. One
    instance backs every agent in a DeepAgents graph, so its responses are
    consumed in call order across the orchestrator and its subagents.
    """

    responses: list[AIMessage]
    index: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:  # noqa: ARG002
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # noqa: ANN001, ARG002
        position = min(self.index, len(self.responses) - 1)
        self.index += 1
        # Copy: the graph stamps ids onto messages, and a replayed turn must not
        # carry the previous run's identity.
        return ChatResult(
            generations=[ChatGeneration(message=self.responses[position].model_copy())]
        )

N_ROWS = 900
SEED = 7


def _base_frame(rng: np.random.Generator) -> pd.DataFrame:
    """Mixed-dtype frame with the messes a real dataset has."""
    n = N_ROWS
    frame = pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(n)],
            "tenure_months": rng.integers(0, 72, n).astype(float),
            "monthly_charges": rng.normal(70, 25, n).round(2),
            "support_tickets": rng.poisson(1.4, n),
            "region": rng.choice(["north", "south", "east", "west"], n),
            "plan": rng.choice([f"plan_{i}" for i in range(40)], n),  # high cardinality
            "signup_date": pd.to_datetime("2021-01-01")
            + pd.to_timedelta(rng.integers(0, 1200, n), unit="D"),
            "internal_note": [f"free text note number {i} about the account" for i in range(n)],
            "always_one": 1,
        }
    )
    # Real data has holes.
    frame.loc[rng.choice(n, 60, replace=False), "monthly_charges"] = np.nan
    frame.loc[rng.choice(n, 40, replace=False), "region"] = np.nan
    return frame


@pytest.fixture(scope="session")
def classification_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Binary churn dataset with one deliberately leaky column."""
    rng = np.random.default_rng(SEED)
    frame = _base_frame(rng)

    logit = (
        -2.0
        + 0.045 * (72 - frame["tenure_months"])
        + 0.02 * frame["monthly_charges"].fillna(70)
        + 0.55 * frame["support_tickets"]
        + np.where(frame["region"] == "south", 0.8, 0.0)
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    frame["churned"] = (rng.random(len(frame)) < probability).astype(int)
    # Recorded only after churn is known — a textbook leak.
    frame["cancellation_reason_code"] = np.where(
        frame["churned"] == 1, rng.integers(1, 6, len(frame)), 0
    )

    path = tmp_path_factory.mktemp("data") / "churn.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def regression_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Continuous price target over the same feature shapes."""
    rng = np.random.default_rng(SEED + 1)
    frame = _base_frame(rng)
    frame["price"] = (
        120.0
        + 3.1 * frame["tenure_months"]
        + 1.4 * frame["monthly_charges"].fillna(70)
        - 8.0 * frame["support_tickets"]
        + np.where(frame["region"] == "north", 45.0, 0.0)
        + rng.normal(0, 18, len(frame))
    ).round(2)
    path = tmp_path_factory.mktemp("data") / "price.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def make_run(tmp_path: Path):
    """Activate a RunContext for a dataset, and tear it down afterwards."""
    tokens = []

    def _make(dataset: Path, goal: str = "predict the target", **overrides):
        ctx = RunContext(
            dataset_path=dataset,
            run_dir=tmp_path / "run",
            goal=goal,
            run_id="run-test",
            **overrides,
        )
        tokens.append(set_run_context(ctx))
        return ctx

    yield _make

    for token in reversed(tokens):
        reset_run_context(token)
