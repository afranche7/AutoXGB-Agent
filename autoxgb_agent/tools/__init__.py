"""The fixed tool suite the agents fill in configs for.

There is deliberately no code-execution tool: every ML operation is a
hand-written function with a validated Pydantic argument schema, so the model
chooses *decisions*, never code.
"""

from autoxgb_agent.tools.common import harden
from autoxgb_agent.tools.data import DATA_TOOLS, preview_data, profile_dataset
from autoxgb_agent.tools.evaluate import EVALUATE_TOOLS, evaluate_model
from autoxgb_agent.tools.features import (
    FEATURE_TOOLS,
    apply_preprocessing,
    build_feature_spec,
)
from autoxgb_agent.tools.package import PACKAGE_TOOLS, export_bundle
from autoxgb_agent.tools.plan import PLAN_TOOLS, check_target_leakage, set_task_plan
from autoxgb_agent.tools.train import TRAIN_TOOLS, train_xgboost
from autoxgb_agent.tools.tune import TUNE_TOOLS, tune_xgboost

ALL_TOOLS = harden(
    [
        *DATA_TOOLS,
        *PLAN_TOOLS,
        *FEATURE_TOOLS,
        *TRAIN_TOOLS,
        *TUNE_TOOLS,
        *EVALUATE_TOOLS,
        *PACKAGE_TOOLS,
    ]
)

__all__ = [
    "ALL_TOOLS",
    "DATA_TOOLS",
    "EVALUATE_TOOLS",
    "FEATURE_TOOLS",
    "PACKAGE_TOOLS",
    "PLAN_TOOLS",
    "TRAIN_TOOLS",
    "TUNE_TOOLS",
    "apply_preprocessing",
    "build_feature_spec",
    "check_target_leakage",
    "evaluate_model",
    "export_bundle",
    "preview_data",
    "profile_dataset",
    "set_task_plan",
    "train_xgboost",
    "tune_xgboost",
]
