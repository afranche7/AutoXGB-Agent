"""Validated configs the agent fills in.

These are the only surface through which the model influences the ML pipeline.
Everything the model supplies passes through one of these before it reaches
pandas, scikit-learn or XGBoost — so a bad decision becomes a readable
validation error the agent can correct, not a stack trace that kills the run.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TaskType = Literal["binary_classification", "multiclass_classification", "regression"]

ImputeStrategy = Literal["median", "mean", "most_frequent", "constant"]
CategoricalEncoding = Literal["one_hot", "ordinal"]
DatetimePart = Literal["year", "month", "day", "dayofweek", "dayofyear", "hour", "quarter"]


class StrictModel(BaseModel):
    """Reject unknown fields so silent typos surface as validation errors."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Task plan
# --------------------------------------------------------------------------- #


class TaskPlan(StrictModel):
    """The target column and task type, plus any columns to drop.

    This is the decision that silently breaks everything downstream if it is
    wrong, which is why it is the one step gated on human approval.
    """

    target: str = Field(description="Name of the column to predict.")
    task_type: TaskType = Field(description="The supervised learning task type.")
    dropped_columns: list[str] = Field(
        default_factory=list,
        description=(
            "Columns to exclude from features: identifiers, leakage candidates, "
            "constants, or free text. Do not include the target here."
        ),
    )
    reasoning: str = Field(
        description=(
            "Two or three sentences justifying the target and task type from the "
            "user's goal and the profile: why this column, why this task type, and "
            "why each dropped column is dropped."
        )
    )

    @model_validator(mode="after")
    def _target_not_dropped(self) -> TaskPlan:
        if self.target in self.dropped_columns:
            msg = f"target {self.target!r} must not appear in dropped_columns"
            raise ValueError(msg)
        return self


# --------------------------------------------------------------------------- #
# Feature spec
# --------------------------------------------------------------------------- #


class NumericFeature(StrictModel):
    name: str
    impute: ImputeStrategy = Field(
        default="median", description="How to fill missing values."
    )

    @field_validator("impute")
    @classmethod
    def _numeric_impute(cls, v: str) -> str:
        if v == "most_frequent":
            msg = "use 'median' or 'mean' for numeric columns"
            raise ValueError(msg)
        return v


class CategoricalFeature(StrictModel):
    name: str
    encoding: CategoricalEncoding = Field(
        default="one_hot",
        description=(
            "'one_hot' for low cardinality (roughly <= 20 distinct values); "
            "'ordinal' for high cardinality, which XGBoost splits on directly."
        ),
    )
    max_categories: int = Field(
        default=20,
        ge=2,
        le=200,
        description="One-hot only: keep the N most frequent levels, bucket the rest.",
    )


class DatetimeFeature(StrictModel):
    name: str
    parts: list[DatetimePart] = Field(
        default_factory=lambda: ["year", "month", "dayofweek"],
        min_length=1,
        description="Calendar components to extract. The raw column is dropped.",
    )


class FeatureSpec(StrictModel):
    """How to turn raw columns into a model matrix, and how to split it."""

    numeric: list[NumericFeature] = Field(default_factory=list)
    categorical: list[CategoricalFeature] = Field(default_factory=list)
    datetime: list[DatetimeFeature] = Field(default_factory=list)
    test_size: float = Field(default=0.2, gt=0.0, lt=0.5)
    val_size: float = Field(
        default=0.2,
        gt=0.0,
        lt=0.5,
        description="Fraction of the post-test-split remainder held out for early stopping.",
    )

    @model_validator(mode="after")
    def _at_least_one_feature(self) -> FeatureSpec:
        if not (self.numeric or self.categorical or self.datetime):
            msg = "feature spec must contain at least one feature"
            raise ValueError(msg)
        seen: set[str] = set()
        for name in self.feature_names():
            if name in seen:
                msg = f"column {name!r} appears more than once in the feature spec"
                raise ValueError(msg)
            seen.add(name)
        return self

    def feature_names(self) -> list[str]:
        """Every raw column the spec consumes."""
        return (
            [f.name for f in self.numeric]
            + [f.name for f in self.categorical]
            + [f.name for f in self.datetime]
        )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


class TrainConfig(StrictModel):
    """XGBoost hyperparameters.

    Deliberately excludes `objective`, `eval_metric` and `num_class` — those are
    derived from the approved task type so a classification/regression mismatch
    is structurally impossible.
    """

    n_estimators: int = Field(default=400, ge=10, le=5000)
    max_depth: int = Field(default=6, ge=1, le=15)
    learning_rate: float = Field(default=0.1, gt=0.0, le=1.0)
    subsample: float = Field(default=0.9, gt=0.0, le=1.0)
    colsample_bytree: float = Field(default=0.9, gt=0.0, le=1.0)
    min_child_weight: float = Field(default=1.0, ge=0.0, le=100.0)
    gamma: float = Field(default=0.0, ge=0.0, le=20.0)
    reg_alpha: float = Field(default=0.0, ge=0.0, le=100.0)
    reg_lambda: float = Field(default=1.0, ge=0.0, le=100.0)
    early_stopping_rounds: int = Field(default=50, ge=1, le=500)
    balance_classes: bool = Field(
        default=False,
        description=(
            "Classification only: weight samples inversely to class frequency. "
            "Use when the profile reports meaningful class imbalance."
        ),
    )


class FloatRange(StrictModel):
    low: float
    high: float
    log: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> FloatRange:
        if self.low > self.high:
            msg = f"low ({self.low}) must be <= high ({self.high})"
            raise ValueError(msg)
        if self.log and self.low <= 0:
            msg = "log-scale ranges require low > 0"
            raise ValueError(msg)
        return self


class IntRange(StrictModel):
    low: int
    high: int

    @model_validator(mode="after")
    def _ordered(self) -> IntRange:
        if self.low > self.high:
            msg = f"low ({self.low}) must be <= high ({self.high})"
            raise ValueError(msg)
        return self


class SearchSpace(StrictModel):
    """Optuna search ranges. The agent picks ranges, never code."""

    n_estimators: IntRange = Field(default_factory=lambda: IntRange(low=100, high=1200))
    max_depth: IntRange = Field(default_factory=lambda: IntRange(low=3, high=10))
    learning_rate: FloatRange = Field(
        default_factory=lambda: FloatRange(low=0.01, high=0.3, log=True)
    )
    subsample: FloatRange = Field(default_factory=lambda: FloatRange(low=0.6, high=1.0))
    colsample_bytree: FloatRange = Field(
        default_factory=lambda: FloatRange(low=0.6, high=1.0)
    )
    min_child_weight: FloatRange = Field(
        default_factory=lambda: FloatRange(low=1.0, high=20.0, log=True)
    )
    gamma: FloatRange = Field(default_factory=lambda: FloatRange(low=0.0, high=5.0))
    reg_alpha: FloatRange = Field(default_factory=lambda: FloatRange(low=0.0, high=10.0))
    reg_lambda: FloatRange = Field(default_factory=lambda: FloatRange(low=0.1, high=20.0))


# --------------------------------------------------------------------------- #
# Task-type-derived XGBoost settings
# --------------------------------------------------------------------------- #


def xgb_objective(task_type: TaskType) -> str:
    return {
        "binary_classification": "binary:logistic",
        "multiclass_classification": "multi:softprob",
        "regression": "reg:squarederror",
    }[task_type]


def xgb_eval_metric(task_type: TaskType) -> str:
    return {
        "binary_classification": "auc",
        "multiclass_classification": "mlogloss",
        "regression": "rmse",
    }[task_type]


def is_classification(task_type: TaskType) -> bool:
    return task_type != "regression"


def primary_metric(task_type: TaskType) -> str:
    """The metric tuning optimises and the report leads with."""
    return {
        "binary_classification": "roc_auc",
        "multiclass_classification": "f1_macro",
        "regression": "rmse",
    }[task_type]


def greater_is_better(task_type: TaskType) -> bool:
    return is_classification(task_type)
