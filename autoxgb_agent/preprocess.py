"""Self-contained preprocessing: fit to a plain-data state, transform from it.

This module has no dependency on the rest of `autoxgb_agent` — only pandas and
numpy — because it is **copied verbatim into every exported bundle**. Training
and inference therefore run the exact same code, and the fitted state is JSON
(imputation values, kept category lists, encoding maps), not a pickle. That means
a bundle is readable, diffable, and immune to version-skew unpickling failures.

If you change the transform here, bundles exported by older runs keep working:
they carry their own copy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

MISSING = "__missing__"
OTHER = "__other__"

_DATETIME_PARTS = {
    "year": lambda s: s.dt.year,
    "month": lambda s: s.dt.month,
    "day": lambda s: s.dt.day,
    "dayofweek": lambda s: s.dt.dayofweek,
    "dayofyear": lambda s: s.dt.dayofyear,
    "hour": lambda s: s.dt.hour,
    "quarter": lambda s: s.dt.quarter,
}


def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _as_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce", format="mixed")


def _as_category_strings(series: pd.Series) -> pd.Series:
    return series.astype("object").where(series.notna(), MISSING).astype(str)


def fit_preprocessor(df: pd.DataFrame, spec: dict[str, Any]) -> dict[str, Any]:
    """Learn imputation values and category vocabularies from training rows.

    Args:
        df: Training frame (features only; the target may be present and is ignored).
        spec: A `FeatureSpec` as a dict.

    Returns:
        A JSON-serialisable state dict consumed by `transform`.
    """
    state: dict[str, Any] = {
        "version": 1,
        "numeric": {},
        "categorical": {},
        "datetime": {},
        "output_columns": [],
    }
    output: list[str] = []

    for feature in spec.get("numeric", []):
        name = feature["name"]
        values = _as_numeric(df[name])
        strategy = feature.get("impute", "median")
        if strategy == "mean":
            fill = values.mean()
        elif strategy == "constant":
            fill = 0.0
        else:
            fill = values.median()
        state["numeric"][name] = {
            "fill": float(fill) if pd.notna(fill) else 0.0,
            "impute": strategy,
        }
        output.append(name)

    for feature in spec.get("categorical", []):
        name = feature["name"]
        encoding = feature.get("encoding", "one_hot")
        counts = _as_category_strings(df[name]).value_counts()
        if encoding == "one_hot":
            max_categories = int(feature.get("max_categories", 20))
            kept = [str(v) for v in counts.head(max_categories).index]
            state["categorical"][name] = {"encoding": "one_hot", "categories": kept}
            output += [f"{name}__{cat}" for cat in kept] + [f"{name}__{OTHER}"]
        else:
            mapping = {str(v): i for i, v in enumerate(counts.index)}
            state["categorical"][name] = {"encoding": "ordinal", "mapping": mapping}
            output.append(name)

    for feature in spec.get("datetime", []):
        name = feature["name"]
        parts = [p for p in feature.get("parts", []) if p in _DATETIME_PARTS]
        state["datetime"][name] = {"parts": parts}
        output += [f"{name}__{part}" for part in parts]

    state["output_columns"] = output
    return state


def transform(df: pd.DataFrame, state: dict[str, Any]) -> pd.DataFrame:
    """Apply a fitted state to a raw frame, producing the model matrix.

    Missing input columns raise; unseen categories fall into the `__other__`
    bucket (one-hot) or encode as -1 (ordinal), so inference never fails on a
    value the training data happened not to contain.
    """
    required = (
        list(state["numeric"]) + list(state["categorical"]) + list(state["datetime"])
    )
    absent = [c for c in required if c not in df.columns]
    if absent:
        msg = f"input is missing required columns: {', '.join(absent)}"
        raise KeyError(msg)

    out: dict[str, np.ndarray] = {}

    for name, info in state["numeric"].items():
        out[name] = _as_numeric(df[name]).fillna(info["fill"]).to_numpy(dtype=float)

    for name, info in state["categorical"].items():
        values = _as_category_strings(df[name])
        if info["encoding"] == "one_hot":
            kept = info["categories"]
            known = set(kept)
            for cat in kept:
                out[f"{name}__{cat}"] = (values == cat).to_numpy(dtype=float)
            out[f"{name}__{OTHER}"] = (~values.isin(known)).to_numpy(dtype=float)
        else:
            mapping = info["mapping"]
            out[name] = values.map(mapping).fillna(-1).to_numpy(dtype=float)

    for name, info in state["datetime"].items():
        parsed = _as_datetime(df[name])
        for part in info["parts"]:
            series = _DATETIME_PARTS[part](parsed)
            out[f"{name}__{part}"] = (
                pd.to_numeric(series, errors="coerce").fillna(-1).to_numpy(dtype=float)
            )

    matrix = pd.DataFrame(out, index=df.index)
    # Reindex rather than trust dict order, so column order is stable across
    # pandas versions and matches what the model was trained on.
    return matrix.reindex(columns=state["output_columns"], fill_value=0.0)
