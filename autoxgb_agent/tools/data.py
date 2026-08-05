"""Data preview and profiling tools."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from autoxgb_agent.tools.common import (
    ctx,
    fmt,
    guard,
    load_raw_dataset,
    write_json,
    write_text,
)

# Columns whose distinct-value ratio exceeds this look like row identifiers.
_ID_UNIQUE_RATIO = 0.95
# Below this ratio (and under the absolute cap) a string column reads as categorical.
_CATEGORICAL_UNIQUE_RATIO = 0.05
_CATEGORICAL_UNIQUE_CAP = 50
# Mean character length above which a string column reads as free text.
_TEXT_MEAN_LENGTH = 40
# Absolute Pearson correlation at which two numeric columns are near-duplicates.
_REDUNDANT_CORR = 0.98


def _looks_like_datetime(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return bool(parsed.notna().mean() > 0.9)


def _classify(series: pd.Series, n_rows: int) -> str:
    n_unique = int(series.nunique(dropna=True))
    if n_unique <= 1:
        return "constant"
    ratio = n_unique / max(n_rows, 1)

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        if n_unique == 2:
            return "binary"
        if pd.api.types.is_integer_dtype(series) and ratio > _ID_UNIQUE_RATIO:
            return "identifier"
        if pd.api.types.is_integer_dtype(series) and n_unique <= 20:
            return "discrete_numeric"
        return "numeric"

    # Object / string / categorical dtypes.
    if _looks_like_datetime(series):
        return "datetime_string"
    # Check length before uniqueness: free-text columns are usually unique too,
    # and "text" is the more actionable label for the agent.
    mean_len = float(series.dropna().astype(str).str.len().mean() or 0.0)
    if mean_len > _TEXT_MEAN_LENGTH:
        return "text"
    if ratio > _ID_UNIQUE_RATIO:
        return "identifier"
    if n_unique <= _CATEGORICAL_UNIQUE_CAP or ratio < _CATEGORICAL_UNIQUE_RATIO:
        return "categorical"
    return "high_cardinality_categorical"


def _column_stats(df: pd.DataFrame) -> list[dict[str, Any]]:
    n_rows = len(df)
    stats: list[dict[str, Any]] = []
    for name in df.columns:
        series = df[name]
        n_missing = int(series.isna().sum())
        entry: dict[str, Any] = {
            "name": name,
            "dtype": str(series.dtype),
            "kind": _classify(series, n_rows),
            "n_missing": n_missing,
            "pct_missing": round(100.0 * n_missing / max(n_rows, 1), 2),
            "n_unique": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            desc = series.describe()
            entry["numeric"] = {
                "min": float(desc.get("min", np.nan)),
                "p25": float(desc.get("25%", np.nan)),
                "median": float(desc.get("50%", np.nan)),
                "p75": float(desc.get("75%", np.nan)),
                "max": float(desc.get("max", np.nan)),
                "mean": float(desc.get("mean", np.nan)),
                "std": float(desc.get("std", np.nan)),
                "skew": float(series.skew()) if series.nunique() > 2 else 0.0,
            }
        else:
            counts = series.value_counts(dropna=True).head(5)
            entry["top_values"] = [
                {"value": str(idx), "count": int(cnt), "pct": round(100.0 * cnt / max(n_rows, 1), 2)}
                for idx, cnt in counts.items()
            ]
        stats.append(entry)
    return stats


def _redundant_pairs(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Numeric column pairs that are near-duplicates of each other."""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return []
    corr = numeric.corr(numeric_only=True).abs()
    pairs: list[dict[str, Any]] = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            value = corr.loc[a, b]
            if pd.notna(value) and value >= _REDUNDANT_CORR:
                pairs.append({"a": a, "b": b, "abs_corr": round(float(value), 4)})
    return sorted(pairs, key=lambda p: -p["abs_corr"])[:20]


def _target_candidates(stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Columns that could plausibly be a supervised target."""
    usable = {"binary", "boolean", "categorical", "discrete_numeric", "numeric"}
    out = []
    for entry in stats:
        if entry["kind"] not in usable or entry["pct_missing"] > 20:
            continue
        if entry["kind"] in {"binary", "boolean"}:
            suggested = "binary_classification"
        elif entry["kind"] in {"categorical", "discrete_numeric"} and entry["n_unique"] <= 20:
            suggested = "multiclass_classification"
        else:
            suggested = "regression"
        out.append(
            {
                "name": entry["name"],
                "kind": entry["kind"],
                "n_unique": entry["n_unique"],
                "suggested_task_type": suggested,
            }
        )
    return out


def _class_balance(df: pd.DataFrame, stats: list[dict[str, Any]]) -> dict[str, Any]:
    """Value distribution for every low-cardinality column."""
    balance: dict[str, Any] = {}
    for entry in stats:
        if entry["n_unique"] > 20 or entry["kind"] in {"identifier", "text", "constant"}:
            continue
        counts = df[entry["name"]].value_counts(dropna=True)
        if counts.empty:
            continue
        total = int(counts.sum())
        balance[entry["name"]] = {
            "counts": {str(k): int(v) for k, v in counts.head(20).items()},
            "minority_pct": round(100.0 * float(counts.min()) / max(total, 1), 2),
            "imbalance_ratio": round(float(counts.max()) / max(float(counts.min()), 1.0), 2),
        }
    return balance


def _render_report(
    df: pd.DataFrame,
    stats: list[dict[str, Any]],
    redundant: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    balance: dict[str, Any],
) -> str:
    run = ctx()
    lines: list[str] = [
        "# Data profile",
        "",
        f"- **Run**: `{run.run_id}`",
        f"- **Goal**: {run.goal}",
        f"- **Source**: `{run.dataset_path.name}`",
        f"- **Shape**: {len(df):,} rows x {df.shape[1]} columns",
        f"- **Duplicate rows**: {int(df.duplicated().sum()):,}",
        "",
        "## Columns",
        "",
        "| column | dtype | kind | missing % | distinct | notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in stats:
        if "numeric" in entry:
            n = entry["numeric"]
            note = f"min {fmt(n['min'], 3)}, median {fmt(n['median'], 3)}, max {fmt(n['max'], 3)}"
        else:
            top = entry.get("top_values", [])
            note = ", ".join(f"{t['value']} ({t['pct']}%)" for t in top[:3]) or "-"
        lines.append(
            f"| `{entry['name']}` | {entry['dtype']} | {entry['kind']} | "
            f"{entry['pct_missing']} | {entry['n_unique']} | {note} |"
        )

    lines += ["", "## Candidate targets", ""]
    if candidates:
        lines.append("| column | kind | distinct | suggested task type |")
        lines.append("| --- | --- | --- | --- |")
        for c in candidates:
            lines.append(
                f"| `{c['name']}` | {c['kind']} | {c['n_unique']} | {c['suggested_task_type']} |"
            )
    else:
        lines.append("_No obvious candidates — every column is an identifier, text or constant._")

    lines += ["", "## Class balance (low-cardinality columns)", ""]
    if balance:
        lines.append("| column | minority % | imbalance ratio |")
        lines.append("| --- | --- | --- |")
        for name, info in balance.items():
            lines.append(f"| `{name}` | {info['minority_pct']} | {info['imbalance_ratio']}:1 |")
    else:
        lines.append("_No low-cardinality columns._")

    lines += ["", "## Columns to consider dropping", ""]
    drop_notes = [
        f"- `{e['name']}` — {e['kind']}"
        for e in stats
        if e["kind"] in {"identifier", "constant", "text"} or e["pct_missing"] > 60
    ]
    lines += drop_notes or ["_None flagged._"]

    lines += ["", "## Near-duplicate numeric columns", ""]
    if redundant:
        lines.append(
            "Pairs with |r| >= 0.98. If one of these is chosen as the target, the other "
            "is almost certainly leakage."
        )
        lines.append("")
        for pair in redundant:
            lines.append(f"- `{pair['a']}` vs `{pair['b']}` — |r| = {pair['abs_corr']}")
    else:
        lines.append("_None found._")

    lines += [
        "",
        "---",
        "",
        "Full machine-readable stats: `column_stats.json`.",
        "",
    ]
    return "\n".join(lines)


@tool
@guard
def preview_data(n_rows: int = 10) -> str:
    """Show the head, dtypes and summary statistics of the run's dataset.

    Use this first, to see what the data actually looks like before profiling it.

    Args:
        n_rows: How many leading rows to show (1-50).
    """
    n_rows = max(1, min(int(n_rows), 50))
    df = load_raw_dataset()
    with pd.option_context("display.max_columns", 60, "display.width", 200):
        head = df.head(n_rows).to_string()
        dtypes = df.dtypes.to_string()
        described = df.describe(include="all").transpose().to_string()
    return (
        f"Shape: {len(df):,} rows x {df.shape[1]} columns\n\n"
        f"--- head({n_rows}) ---\n{head}\n\n"
        f"--- dtypes ---\n{dtypes}\n\n"
        f"--- describe ---\n{described}\n"
    )


@tool
@guard
def profile_dataset() -> str:
    """Profile the dataset and write `profile_report.md` and `column_stats.json`.

    Computes dtypes, missingness, cardinality, per-column distributions, class
    balance, candidate target columns and near-duplicate numeric pairs. Returns a
    short summary; read the written files for the full detail.
    """
    df = load_raw_dataset()
    stats = _column_stats(df)
    redundant = _redundant_pairs(df)
    candidates = _target_candidates(stats)
    balance = _class_balance(df, stats)

    write_json(
        "column_stats.json",
        {
            "n_rows": len(df),
            "n_columns": df.shape[1],
            "duplicate_rows": int(df.duplicated().sum()),
            "columns": stats,
            "candidate_targets": candidates,
            "redundant_numeric_pairs": redundant,
            "class_balance": balance,
        },
    )
    write_text("profile_report.md", _render_report(df, stats, redundant, candidates, balance))

    flagged = [e["name"] for e in stats if e["kind"] in {"identifier", "constant", "text"}]
    top_candidates = ", ".join(
        f"{c['name']} ({c['suggested_task_type']})" for c in candidates[:8]
    )
    return (
        f"Profiled {len(df):,} rows x {df.shape[1]} columns.\n"
        f"Wrote profile_report.md and column_stats.json.\n"
        f"Candidate targets: {top_candidates or 'none'}\n"
        f"Drop candidates (id/constant/text): {', '.join(flagged) or 'none'}\n"
        f"Near-duplicate numeric pairs: {len(redundant)}\n"
        f"Columns with >20% missing: "
        f"{', '.join(e['name'] for e in stats if e['pct_missing'] > 20) or 'none'}"
    )


DATA_TOOLS = [preview_data, profile_dataset]
