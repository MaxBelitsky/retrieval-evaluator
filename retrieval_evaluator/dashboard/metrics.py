from __future__ import annotations

import re

import pandas as pd

R_METRIC_RE = re.compile(r"^R@(\d+)$")
MRR_METRIC_RE = re.compile(r"^MRR@(\d+)$")


def _available_metric_ks(source, pattern: re.Pattern[str]) -> list[int]:
    if isinstance(source, pd.DataFrame):
        items = ((col, source[col].notna().any()) for col in source.columns)
    else:
        items = ((col, pd.notna(value)) for col, value in source.items())

    ks: list[int] = []
    for col, has_value in items:
        match = pattern.fullmatch(str(col))
        if match and has_value:
            ks.append(int(match.group(1)))
    return sorted(set(ks))


def available_r_ks(source) -> list[int]:
    return _available_metric_ks(source, R_METRIC_RE)


def available_r_metrics(source) -> list[str]:
    return [f"R@{k}" for k in available_r_ks(source)]


def available_mrr_metric(source) -> str | None:
    ks = _available_metric_ks(source, MRR_METRIC_RE)
    return f"MRR@{ks[-1]}" if ks else None


def primary_r_metric(source) -> str | None:
    metrics = available_r_metrics(source)
    return metrics[-1] if metrics else None


def format_metric(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.1f}%"


def format_runtime(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.1f}s"


def truncate_text(text: str, limit: int = 280) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"
