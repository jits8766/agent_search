"""Sole keyword probability gate for L0 JSON, FIND wire, and soft rank-boost.

Source of truth: ``qi.l0_llm_entity.keyword_min_probability`` (percent in YAML).
Callers convert once via :func:`min_probability_fraction`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def min_probability_fraction(l0_cfg: Any) -> float:
    """Return keyword keep threshold as a fraction in [0, 1].

    :param l0_cfg: object with ``keyword_min_probability`` in percent [0, 100]
    :raises TypeError: when ``l0_cfg`` lacks a numeric percent field
    :raises ValueError: when percent is outside [0, 100]
    """
    if l0_cfg is None:
        raise TypeError("min_probability_fraction requires l0_cfg")
    try:
        pct = float(l0_cfg.keyword_min_probability)
    except (TypeError, ValueError, AttributeError) as exc:
        raise TypeError(
            "l0_cfg.keyword_min_probability must be a number (percent)"
        ) from exc
    if not 0.0 <= pct <= 100.0:
        raise ValueError(
            f"keyword_min_probability percent must be in [0, 100], got {pct!r}"
        )
    return pct / 100.0


def filter_keywords(
    keywords: Sequence[Dict[str, Any]] | None,
    min_fraction: float,
) -> List[Dict[str, Any]]:
    """Keep keyword dicts with term + probability >= ``min_fraction``."""
    if not 0.0 <= float(min_fraction) <= 1.0:
        raise ValueError(
            f"min_fraction must be in [0, 1], got {min_fraction!r}"
        )
    if not keywords:
        return []
    kept: List[Dict[str, Any]] = []
    for item in keywords:
        if not isinstance(item, dict):
            continue
        term = str(item.get("term") or "").strip()
        if not term:
            continue
        raw_prob = item.get("probability")
        try:
            prob = float(raw_prob) if raw_prob is not None else float("nan")
        except (TypeError, ValueError):
            continue
        if prob != prob:  # NaN
            continue
        if prob < float(min_fraction):
            continue
        kept.append({"term": term, "probability": prob})
    return kept


def keyword_terms(
    keywords: Sequence[Dict[str, Any]] | None,
    min_fraction: float,
) -> List[str]:
    """Return keyword terms that pass ``min_fraction``, preserving filter order."""
    return [str(k["term"]) for k in filter_keywords(keywords, min_fraction)]


__all__ = [
    "min_probability_fraction",
    "filter_keywords",
    "keyword_terms",
]
