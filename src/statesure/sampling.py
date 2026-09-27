"""Decide which samples are stored for human review, and how they are weighted.

Storing a sample (it enters the population) and the order in which a person
labels it are separate. Metrics are weighted by the probability of being
stored, never by labeling priority.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass

from .readings import Outcome, SignalReading
from .reconcile import Comparison

STRATA = ("error", "conflict", "change", "low_confidence", "ordinary")
PRIORITIES = ("critical", "high", "routine", "archive")
DEFAULT_ROUTINE_RATE = 1 / 6


@dataclass(frozen=True)
class ReviewClass:
    stratum: str
    priority: str
    stored: bool
    inclusion_probability: float
    display_key: float
    """Random key for display order within a priority."""


def classify_for_review(
    primary: Mapping[str, SignalReading],
    *,
    primary_failed: bool,
    comparison: Comparison | None,
    reasons: tuple[str, ...],
    rng: random.Random,
    save_all: bool,
    routine_rate: float = DEFAULT_ROUTINE_RATE,
) -> ReviewClass:
    """Assign a stratum, a labeling priority and the storage probability."""
    if not 0.0 < routine_rate <= 1.0:
        raise ValueError("routine_rate must be in (0, 1]")
    display_key = rng.random()
    all_error = bool(primary) and all(r.outcome is Outcome.ERROR for r in primary.values())
    if primary_failed or all_error:
        return ReviewClass("error", "critical", True, 1.0, display_key)
    if comparison is not None and comparison.agreement == "conflict":
        return ReviewClass("conflict", "critical", True, 1.0, display_key)
    if "change_candidate" in reasons:
        return ReviewClass("change", "high", True, 1.0, display_key)
    if "low_quality" in reasons:
        return ReviewClass("low_confidence", "high", True, 1.0, display_key)

    selected = rng.random() < routine_rate
    if save_all:
        return ReviewClass("ordinary", "routine" if selected else "archive", True, 1.0, display_key)
    return ReviewClass("ordinary", "routine", selected, routine_rate, display_key)
