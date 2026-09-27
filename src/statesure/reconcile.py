"""Compare a primary reading with a focused re-check of the same image."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .normalize import apply_implications
from .readings import Outcome, SignalReading
from .recipe import Recipe


@dataclass(frozen=True)
class Comparison:
    """How the primary and verification readings related, per signal."""

    agreement: str
    agreed: tuple[str, ...]
    disagreed: tuple[str, ...]
    recovered: tuple[str, ...]
    unconfirmed: tuple[str, ...]


def reconcile(
    recipe: Recipe,
    primary: Mapping[str, SignalReading],
    verified: Mapping[str, SignalReading],
) -> tuple[dict[str, SignalReading], Comparison]:
    """Merge readings: agreement is kept, conflict abstains, a re-check may recover."""
    merged: dict[str, SignalReading] = {}
    agreed: list[str] = []
    disagreed: list[str] = []
    recovered: list[str] = []
    unconfirmed: list[str] = []
    for name in recipe.signal_names:
        first = primary.get(name, SignalReading.invalid("missing"))
        second = verified.get(name, SignalReading.invalid("missing"))
        if first.is_decided and second.is_decided:
            if first.value == second.value:
                agreed.append(name)
                merged[name] = first
            else:
                disagreed.append(name)
                merged[name] = SignalReading.abstained("conflict")
        elif first.is_decided:
            unconfirmed.append(name)
            merged[name] = SignalReading.abstained("unconfirmed")
        elif second.is_decided:
            recovered.append(name)
            merged[name] = second
        else:
            merged[name] = first

    if disagreed or unconfirmed:
        agreement = "conflict"
    elif recovered and agreed:
        agreement = "partial"
    elif recovered:
        agreement = "verification_only"
    elif agreed:
        agreement = "full"
    else:
        agreement = "no_comparable_signals"
    comparison = Comparison(
        agreement=agreement,
        agreed=tuple(agreed),
        disagreed=tuple(disagreed),
        recovered=tuple(recovered),
        unconfirmed=tuple(unconfirmed),
    )
    return apply_implications(recipe, merged), comparison


def needs_verification(
    recipe: Recipe,
    quality: Mapping[str, str],
    primary: Mapping[str, SignalReading],
    confirmed_values: Mapping[str, str | int | None],
    *,
    always_verify: bool = False,
) -> tuple[str, ...]:
    """Return why the sample should be re-checked; empty means no re-check."""
    reasons: list[str] = []
    if always_verify:
        reasons.append("always_verify")
    low_quality = any(quality.get(rule.field) == rule.equals for rule in recipe.abstain_when)
    below_threshold = any(
        reading.outcome is Outcome.ABSTAINED and reading.reason == "below_threshold"
        for reading in primary.values()
    )
    if low_quality or below_threshold:
        reasons.append("low_quality")
    change = False
    baseline = False
    for name in recipe.signal_names:
        reading = primary.get(name)
        if reading is None or not reading.is_decided:
            continue
        confirmed = confirmed_values.get(name)
        if confirmed is None:
            baseline = True
        elif reading.value != confirmed:
            change = True
    if change:
        reasons.append("change_candidate")
    if baseline:
        reasons.append("baseline_unconfirmed")
    return tuple(reasons)
