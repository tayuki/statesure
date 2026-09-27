"""Turn judge responses into signal readings.

These functions are pure: they take an already-received response and return
readings. Network calls, retries and images belong to the judge adapters.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import Any

from .readings import Outcome, SignalReading
from .recipe import ABSTAIN_WORDS, Recipe, SignalSpec

_FENCE_START = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_END = re.compile(r"\s*```$")
_DIGITS = re.compile(r"^\d{1,3}$")
_DISTRIBUTION_TOLERANCE = 1e-6


def parse_json_object(content: object) -> dict[str, Any] | None:
    """Extract the JSON object from a model reply, or return None."""
    if not isinstance(content, str):
        return None
    text = _FENCE_END.sub("", _FENCE_START.sub("", content.strip()))
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def normalize_vlm_json(
    recipe: Recipe, content: object
) -> tuple[dict[str, str], dict[str, SignalReading]]:
    """Normalize a VLM reply that should contain one JSON object.

    Returns ``(quality, signals)``. Keys outside the recipe are dropped, so free
    text the model adds is never kept.
    """
    obj = parse_json_object(content)
    if obj is None:
        return {}, {spec.name: SignalReading.invalid("no_json") for spec in recipe.signals}

    quality = {
        field: obj[field]
        for field, allowed in recipe.quality.items()
        if isinstance(obj.get(field), str) and obj[field] in allowed
    }
    signals = {spec.name: _read_vlm_value(spec, obj) for spec in recipe.signals}
    if any(quality.get(rule.field) == rule.equals for rule in recipe.abstain_when):
        signals = {
            name: reading
            if reading.outcome is Outcome.ERROR
            else SignalReading.abstained("low_quality")
            for name, reading in signals.items()
        }
    return quality, apply_implications(recipe, signals)


def normalize_jev_answers(
    recipe: Recipe, answers: object, *, min_probability: float | None = None
) -> dict[str, SignalReading]:
    """Normalize the ``answers`` map of a Jev-style decision response.

    ``presence`` signals expect ``noul`` answers; ``enum`` and ``count`` signals
    expect ``choice`` answers whose options are the recipe values.
    """
    if not isinstance(answers, Mapping):
        return {spec.name: SignalReading.invalid("no_answers") for spec in recipe.signals}
    signals: dict[str, SignalReading] = {}
    for spec in recipe.signals:
        answer = answers.get(spec.name)
        if answer is None:
            signals[spec.name] = SignalReading.invalid("missing")
            continue
        reading = _read_jev_answer(spec, answer)
        if (
            reading.is_decided
            and min_probability is not None
            and reading.probability is not None
            and reading.probability < min_probability
        ):
            reading = SignalReading.abstained("below_threshold", distribution=reading.distribution)
        signals[spec.name] = reading
    return apply_implications(recipe, signals)


def error_readings(recipe: Recipe, reason: str = "judge_error") -> dict[str, SignalReading]:
    """Readings for a judge call that failed. The sample stays in the population."""
    return {spec.name: SignalReading.error(reason) for spec in recipe.signals}


def apply_implications(
    recipe: Recipe, signals: Mapping[str, SignalReading]
) -> dict[str, SignalReading]:
    """Apply the recipe's consistency rules once, in file order."""
    result = dict(signals)
    for rule in recipe.implications:
        current = result.get(rule.when.signal)
        if current is None:
            continue
        if rule.when.abstained:
            matched = current.outcome is Outcome.ABSTAINED
        else:
            matched = current.is_decided and current.value == rule.when.equals
        if not matched:
            continue
        for target, value in rule.set.items():
            if _replaceable(result.get(target)):
                result[target] = SignalReading.decided(value, reason="implied")
        for target, fix in rule.fix.items():
            reading = result.get(target)
            if reading is not None and reading.is_decided and reading.value == fix.if_equals:
                result[target] = (
                    SignalReading.abstained("implied")
                    if fix.abstain
                    else SignalReading.decided(fix.set, reason="implied")
                )
        for target in rule.abstain:
            if _replaceable(result.get(target)):
                result[target] = SignalReading.abstained("implied")
    return result


def _replaceable(reading: SignalReading | None) -> bool:
    # A failed judge call stays visible as an error; rules never hide it.
    return reading is None or reading.outcome is not Outcome.ERROR


def _read_vlm_value(spec: SignalSpec, obj: Mapping[str, Any]) -> SignalReading:
    if spec.name not in obj:
        return SignalReading.invalid("missing")
    raw = obj[spec.name]
    if isinstance(raw, str) and raw in ABSTAIN_WORDS:
        return SignalReading.abstained("not_visible" if raw == "not_visible" else "model_uncertain")
    if spec.type == "count":
        value: object = raw
        if isinstance(raw, str) and _DIGITS.fullmatch(raw):
            value = int(raw)
        # bool is a subclass of int; a boolean is never a count.
        if type(value) is int and spec.accepts(value):
            return SignalReading.decided(value)
        return SignalReading.invalid("out_of_schema")
    if spec.accepts(raw):
        return SignalReading.decided(raw)
    return SignalReading.invalid("out_of_schema")


def _read_jev_answer(spec: SignalSpec, answer: object) -> SignalReading:
    if not isinstance(answer, Mapping):
        return SignalReading.invalid("invalid_answer")
    if spec.type == "presence":
        if answer.get("type") != "noul" or not _probability(answer.get("noul")):
            return SignalReading.invalid("invalid_distribution")
        p_present = float(answer["noul"])
        distribution = {"present": p_present, "absent": 1.0 - p_present}
        if p_present == 0.5:
            return SignalReading.abstained("tie", distribution=distribution)
        value = "present" if p_present > 0.5 else "absent"
        return SignalReading.decided(
            value, probability=distribution[value], distribution=distribution
        )

    if answer.get("type") != "choice":
        return SignalReading.invalid("invalid_answer_type")
    probabilities = answer.get("probabilities")
    expected = {str(value) for value in spec.values}
    if not isinstance(probabilities, Mapping) or set(probabilities) != expected:
        return SignalReading.invalid("invalid_distribution")
    if not all(_probability(p) for p in probabilities.values()):
        return SignalReading.invalid("invalid_distribution")
    distribution = {key: float(p) for key, p in probabilities.items()}
    if abs(sum(distribution.values()) - 1.0) > _DISTRIBUTION_TOLERANCE:
        return SignalReading.invalid("invalid_distribution")
    choice = answer.get("choice")
    if not isinstance(choice, str) or choice not in distribution:
        return SignalReading.invalid("invalid_choice")
    if distribution[choice] != max(distribution.values()):
        return SignalReading.invalid("inconsistent_choice")
    value = int(choice) if spec.type == "count" else choice
    return SignalReading.decided(value, probability=distribution[choice], distribution=distribution)


def _probability(value: object) -> bool:
    return (
        type(value) in (int, float) and math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0
    )
