from __future__ import annotations

import json

import pytest

from statesure.normalize import (
    error_readings,
    normalize_jev_answers,
    normalize_vlm_json,
    parse_json_object,
)
from statesure.readings import Outcome, SignalReading
from statesure.recipe import Recipe


def _vlm(**values: object) -> str:
    base = {"visibility": "good", "analysis_confidence": "high"}
    return json.dumps({**base, **values})


def test_parse_json_object_handles_fences_and_noise() -> None:
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! {"a": 1} hope this helps') == {"a": 1}
    assert parse_json_object("no json here") is None
    assert parse_json_object("[1, 2]") is None
    assert parse_json_object(None) is None


def test_decided_values(package_recipe: Recipe) -> None:
    quality, signals = normalize_vlm_json(
        package_recipe,
        _vlm(package_present="present", package_count=2, package_location="doorstep"),
    )
    assert quality == {"visibility": "good", "analysis_confidence": "high"}
    assert signals["package_present"] == SignalReading.decided("present")
    assert signals["package_count"] == SignalReading.decided(2)
    assert signals["package_location"] == SignalReading.decided("doorstep")


def test_abstain_words_are_not_values(package_recipe: Recipe) -> None:
    _, signals = normalize_vlm_json(package_recipe, _vlm(package_present="not_visible"))
    assert signals["package_present"].outcome is Outcome.ABSTAINED
    assert signals["package_present"].reason == "not_visible"
    _, signals = normalize_vlm_json(package_recipe, _vlm(package_present="uncertain"))
    assert signals["package_present"].reason == "model_uncertain"


@pytest.mark.parametrize("raw", [True, 11, -1, 2.5, "two", None])
def test_invalid_counts(package_recipe: Recipe, raw: object) -> None:
    # Out-of-range counts are not clamped: clamping would hide a misread.
    _, signals = normalize_vlm_json(
        package_recipe, _vlm(package_present="present", package_count=raw)
    )
    assert signals["package_count"].outcome is Outcome.INVALID


def test_count_accepts_digit_strings(package_recipe: Recipe) -> None:
    _, signals = normalize_vlm_json(
        package_recipe, _vlm(package_present="present", package_count="3")
    )
    assert signals["package_count"] == SignalReading.decided(3)


def test_missing_and_out_of_schema(package_recipe: Recipe) -> None:
    _, signals = normalize_vlm_json(package_recipe, _vlm(package_present="maybe"))
    assert signals["package_present"] == SignalReading.invalid("out_of_schema")
    _, signals = normalize_vlm_json(package_recipe, "not json")
    assert all(r == SignalReading.invalid("no_json") for r in signals.values())


def test_low_quality_abstains_everything(package_recipe: Recipe) -> None:
    content = json.dumps(
        {
            "visibility": "poor",
            "analysis_confidence": "high",
            "package_present": "absent",
            "package_count": 0,
            "package_location": "none",
        }
    )
    _, signals = normalize_vlm_json(package_recipe, content)
    assert {r.outcome for r in signals.values()} == {Outcome.ABSTAINED}
    assert signals["package_present"].reason == "low_quality"


def test_implication_absent_sets_count_and_location(package_recipe: Recipe) -> None:
    _, signals = normalize_vlm_json(
        package_recipe,
        _vlm(package_present="absent", package_count=3, package_location="side"),
    )
    assert signals["package_count"].value == 0
    assert signals["package_location"].value == "none"
    assert signals["package_count"].reason == "implied"


def test_implication_present_fixes_contradictions(package_recipe: Recipe) -> None:
    _, signals = normalize_vlm_json(
        package_recipe,
        _vlm(package_present="present", package_count=0, package_location="none"),
    )
    assert signals["package_count"].value == 1
    assert signals["package_location"].outcome is Outcome.ABSTAINED


def test_implication_abstained_presence_abstains_others(package_recipe: Recipe) -> None:
    _, signals = normalize_vlm_json(
        package_recipe,
        _vlm(package_present="uncertain", package_count=2, package_location="side"),
    )
    assert signals["package_count"].outcome is Outcome.ABSTAINED
    assert signals["package_location"].outcome is Outcome.ABSTAINED


def test_free_text_is_dropped(package_recipe: Recipe) -> None:
    quality, signals = normalize_vlm_json(
        package_recipe,
        _vlm(package_present="absent", note="a person in red is standing there"),
    )
    assert "note" not in quality and "note" not in signals


def _jev(present: float, count: dict[str, float] | None = None) -> dict:
    count = count or {str(i): (1.0 if i == 1 else 0.0) for i in range(11)}
    choice = max(count, key=lambda k: count[k])
    location = {v: 0.0 for v in ("none", "doorstep", "side", "walkway", "multiple")}
    location["doorstep"] = 1.0
    return {
        "package_present": {"type": "noul", "noul": present},
        "package_count": {
            "type": "choice",
            "choice": choice,
            "probabilities": count,
            "confidence": 0.9,
        },
        "package_location": {
            "type": "choice",
            "choice": "doorstep",
            "probabilities": location,
            "confidence": 0.9,
        },
    }


def test_jev_noul_and_choice(package_recipe: Recipe) -> None:
    signals = normalize_jev_answers(package_recipe, _jev(0.8))
    present = signals["package_present"]
    assert present.value == "present"
    assert present.probability == pytest.approx(0.8)
    assert present.distribution == pytest.approx({"present": 0.8, "absent": 0.2})
    assert signals["package_count"].value == 1
    assert isinstance(signals["package_count"].value, int)


def test_jev_threshold_abstains_but_keeps_distribution(package_recipe: Recipe) -> None:
    signals = normalize_jev_answers(package_recipe, _jev(0.7), min_probability=0.9)
    present = signals["package_present"]
    assert present.outcome is Outcome.ABSTAINED
    assert present.reason == "below_threshold"
    assert present.distribution == pytest.approx({"present": 0.7, "absent": 0.3})


def test_jev_tie_abstains(package_recipe: Recipe) -> None:
    signals = normalize_jev_answers(package_recipe, _jev(0.5))
    assert signals["package_present"].reason == "tie"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda a: a["package_present"].update(noul=1.5),
        lambda a: a["package_present"].update(noul=float("nan")),
        lambda a: a["package_present"].update(type="choice"),
        lambda a: a["package_count"]["probabilities"].update({"1": 0.5}),
        lambda a: a["package_count"]["probabilities"].pop("10"),
        lambda a: a["package_count"].update(choice="2"),
        lambda a: a["package_count"].update(choice="11"),
    ],
)
def test_jev_rejects_invalid_distributions(package_recipe: Recipe, mutation) -> None:
    answers = _jev(0.9)
    mutation(answers)
    signals = normalize_jev_answers(package_recipe, answers)
    assert any(r.outcome is Outcome.INVALID for r in signals.values())


def test_jev_missing_answers(package_recipe: Recipe) -> None:
    signals = normalize_jev_answers(package_recipe, {})
    assert signals["package_present"] == SignalReading.invalid("missing")
    assert normalize_jev_answers(package_recipe, "oops")["package_present"].reason == "no_answers"


def test_error_readings_are_errors(package_recipe: Recipe) -> None:
    readings = error_readings(package_recipe, "timeout")
    assert {r.outcome for r in readings.values()} == {Outcome.ERROR}
