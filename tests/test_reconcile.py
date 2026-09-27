from __future__ import annotations

from statesure.readings import Outcome, SignalReading
from statesure.recipe import Recipe
from statesure.reconcile import needs_verification, reconcile

D = SignalReading.decided
A = SignalReading.abstained


def _readings(present, count=None, location=None) -> dict[str, SignalReading]:
    return {
        "package_present": present,
        "package_count": count or A("model_uncertain"),
        "package_location": location or A("model_uncertain"),
    }


def test_agreement_is_kept(package_recipe: Recipe) -> None:
    merged, comparison = reconcile(
        package_recipe,
        _readings(D("present"), D(1), D("doorstep")),
        _readings(D("present"), D(1), D("doorstep")),
    )
    assert merged["package_present"] == D("present")
    assert comparison.agreement == "full"


def test_disagreement_abstains(package_recipe: Recipe) -> None:
    merged, comparison = reconcile(
        package_recipe,
        _readings(D("present"), D(1), D("doorstep")),
        _readings(D("absent"), D(0), D("none")),
    )
    assert merged["package_present"].reason == "conflict"
    assert "package_present" in comparison.disagreed
    assert comparison.agreement == "conflict"
    # Implications are re-applied: presence abstained -> others abstain.
    assert merged["package_count"].outcome is Outcome.ABSTAINED


def test_unconfirmed_abstains(package_recipe: Recipe) -> None:
    merged, comparison = reconcile(
        package_recipe,
        _readings(D("present"), D(1), D("doorstep")),
        _readings(A("model_uncertain")),
    )
    assert merged["package_present"].reason == "unconfirmed"
    assert comparison.agreement == "conflict"
    assert "package_present" in comparison.unconfirmed


def test_verification_recovers(package_recipe: Recipe) -> None:
    merged, comparison = reconcile(
        package_recipe,
        _readings(A("not_visible")),
        _readings(D("absent"), D(0), D("none")),
    )
    assert merged["package_present"] == D("absent")
    assert comparison.agreement == "verification_only"


def test_partial(package_recipe: Recipe) -> None:
    _, comparison = reconcile(
        package_recipe,
        _readings(D("present"), A("model_uncertain"), A("model_uncertain")),
        _readings(D("present"), D(2), A("model_uncertain")),
    )
    assert comparison.agreement == "partial"


def test_no_comparable(package_recipe: Recipe) -> None:
    merged, comparison = reconcile(
        package_recipe, _readings(A("model_uncertain")), _readings(SignalReading.error())
    )
    assert comparison.agreement == "no_comparable_signals"
    assert merged["package_present"].outcome is Outcome.ABSTAINED


def test_needs_verification_reasons(package_recipe: Recipe) -> None:
    primary = _readings(D("present"), D(1), D("doorstep"))
    assert needs_verification(package_recipe, {}, primary, {}) == ("baseline_unconfirmed",)
    confirmed = {"package_present": "absent", "package_count": 0, "package_location": "none"}
    assert needs_verification(package_recipe, {}, primary, confirmed) == ("change_candidate",)
    same = {"package_present": "present", "package_count": 1, "package_location": "doorstep"}
    assert needs_verification(package_recipe, {}, primary, same) == ()
    reasons = needs_verification(
        package_recipe, {"visibility": "poor"}, primary, same, always_verify=True
    )
    assert reasons == ("always_verify", "low_quality")
