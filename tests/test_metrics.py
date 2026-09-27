from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from statesure.confirm import replay
from statesure.metrics import (
    EvalKey,
    build_report,
    choose_min_probability,
    evaluate_promotion,
    evaluate_signal,
    event_metrics,
    ratio,
    stratum_weights,
    wilson,
)
from statesure.readings import SignalReading, new_sample_id
from statesure.recipe import Recipe
from statesure.sampling import ReviewClass
from statesure.store import LabelStore

from .conftest import T0, TZ, presence_obs

JUDGE = "vlm_openai:test-model"
SIGNAL = "package_present"
ORDINARY = ReviewClass("ordinary", "routine", True, 1.0, 0.5)


class Builder:
    """Fill a store with synthetic samples for one presence signal."""

    def __init__(self, tmp_path: Path, recipe: Recipe) -> None:
        self.store = LabelStore(tmp_path / "store.sqlite")
        self.recipe = recipe
        self.count = 0

    def add(
        self,
        truth: str | None,
        readings: dict[str, SignalReading],
        *,
        review: ReviewClass = ORDINARY,
        install: str = "install-a",
        at=None,
    ) -> str:
        sample_id = new_sample_id()
        at = at or T0 + timedelta(minutes=30 * self.count)
        self.count += 1
        self.store.add_sample(
            sample_id=sample_id,
            source_id="cam-a",
            install_fingerprint=install,
            captured_at=at,
            recipe=self.recipe,
            review=review,
            save_all=True,
        )
        for stage, reading in readings.items():
            if stage == "confirmed":
                self.store.add_confirmed(sample_id, JUDGE, {SIGNAL: reading})
            else:
                self.store.add_observation(
                    presence_obs(
                        self.recipe,
                        None,
                        at,
                        stage=stage,
                        install=install,
                        sample_id=sample_id,
                        reading=reading,
                    )
                )
        if truth is not None:
            self.store.add_label(
                self.recipe,
                sample_id,
                SIGNAL,
                truth,
                assessable=True,
                reviewer_confidence="confident",
                labeled_at=at,
            )
        return sample_id

    def metrics(self, stage: str = "primary", install: str = "install-a"):
        records = self.store.records(self.recipe.id)
        key = EvalKey(install, self.recipe.id, self.recipe.fingerprint, JUDGE, stage, SIGNAL)
        return evaluate_signal(self.recipe, records, key, stratum_weights(records, SIGNAL))


D = SignalReading.decided


def test_wilson_matches_reference() -> None:
    low, high = wilson(0.8, 10)
    assert low == pytest.approx(0.4902, abs=1e-4)
    assert high == pytest.approx(0.9433, abs=1e-4)
    assert wilson(0.5, 0) is None


def test_ratio_uses_effective_sample_size() -> None:
    uniform = ratio([(1.0, True)] * 8 + [(1.0, False)] * 2)
    skewed = ratio([(5.0, True)] + [(1.0, True)] * 7 + [(1.0, False)] * 2)
    assert uniform.n_eff == pytest.approx(10)
    assert skewed.n_eff < 10


def test_recall_counts_abstentions_and_failures_as_misses(
    tmp_path: Path, package_recipe: Recipe
) -> None:
    """Regression (P1(b)): abstentions stay in the recall denominator."""
    b = Builder(tmp_path, package_recipe)
    b.add("present", {"primary": D("present")})
    b.add("present", {"primary": SignalReading.abstained("model_uncertain")})
    b.add("present", {"primary": SignalReading.error("timeout")})
    presence = b.metrics().presence
    assert presence.recall_all.value == pytest.approx(1 / 3)
    assert presence.recall_when_decided.value == pytest.approx(1.0)
    assert (presence.miss_by_abstain, presence.miss_by_failure) == (1, 1)


def test_failed_inference_stays_in_population(tmp_path: Path, package_recipe: Recipe) -> None:
    """Regression (P1(a)): a judge failure is stored, reviewed and scored."""
    b = Builder(tmp_path, package_recipe)
    error_class = ReviewClass("error", "critical", True, 1.0, 0.1)
    b.add("present", {"primary": SignalReading.error("timeout")}, review=error_class)
    metrics = b.metrics()
    assert metrics.scored == 1
    assert metrics.counts["failed"] == 1
    assert metrics.labeled_by_stratum == {"error": 1}
    assert metrics.presence.recall_all.value == 0.0


def test_stages_are_measured_separately(tmp_path: Path, package_recipe: Recipe) -> None:
    """Regression (#3): primary, merged and confirmed are distinct results."""
    b = Builder(tmp_path, package_recipe)
    b.add(
        "present",
        {
            "primary": D("absent"),
            "merged": D("present"),
            "confirmed": SignalReading.abstained("not_confirmed"),
        },
    )
    assert b.metrics("primary").counts["wrong"] == 1
    assert b.metrics("merged").counts["correct"] == 1
    assert b.metrics("confirmed").counts["abstained"] == 1


def test_human_uncertain_is_excluded(tmp_path: Path, package_recipe: Recipe) -> None:
    b = Builder(tmp_path, package_recipe)
    b.add("human_uncertain", {"primary": D("present")})
    b.add(None, {"primary": D("present")})
    assert b.metrics().scored == 0


def test_save_all_weights_are_uniform_within_ordinary(
    tmp_path: Path, package_recipe: Recipe
) -> None:
    """Regression (PR #212 review 1)."""
    b = Builder(tmp_path, package_recipe)
    routine_ids = []
    for index in range(60):
        routine = index % 6 == 0
        review = ReviewClass("ordinary", "routine" if routine else "archive", True, 1.0, 0.5)
        sample_id = b.add("absent" if routine else None, {"primary": D("absent")}, review=review)
        if routine:
            routine_ids.append(sample_id)
    weights = stratum_weights(b.store.records(package_recipe.id), SIGNAL)
    assert [weights[sample_id] for sample_id in routine_ids] == pytest.approx([6.0] * 10)


def test_sampled_mode_weights_by_inclusion_probability(
    tmp_path: Path, package_recipe: Recipe
) -> None:
    b = Builder(tmp_path, package_recipe)
    sampled = ReviewClass("ordinary", "routine", True, 1 / 6, 0.5)
    change = ReviewClass("change", "high", True, 1.0, 0.5)
    ordinary_id = b.add("absent", {"primary": D("absent")}, review=sampled)
    change_id = b.add("present", {"primary": D("present")}, review=change)
    weights = stratum_weights(b.store.records(package_recipe.id), SIGNAL)
    assert weights[ordinary_id] == pytest.approx(6.0)
    assert weights[change_id] == pytest.approx(1.0)


def _fill_good(b: Builder, install: str, positives: int = 40, negatives: int = 60) -> None:
    # With no errors, the Wilson lower bound reaches 0.90 only from about 35 samples.
    for _ in range(positives):
        b.add("present", {"confirmed": D("present")}, install=install)
    for _ in range(negatives):
        b.add("absent", {"confirmed": D("absent")}, install=install)


def test_promotion_eligible_and_per_installation(tmp_path: Path, package_recipe: Recipe) -> None:
    """Regression (PR #212 review 2): installations are evaluated separately."""
    b = Builder(tmp_path, package_recipe)
    _fill_good(b, "install-a")
    for _ in range(3):
        b.add("present", {"confirmed": D("present")}, install="install-b")
    report = build_report(package_recipe, b.store.records(package_recipe.id))
    verdicts = report.promotions
    assert verdicts[f"install-a/{JUDGE}/{SIGNAL}"].status == "eligible"
    assert verdicts[f"install-b/{JUDGE}/{SIGNAL}"].status == "not_enough_data"
    assert "min_positive" in verdicts[f"install-b/{JUDGE}/{SIGNAL}"].unmet


def test_promotion_below_threshold(tmp_path: Path, package_recipe: Recipe) -> None:
    b = Builder(tmp_path, package_recipe)
    for index in range(80):
        reading = D("present") if index % 2 else SignalReading.abstained("model_uncertain")
        b.add("present", {"confirmed": reading})
    for _ in range(60):
        b.add("absent", {"confirmed": D("absent")})
    verdict = evaluate_promotion(package_recipe, b.metrics("confirmed"))
    assert verdict.status == "below_threshold"
    assert verdict.unmet == ("recall_lower_bound",)
    assert verdict.action == "manual_review_required"


def test_promotion_needs_ordinary_labels(tmp_path: Path, package_recipe: Recipe) -> None:
    b = Builder(tmp_path, package_recipe)
    change = ReviewClass("change", "high", True, 1.0, 0.5)
    for _ in range(25):
        b.add("present", {"confirmed": D("present")}, review=change)
    for _ in range(60):
        b.add("absent", {"confirmed": D("absent")}, review=change)
    verdict = evaluate_promotion(package_recipe, b.metrics("confirmed"))
    assert verdict.unmet == ("ordinary_labels",)


def _jev_reading(p_present: float) -> SignalReading:
    distribution = {"present": p_present, "absent": 1 - p_present}
    value = "present" if p_present > 0.5 else "absent"
    return SignalReading.decided(value, probability=distribution[value], distribution=distribution)


def test_calibration_of_a_calibrated_judge(tmp_path: Path, package_recipe: Recipe) -> None:
    b = Builder(tmp_path, package_recipe)
    # 10 samples at p=0.9 with 9 hits; 10 at p=0.1 with 1 hit: perfectly calibrated.
    for index in range(10):
        b.add("present" if index < 9 else "absent", {"primary": _jev_reading(0.9)})
        b.add("present" if index < 1 else "absent", {"primary": _jev_reading(0.1)})
    calibration = b.metrics().calibration
    assert calibration.n == 20
    assert calibration.ece == pytest.approx(0.0, abs=1e-9)
    assert calibration.brier == pytest.approx(2 * 0.09, abs=1e-9)


def test_threshold_is_chosen_and_measured_on_different_days(
    tmp_path: Path, package_recipe: Recipe
) -> None:
    b = Builder(tmp_path, package_recipe)
    for day in range(4):
        base = T0 + timedelta(days=day)
        for index in range(20):
            at = base + timedelta(minutes=10 * index)
            b.add("present", {"primary": _jev_reading(0.95)}, at=at)
            b.add("absent", {"primary": _jev_reading(0.6)}, at=at + timedelta(minutes=1))
            b.add("absent", {"primary": _jev_reading(0.05)}, at=at + timedelta(minutes=2))
    choice = choose_min_probability(
        package_recipe,
        b.store.records(package_recipe.id),
        install_fingerprint="install-a",
        judge_id=JUDGE,
        signal=SIGNAL,
        tz=TZ,
        target_precision=0.9,
    )
    assert choice.threshold == pytest.approx(0.61)
    assert (choice.selection_days, choice.evaluation_days) == (2, 2)
    assert choice.evaluation_precision.value == 1.0
    assert choice.evaluation_recall_all.value == 1.0


def test_threshold_none_when_target_unreachable(tmp_path: Path, package_recipe: Recipe) -> None:
    b = Builder(tmp_path, package_recipe)
    for index in range(20):
        b.add("absent", {"primary": _jev_reading(0.99)}, at=T0 + timedelta(minutes=index))
    choice = choose_min_probability(
        package_recipe,
        b.store.records(package_recipe.id),
        install_fingerprint="install-a",
        judge_id=JUDGE,
        signal=SIGNAL,
        tz=TZ,
        target_precision=0.9,
    )
    assert choice.threshold is None


def test_event_metrics(tmp_path: Path, package_recipe: Recipe) -> None:
    b = Builder(tmp_path, package_recipe)
    observations = []
    truths = ["absent", "absent", "absent", "present"]
    values = ["absent", "absent", "present", "present"]
    for index, (truth, value) in enumerate(zip(truths, values, strict=True)):
        at = T0 + timedelta(minutes=30 * index)
        sample_id = b.add(truth, {"primary": D(value)}, at=at)
        observations.append(presence_obs(package_recipe, value, at, sample_id=sample_id))
    result = replay(observations, {package_recipe.id: package_recipe.confirmation})
    metrics = event_metrics(result.events, b.store.records(package_recipe.id))
    assert metrics.events == 2
    assert metrics.labeled_events == 2
    assert metrics.false_events == 0
    assert metrics.latency_seconds_max == 1800
