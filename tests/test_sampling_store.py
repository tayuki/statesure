from __future__ import annotations

import random
import stat
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from statesure.errors import StoreError
from statesure.readings import SignalReading, new_sample_id
from statesure.recipe import Recipe
from statesure.reconcile import Comparison
from statesure.sampling import ReviewClass, classify_for_review
from statesure.store import HUMAN_UNCERTAIN, LabelStore

from .conftest import T0, presence_obs

OK = {"package_present": SignalReading.decided("absent")}
CONFLICT = Comparison("conflict", (), ("package_present",), (), ())


def _classify(**kwargs) -> ReviewClass:
    params = {
        "primary": OK,
        "primary_failed": False,
        "comparison": None,
        "reasons": (),
        "rng": random.Random(1),
        "save_all": True,
    }
    params.update(kwargs)
    return classify_for_review(**params)


def test_strata_in_priority_order() -> None:
    assert _classify(primary_failed=True, comparison=CONFLICT).stratum == "error"
    errors = {"package_present": SignalReading.error()}
    assert _classify(primary=errors).stratum == "error"
    assert _classify(comparison=CONFLICT, reasons=("change_candidate",)).stratum == "conflict"
    assert _classify(reasons=("change_candidate", "low_quality")).stratum == "change"
    assert _classify(reasons=("low_quality",)).stratum == "low_confidence"
    assert _classify().stratum == "ordinary"


def test_save_all_stores_every_ordinary_sample_with_probability_one() -> None:
    """Regression (PR #212 review 1): priority must not change the weight."""
    rng = random.Random(3)
    classes = [_classify(rng=rng) for _ in range(600)]
    assert all(c.stored and c.inclusion_probability == 1.0 for c in classes)
    routine = sum(c.priority == "routine" for c in classes)
    assert 60 < routine < 140
    assert {c.priority for c in classes} == {"routine", "archive"}


def test_sampled_mode_records_selection_probability() -> None:
    rng = random.Random(5)
    classes = [_classify(rng=rng, save_all=False) for _ in range(600)]
    assert all(c.inclusion_probability == pytest.approx(1 / 6) for c in classes)
    stored = sum(c.stored for c in classes)
    assert 60 < stored < 140


def _store(tmp_path: Path) -> LabelStore:
    return LabelStore(tmp_path / "labels" / "store.sqlite")


def _add(store: LabelStore, recipe: Recipe, *, image: bytes | None = None, **kwargs) -> str:
    sample_id = kwargs.pop("sample_id", new_sample_id())
    store.add_sample(
        sample_id=sample_id,
        source_id="cam-a",
        install_fingerprint=kwargs.pop("install", "install-a"),
        captured_at=kwargs.pop("at", T0),
        recipe=recipe,
        review=kwargs.pop("review", ReviewClass("ordinary", "routine", True, 1.0, 0.5)),
        save_all=True,
        image=image,
        now=T0,
        retention=timedelta(days=7),
    )
    return sample_id


def test_store_permissions(tmp_path: Path, package_recipe: Recipe) -> None:
    store = _store(tmp_path)
    _add(store, package_recipe, image=b"synthetic-bytes")
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.path.parent.stat().st_mode) == 0o700
    frame = next(store.frames_dir.iterdir())
    assert stat.S_IMODE(frame.stat().st_mode) == 0o600


def test_readings_and_labels_round_trip(tmp_path: Path, package_recipe: Recipe) -> None:
    store = _store(tmp_path)
    sample_id = _add(store, package_recipe)
    obs = presence_obs(package_recipe, "present", T0, sample_id=sample_id)
    store.add_observation(obs)
    store.add_confirmed(
        sample_id, obs.judge_id, {"package_present": SignalReading.decided("absent")}
    )
    store.add_label(
        package_recipe,
        sample_id,
        "package_present",
        "present",
        assessable=True,
        reviewer_confidence="confident",
        labeled_at=T0,
    )
    (record,) = store.records(package_recipe.id)
    assert record.readings[(obs.judge_id, "primary", "package_present")].value == "present"
    assert record.readings[(obs.judge_id, "confirmed", "package_present")].value == "absent"
    assert record.labels["package_present"].truth == "present"


def test_labels_are_validated(tmp_path: Path, package_recipe: Recipe) -> None:
    store = _store(tmp_path)
    sample_id = _add(store, package_recipe)

    def label(**kwargs) -> None:
        params = {
            "assessable": True,
            "reviewer_confidence": "confident",
            "labeled_at": T0,
        }
        params.update(kwargs)
        truth = params.pop("truth", "present")
        signal = params.pop("signal", "package_present")
        store.add_label(package_recipe, sample_id, signal, truth, **params)

    for kwargs, code in [
        ({"label_source": "teacher"}, "only_human_labels"),
        ({"truth": "maybe"}, "invalid_truth"),
        ({"truth": "uncertain"}, "invalid_truth"),
        ({"signal": "ghost"}, "unknown_signal"),
        ({"reviewer_confidence": "sure"}, "invalid_reviewer_confidence"),
        ({"assessable": False}, "truth_without_assessable"),
    ]:
        with pytest.raises(StoreError) as info:
            label(**kwargs)
        assert info.value.code == code
    label(truth=HUMAN_UNCERTAIN)
    label(truth=None, assessable=False)


def test_unselected_and_duplicate_samples(tmp_path: Path, package_recipe: Recipe) -> None:
    store = _store(tmp_path)
    with pytest.raises(StoreError):
        _add(store, package_recipe, review=ReviewClass("ordinary", "routine", False, 0.2, 0.1))
    sample_id = _add(store, package_recipe, image=b"first")
    with pytest.raises(StoreError):
        _add(store, package_recipe, image=b"second", sample_id=sample_id)
    # The existing frame is untouched by the rejected duplicate.
    assert store.image_path(sample_id).read_bytes() == b"first"


def test_purge_is_time_based_and_idempotent(tmp_path: Path, package_recipe: Recipe) -> None:
    """Regression (#4): expiry does not depend on the store being accessed."""
    store = _store(tmp_path)
    sample_id = _add(store, package_recipe, image=b"synthetic")
    assert store.purge_expired(T0 + timedelta(days=6)) == 0
    assert store.image_path(sample_id) is not None
    assert store.purge_expired(T0 + timedelta(days=7)) == 1
    assert store.image_path(sample_id) is None
    assert not any(store.frames_dir.iterdir())
    assert store.purge_expired(T0 + timedelta(days=8)) == 0


def test_observation_must_match_sample(tmp_path: Path, package_recipe: Recipe) -> None:
    store = _store(tmp_path)
    sample_id = _add(store, package_recipe)
    other = presence_obs(package_recipe, "present", T0, sample_id=sample_id, install="install-b")
    with pytest.raises(StoreError) as info:
        store.add_observation(other)
    assert info.value.code == "observation_sample_mismatch"


def test_labels_require_the_same_recipe_version(tmp_path: Path, package_recipe: Recipe) -> None:
    """Regression (statesure#1 review): labels follow the sample's recipe version."""
    store = _store(tmp_path)
    sample_id = _add(store, package_recipe)
    newer = replace(package_recipe, fingerprint="f" * 16)
    with pytest.raises(StoreError) as info:
        store.add_label(
            newer,
            sample_id,
            "package_present",
            "present",
            assessable=True,
            reviewer_confidence="confident",
            labeled_at=T0,
        )
    assert info.value.code == "label_recipe_mismatch"
