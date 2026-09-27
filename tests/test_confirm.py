from __future__ import annotations

import random
from dataclasses import replace
from datetime import timedelta

from statesure.confirm import (
    SignalState,
    StateKey,
    change_event_type,
    confirmed_reading,
    replay,
    step,
)
from statesure.readings import Outcome, SignalReading
from statesure.recipe import Confirmation, Recipe

from .conftest import T0, presence_obs

CFG = Confirmation(samples=2, max_gap=timedelta(hours=2))
KEY = StateKey("cam-a", "install-a", "package_at_door", "fp", "judge", "package_present")
HALF_HOUR = timedelta(minutes=30)


def _run(values: list[str | None], start=T0, gap=HALF_HOUR):
    state = SignalState()
    events = []
    for index, value in enumerate(values):
        reading = (
            SignalReading.decided(value) if value else SignalReading.abstained("model_uncertain")
        )
        state, event = step(
            state, reading, start + gap * index, CFG, key=KEY, sample_id=f"{index:032x}"
        )
        if event:
            events.append(event)
    return state, events


def test_two_consecutive_samples_confirm() -> None:
    state, events = _run(["absent", "absent"])
    assert state.confirmed_value == "absent"
    assert [e.event_type for e in events] == ["baseline_confirmed"]
    assert events[0].first_seen_at == T0
    assert events[0].confirmed_at == T0 + HALF_HOUR


def test_single_sample_does_not_change_state() -> None:
    state, events = _run(["absent", "absent", "present", "absent"])
    assert state.confirmed_value == "absent"
    assert len(events) == 1


def test_abstention_breaks_streak_but_keeps_state() -> None:
    state, events = _run(["absent", "absent", "present", None, "present"])
    # "could not see" does not end the confirmed "absent" and resets the streak.
    assert state.confirmed_value == "absent"
    assert len(events) == 1
    state, events = _run(["absent", "absent", "present", None, "present", "present"])
    assert state.confirmed_value == "present"
    assert events[-1].event_type == "appeared"
    assert events[-1].previous_last_seen_at == T0 + HALF_HOUR
    assert events[-1].first_seen_at == T0 + HALF_HOUR * 4


def test_overnight_gap_is_not_consecutive() -> None:
    """Regression (#5): 22:30 then 07:00 must not confirm with max_gap=2h."""
    evening = T0.replace(hour=22, minute=30)
    state = SignalState()
    state, event = step(
        state, SignalReading.decided("present"), evening, CFG, key=KEY, sample_id="0" * 32
    )
    morning = evening + timedelta(hours=8, minutes=30)
    state, event = step(
        state, SignalReading.decided("present"), morning, CFG, key=KEY, sample_id="1" * 32
    )
    assert event is None and state.confirmed_value is None
    state, event = step(
        state,
        SignalReading.decided("present"),
        morning + HALF_HOUR,
        CFG,
        key=KEY,
        sample_id="2" * 32,
    )
    assert event is not None and event.first_seen_at == morning


def test_out_of_order_and_duplicate_samples_are_ignored() -> None:
    state, _ = _run(["absent", "absent"])
    same, event = step(
        state, SignalReading.decided("present"), T0, CFG, key=KEY, sample_id="f" * 32
    )
    assert same == state and event is None


def test_event_types() -> None:
    assert change_event_type(0, 2) == "appeared"
    assert change_event_type(2, 0) == "cleared"
    assert change_event_type(1, 3) == "increased"
    assert change_event_type(3, 1) == "decreased"
    assert change_event_type("side", "doorstep") == "changed"
    assert change_event_type("present", "absent") == "cleared"


def test_replay_is_deterministic(package_recipe: Recipe) -> None:
    values = ["absent", "absent", "present", "present", None, "absent", "absent"]
    observations = [
        presence_obs(package_recipe, v, T0 + HALF_HOUR * i) for i, v in enumerate(values)
    ]
    configs = {package_recipe.fingerprint: package_recipe.confirmation}
    expected = replay(observations, configs)
    shuffled = observations[:]
    random.Random(7).shuffle(shuffled)
    again = replay(shuffled, configs)
    assert [e.to_dict() for e in again.events] == [e.to_dict() for e in expected.events]
    assert [e.event_type for e in expected.events] == ["baseline_confirmed", "appeared", "cleared"]


def test_replay_prefers_merged_and_tracks_per_sample(package_recipe: Recipe) -> None:
    first = presence_obs(package_recipe, "present", T0)
    merged = presence_obs(package_recipe, "absent", T0, stage="merged", sample_id=first.sample_id)
    second = presence_obs(package_recipe, "absent", T0 + HALF_HOUR)
    result = replay(
        [first, merged, second], {package_recipe.fingerprint: package_recipe.confirmation}
    )
    assert result.events[0].to_value == "absent"
    judge = first.judge_id
    assert result.confirmed_at_sample[(first.sample_id, judge, "package_present")] is None
    assert result.confirmed_at_sample[(second.sample_id, judge, "package_present")] == "absent"


def test_replay_keys_by_installation(package_recipe: Recipe) -> None:
    observations = [
        presence_obs(package_recipe, "present", T0, install="install-a"),
        presence_obs(package_recipe, "present", T0 + HALF_HOUR, install="install-b"),
    ]
    result = replay(observations, {package_recipe.fingerprint: package_recipe.confirmation})
    # Changing the installation starts a new state instead of continuing a streak.
    assert result.events == ()
    assert len(result.states) == 2


def test_confirmed_reading() -> None:
    assert confirmed_reading(None).outcome is Outcome.ABSTAINED
    assert confirmed_reading("present") == SignalReading.decided("present")


def test_replay_keeps_recipe_versions_apart(package_recipe: Recipe) -> None:
    """Regression (statesure#1 review): versions never share a streak."""
    old = presence_obs(package_recipe, "present", T0)
    new = presence_obs(package_recipe, "present", T0 + HALF_HOUR)
    old = replace(old, recipe_fingerprint="0" * 16)
    configs = {
        package_recipe.fingerprint: package_recipe.confirmation,
        "0" * 16: package_recipe.confirmation,
    }
    assert replay([old, new], configs).events == ()
    # An unknown version is skipped instead of borrowing another version's settings.
    only_current = replay([old, new], {package_recipe.fingerprint: package_recipe.confirmation})
    assert len(only_current.states) == 1
