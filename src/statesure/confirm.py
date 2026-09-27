"""Temporal confirmation.

A value becomes the confirmed state only after it is read in ``samples``
consecutive samples. Abstentions and errors break the streak but never change
the confirmed state, so "could not see" is never treated as "gone". A gap longer
than ``max_gap`` (for example overnight) also breaks the streak.

The ledger is rebuilt deterministically from observations with ``replay``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import NamedTuple

from .readings import Observation, SignalReading
from .recipe import Confirmation

EVENT_SCHEMA_VERSION = 1


class StateKey(NamedTuple):
    """Confirmation state is never shared across installations or recipe versions."""

    source_id: str
    install_fingerprint: str
    recipe_id: str
    recipe_fingerprint: str
    judge_id: str
    signal: str


@dataclass(frozen=True)
class SignalState:
    confirmed_value: str | int | None = None
    confirmed_since: datetime | None = None
    confirmed_last_seen_at: datetime | None = None
    candidate_value: str | int | None = None
    candidate_first_seen_at: datetime | None = None
    candidate_previous_last_seen_at: datetime | None = None
    candidate_samples: int = 0
    last_sample_at: datetime | None = None

    def without_candidate(self) -> SignalState:
        return replace(
            self,
            candidate_value=None,
            candidate_first_seen_at=None,
            candidate_previous_last_seen_at=None,
            candidate_samples=0,
        )


@dataclass(frozen=True)
class ChangeEvent:
    """A confirmed change. The change happened somewhere in
    (previous_last_seen_at, first_seen_at]; it is not an exact time."""

    key: StateKey
    event_type: str
    from_value: str | int | None
    to_value: str | int
    previous_confirmed_since: datetime | None
    previous_last_seen_at: datetime | None
    first_seen_at: datetime
    confirmed_at: datetime
    confirming_sample_id: str
    confirmation_samples: int

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": EVENT_SCHEMA_VERSION,
            **self.key._asdict(),
            "event_type": self.event_type,
            "from_value": self.from_value,
            "to_value": self.to_value,
            "previous_confirmed_since": _iso(self.previous_confirmed_since),
            "previous_last_seen_at": _iso(self.previous_last_seen_at),
            "first_seen_at": self.first_seen_at.isoformat(),
            "confirmed_at": self.confirmed_at.isoformat(),
            "confirming_sample_id": self.confirming_sample_id,
            "confirmation_samples": self.confirmation_samples,
        }


def step(
    state: SignalState,
    reading: SignalReading,
    at: datetime,
    config: Confirmation,
    *,
    key: StateKey,
    sample_id: str,
) -> tuple[SignalState, ChangeEvent | None]:
    """Advance one signal's state by one sample."""
    if state.last_sample_at is not None and at <= state.last_sample_at:
        return state, None
    if state.last_sample_at is not None and at - state.last_sample_at > config.max_gap:
        state = state.without_candidate()
    state = replace(state, last_sample_at=at)

    if not reading.is_decided:
        return state.without_candidate(), None
    value = reading.value
    if value == state.confirmed_value:
        return replace(state.without_candidate(), confirmed_last_seen_at=at), None
    if value == state.candidate_value:
        state = replace(state, candidate_samples=state.candidate_samples + 1)
    else:
        state = replace(
            state,
            candidate_value=value,
            candidate_first_seen_at=at,
            candidate_previous_last_seen_at=state.confirmed_last_seen_at,
            candidate_samples=1,
        )
    if state.candidate_samples < config.samples:
        return state, None

    previous = state.confirmed_value
    event = ChangeEvent(
        key=key,
        event_type=(
            "baseline_confirmed" if previous is None else change_event_type(previous, value)
        ),
        from_value=previous,
        to_value=value,
        previous_confirmed_since=state.confirmed_since,
        previous_last_seen_at=state.candidate_previous_last_seen_at,
        first_seen_at=state.candidate_first_seen_at or at,
        confirmed_at=at,
        confirming_sample_id=sample_id,
        confirmation_samples=config.samples,
    )
    confirmed = replace(
        state.without_candidate(),
        confirmed_value=value,
        confirmed_since=event.first_seen_at,
        confirmed_last_seen_at=at,
    )
    return confirmed, event


def change_event_type(previous: str | int, current: str | int) -> str:
    if isinstance(previous, int) and isinstance(current, int):
        if previous == 0 and current > 0:
            return "appeared"
        if previous > 0 and current == 0:
            return "cleared"
        return "increased" if current > previous else "decreased"
    if previous == "absent" and current == "present":
        return "appeared"
    if previous == "present" and current == "absent":
        return "cleared"
    return "changed"


@dataclass(frozen=True)
class ReplayResult:
    events: tuple[ChangeEvent, ...]
    states: Mapping[StateKey, SignalState]
    confirmed_at_sample: Mapping[tuple[str, str, str], str | int | None]
    """(sample_id, judge_id, signal) -> confirmed value right after that sample."""


def replay(
    observations: Iterable[Observation], configs: Mapping[str, Confirmation]
) -> ReplayResult:
    """Rebuild the confirmation ledger from observations.

    For each (sample, judge) the ``merged`` observation is used when present,
    otherwise ``primary``. ``configs`` maps a recipe *fingerprint* to its
    confirmation settings; observations of other versions are skipped.
    """
    chosen: dict[tuple[str, str], Observation] = {}
    for observation in observations:
        if observation.stage == "verified":
            continue
        slot = (observation.sample_id, observation.judge_id)
        if observation.stage == "merged" or slot not in chosen:
            chosen[slot] = observation

    ordered = sorted(
        chosen.values(), key=lambda o: (o.captured_at, o.source_id, o.sample_id, o.judge_id)
    )
    states: dict[StateKey, SignalState] = {}
    events: list[ChangeEvent] = []
    confirmed_at_sample: dict[tuple[str, str, str], str | int | None] = {}
    for observation in ordered:
        config = configs.get(observation.recipe_fingerprint)
        if config is None:
            continue
        for signal, reading in observation.signals.items():
            key = StateKey(
                observation.source_id,
                observation.install_fingerprint,
                observation.recipe_id,
                observation.recipe_fingerprint,
                observation.judge_id,
                signal,
            )
            state, event = step(
                states.get(key, SignalState()),
                reading,
                observation.captured_at,
                config,
                key=key,
                sample_id=observation.sample_id,
            )
            states[key] = state
            if event is not None:
                events.append(event)
            confirmed_at_sample[(observation.sample_id, observation.judge_id, signal)] = (
                state.confirmed_value
            )
    return ReplayResult(tuple(events), states, confirmed_at_sample)


def confirmed_reading(value: str | int | None) -> SignalReading:
    """The ``confirmed`` stage reading for evaluation."""
    if value is None:
        return SignalReading.abstained("not_confirmed")
    return SignalReading.decided(value)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


__all__ = [
    "ChangeEvent",
    "ReplayResult",
    "SignalState",
    "StateKey",
    "change_event_type",
    "confirmed_reading",
    "replay",
    "step",
]
