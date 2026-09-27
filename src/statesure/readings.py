"""Data types for judge readings.

An abstention is not a value. Every signal reading has an explicit outcome, so
"the judge could not see the doorstep" is never confused with "there is no
package".
"""

from __future__ import annotations

import math
import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from .errors import InputError

Stage = Literal["primary", "verified", "merged"]
STAGES: tuple[Stage, ...] = ("primary", "verified", "merged")

_SAMPLE_ID = re.compile(r"^[0-9a-f]{32}$")
_TOKEN = re.compile(r"^[A-Za-z0-9_.:@/-]{1,128}$")
_REASON = re.compile(r"^[a-z][a-z0-9_:]{0,63}$")


class Outcome(StrEnum):
    """How a signal reading ended."""

    DECIDED = "decided"
    ABSTAINED = "abstained"
    INVALID = "invalid"
    ERROR = "error"


@dataclass(frozen=True)
class SignalReading:
    """The reading of one signal by one judge at one stage."""

    outcome: Outcome
    value: str | int | None = None
    probability: float | None = None
    distribution: Mapping[str, float] | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, Outcome):
            raise InputError("invalid_outcome")
        if (self.outcome is Outcome.DECIDED) != (self.value is not None):
            raise InputError("value_outcome_mismatch")
        if self.value is not None and type(self.value) not in (str, int):
            raise InputError("invalid_value_type")
        if self.probability is not None and not _is_probability(self.probability):
            raise InputError("invalid_probability")
        if self.distribution is not None:
            if not isinstance(self.distribution, Mapping) or not self.distribution:
                raise InputError("invalid_distribution")
            for key, prob in self.distribution.items():
                if not isinstance(key, str) or not _is_probability(prob):
                    raise InputError("invalid_distribution")
        if self.reason is not None and not _REASON.fullmatch(self.reason):
            raise InputError("invalid_reason")

    @classmethod
    def decided(
        cls,
        value: str | int,
        *,
        probability: float | None = None,
        distribution: Mapping[str, float] | None = None,
        reason: str | None = None,
    ) -> SignalReading:
        return cls(Outcome.DECIDED, value, probability, distribution, reason)

    @classmethod
    def abstained(
        cls, reason: str, *, distribution: Mapping[str, float] | None = None
    ) -> SignalReading:
        return cls(Outcome.ABSTAINED, None, None, distribution, reason)

    @classmethod
    def invalid(cls, reason: str) -> SignalReading:
        return cls(Outcome.INVALID, reason=reason)

    @classmethod
    def error(cls, reason: str = "judge_error") -> SignalReading:
        return cls(Outcome.ERROR, reason=reason)

    @property
    def is_decided(self) -> bool:
        return self.outcome is Outcome.DECIDED

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "value": self.value,
            "probability": self.probability,
            "distribution": dict(self.distribution) if self.distribution else None,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: object) -> SignalReading:
        if not isinstance(data, Mapping) or set(data) != {
            "outcome",
            "value",
            "probability",
            "distribution",
            "reason",
        }:
            raise InputError("invalid_reading_fields")
        try:
            outcome = Outcome(data["outcome"])
        except ValueError:
            raise InputError("invalid_outcome") from None
        return cls(
            outcome,
            data["value"],
            data["probability"],
            data["distribution"],
            data["reason"],
        )


@dataclass(frozen=True)
class Observation:
    """All signal readings of one judge at one stage for one sample."""

    sample_id: str
    source_id: str
    install_fingerprint: str
    captured_at: datetime
    recipe_id: str
    recipe_fingerprint: str
    judge_id: str
    stage: Stage
    signals: Mapping[str, SignalReading]
    quality: Mapping[str, str] = field(default_factory=dict)
    error_class: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not _SAMPLE_ID.fullmatch(self.sample_id):
            raise InputError("invalid_sample_id")
        for value in (
            self.source_id,
            self.install_fingerprint,
            self.recipe_id,
            self.recipe_fingerprint,
            self.judge_id,
        ):
            if not isinstance(value, str) or not _TOKEN.fullmatch(value):
                raise InputError("invalid_identifier")
        if not isinstance(self.captured_at, datetime) or self.captured_at.tzinfo is None:
            raise InputError("naive_timestamp")
        if self.stage not in STAGES:
            raise InputError("invalid_stage")
        if not isinstance(self.signals, Mapping):
            raise InputError("invalid_signals")
        for name, reading in self.signals.items():
            if not isinstance(name, str) or not isinstance(reading, SignalReading):
                raise InputError("invalid_signals")
        for key, value in self.quality.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise InputError("invalid_quality")
        if self.error_class is not None and not _TOKEN.fullmatch(self.error_class):
            raise InputError("invalid_error_class")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_id": self.source_id,
            "install_fingerprint": self.install_fingerprint,
            "captured_at": self.captured_at.isoformat(),
            "recipe_id": self.recipe_id,
            "recipe_fingerprint": self.recipe_fingerprint,
            "judge_id": self.judge_id,
            "stage": self.stage,
            "signals": {name: reading.to_dict() for name, reading in self.signals.items()},
            "quality": dict(self.quality),
            "error_class": self.error_class,
        }

    @classmethod
    def from_dict(cls, data: object) -> Observation:
        keys = {
            "sample_id",
            "source_id",
            "install_fingerprint",
            "captured_at",
            "recipe_id",
            "recipe_fingerprint",
            "judge_id",
            "stage",
            "signals",
            "quality",
            "error_class",
        }
        if not isinstance(data, Mapping) or set(data) != keys:
            raise InputError("invalid_observation_fields")
        try:
            captured_at = datetime.fromisoformat(data["captured_at"])
        except (TypeError, ValueError):
            raise InputError("invalid_timestamp") from None
        signals = data["signals"]
        if not isinstance(signals, Mapping):
            raise InputError("invalid_signals")
        quality = data["quality"]
        if not isinstance(quality, Mapping):
            raise InputError("invalid_quality")
        return cls(
            sample_id=data["sample_id"],
            source_id=data["source_id"],
            install_fingerprint=data["install_fingerprint"],
            captured_at=captured_at,
            recipe_id=data["recipe_id"],
            recipe_fingerprint=data["recipe_fingerprint"],
            judge_id=data["judge_id"],
            stage=data["stage"],
            signals={name: SignalReading.from_dict(r) for name, r in signals.items()},
            quality=dict(quality),
            error_class=data["error_class"],
        )


def new_sample_id() -> str:
    """Return a random sample id. It is never derived from time or image content."""
    return secrets.token_hex(16)


def _is_probability(value: object) -> bool:
    return (
        type(value) in (int, float) and math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0
    )
