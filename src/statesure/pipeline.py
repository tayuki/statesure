"""One observation cycle for one installation.

capture -> primary judgment -> (re-check) -> reconcile -> observation log ->
confirmation -> review store -> purge expired images.

Every judged sample is appended to the observation log (structured readings
only, no pixels), so confirmation never depends on which samples were kept for
review. Image bytes live only in memory, except for samples kept for review,
which the store deletes after the retention period.
"""

from __future__ import annotations

import fcntl
import json
import os
import random
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from . import images as image_ops
from .confirm import SignalState, StateKey, replay, step
from .confirm import confirmed_reading as to_confirmed_reading
from .errors import InputError, StatesureError
from .install import Installation, InstallConfig, render_prompt
from .judges import JudgeError, Transport, build_judge
from .normalize import error_readings
from .readings import Observation, SignalReading, new_sample_id
from .recipe import Recipe
from .recipe import load as load_recipe
from .reconcile import Comparison, needs_verification, reconcile
from .sampling import classify_for_review
from .sources import CaptureError, Fetcher, capture
from .store import LabelStore

MAX_LOG_LINE = 64 * 1024

VERIFY_SUFFIX_MULTI = (
    "\n\nThis is a focused re-check of the same moment. The first image is the full "
    "frame and the others are overlapping crops of it. Do not count the same object "
    "twice across images, and do not guess details that are not visible."
)
VERIFY_SUFFIX_SINGLE = (
    "\n\nThis is a focused re-check: the image is an enlarged crop of the target area "
    "from the same moment. Do not guess details that are not visible."
)


@dataclass
class RunResult:
    installation: str
    status: str
    sample_id: str | None = None
    judge_failed: bool = False
    reasons: tuple[str, ...] = ()
    agreement: str | None = None
    stored: bool = False
    stratum: str | None = None
    signals: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Summary without images, URLs or site wording."""
        return {
            "installation": self.installation,
            "status": self.status,
            "sample_id": self.sample_id,
            "judge_failed": self.judge_failed,
            "reasons": list(self.reasons),
            "agreement": self.agreement,
            "stored": self.stored,
            "stratum": self.stratum,
            "signals": self.signals,
            "events": self.events,
        }


@contextmanager
def run_lock(store_path: Path) -> Iterator[bool]:
    """Exclusive, non-blocking lock so two runs never interleave."""
    store_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = store_path.with_suffix(store_path.suffix + ".lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield True
    finally:
        os.close(fd)


def run_once(
    config: InstallConfig,
    name: str,
    *,
    now: datetime,
    rng: random.Random | None = None,
    fetch: Fetcher = capture,
    transport: Transport | None = None,
) -> RunResult:
    """Run one cycle for installation ``name``. The caller holds ``run_lock``."""
    if now.tzinfo is None:
        raise InputError("naive_timestamp")
    installation = config.installations.get(name)
    if installation is None:
        raise InputError("unknown_installation")
    rng = rng or random.Random()
    recipe = load_recipe(config.recipes_dir / f"{installation.recipe_id}.yaml")
    judge = build_judge(config.judges[installation.judge], transport)
    judge_id = judge.config.judge_id
    install_fp = installation.fingerprint()

    try:
        raw = fetch(installation.source)
        primary_image = image_ops.primary_view(raw)
    except (CaptureError, image_ops.ImageError) as exc:
        # No image means nothing to judge or review; the log is not touched.
        return RunResult(installation=name, status=exc.code)

    sample_id = new_sample_id()
    prompt = render_prompt(recipe, installation)
    judge_failed = False
    quality: Mapping[str, str] = {}
    try:
        result = judge.judge(recipe, prompt, [primary_image])
        primary, quality = dict(result.signals), dict(result.quality)
    except JudgeError as exc:
        # A failed call is recorded, stored for review and scored as a miss.
        primary = error_readings(recipe, exc.code)
        judge_failed = True

    log = ObservationLog(config.observations)
    states = _current_states(log, recipe, installation, install_fp, judge_id)
    confirmed_values = {key.signal: state.confirmed_value for key, state in states.items()}

    verified: dict[str, SignalReading] | None = None
    comparison: Comparison | None = None
    reasons: tuple[str, ...] = ()
    merged = primary
    if not judge_failed:
        reasons = needs_verification(
            recipe,
            quality,
            primary,
            confirmed_values,
            always_verify=installation.always_verify,
        )
        if reasons:
            verified = _verify(judge, recipe, prompt, raw, installation)
            merged, comparison = reconcile(recipe, primary, verified)

    def observation(stage: str, signals: Mapping[str, SignalReading]) -> Observation:
        return Observation(
            sample_id=sample_id,
            source_id=installation.name,
            install_fingerprint=install_fp,
            captured_at=now,
            recipe_id=recipe.id,
            recipe_fingerprint=recipe.fingerprint,
            judge_id=judge_id,
            stage=stage,
            signals=signals,
            quality=quality if stage == "primary" else {},
            error_class=None if not judge_failed or stage != "primary" else "judge_error",
        )

    merged_obs = observation("merged", merged)
    log.append(merged_obs)

    events: list[dict[str, Any]] = []
    confirmed: dict[str, SignalReading] = {}
    for signal in recipe.signal_names:
        key = StateKey(
            installation.name, install_fp, recipe.id, recipe.fingerprint, judge_id, signal
        )
        state, event = step(
            states.get(key, SignalState()),
            merged[signal],
            now,
            recipe.confirmation,
            key=key,
            sample_id=sample_id,
        )
        confirmed[signal] = to_confirmed_reading(state.confirmed_value)
        if event is not None:
            events.append(event.to_dict())

    review = classify_for_review(
        primary,
        primary_failed=judge_failed,
        comparison=comparison,
        reasons=reasons,
        rng=rng,
        save_all=config.evaluation.save_all,
        routine_rate=config.evaluation.routine_rate,
    )
    store = LabelStore(config.store)
    if review.stored:
        store.add_sample(
            sample_id=sample_id,
            source_id=installation.name,
            install_fingerprint=install_fp,
            captured_at=now,
            recipe=recipe,
            review=review,
            save_all=config.evaluation.save_all,
            image=image_ops.stored_view(raw),
            now=now,
            retention=timedelta(days=config.evaluation.retention_days),
        )
        store.add_observation(observation("primary", primary))
        if verified is not None:
            store.add_observation(observation("verified", verified))
        store.add_observation(merged_obs)
        store.add_confirmed(sample_id, judge_id, confirmed)
    store.purge_expired(now)

    return RunResult(
        installation=name,
        status="ok",
        sample_id=sample_id,
        judge_failed=judge_failed,
        reasons=reasons,
        agreement=comparison.agreement if comparison else None,
        stored=review.stored,
        stratum=review.stratum,
        signals={
            signal: {
                "merged": _summary(merged[signal]),
                "confirmed": confirmed[signal].value,
            }
            for signal in recipe.signal_names
        },
        events=events,
    )


def _verify(
    judge: Any,
    recipe: Recipe,
    prompt: str,
    raw: bytes,
    installation: Installation,
) -> dict[str, SignalReading]:
    try:
        views = image_ops.verification_views(raw, installation.roi)
        if judge.config.kind == "jev":
            # One image per request: use the first crop, or the full frame.
            images = [views[1] if len(views) > 1 else views[0]]
            suffix = VERIFY_SUFFIX_SINGLE
        else:
            images = views
            suffix = VERIFY_SUFFIX_MULTI
        return dict(judge.judge(recipe, prompt + suffix, images).signals)
    except (JudgeError, image_ops.ImageError) as exc:
        return error_readings(recipe, exc.code)


def _current_states(
    log: ObservationLog,
    recipe: Recipe,
    installation: Installation,
    install_fp: str,
    judge_id: str,
) -> dict[StateKey, SignalState]:
    relevant = [
        obs
        for obs in log.read()
        if obs.source_id == installation.name
        and obs.install_fingerprint == install_fp
        and obs.recipe_fingerprint == recipe.fingerprint
        and obs.judge_id == judge_id
    ]
    result = replay(relevant, {recipe.fingerprint: recipe.confirmation})
    return dict(result.states)


def _summary(reading: SignalReading) -> dict[str, Any]:
    return {"outcome": reading.outcome.value, "value": reading.value, "reason": reading.reason}


class ObservationLog:
    """Append-only JSONL of merged observations (no pixels)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, observation: Observation) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        line = json.dumps(observation.to_dict(), ensure_ascii=False, allow_nan=False)
        fd = os.open(self.path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def read(self) -> list[Observation]:
        if not self.path.exists():
            return []
        observations: list[Observation] = []
        with self.path.open("rb") as handle:
            for line in handle:
                if len(line) > MAX_LOG_LINE or not line.strip():
                    continue
                try:
                    observations.append(Observation.from_dict(json.loads(line)))
                except (json.JSONDecodeError, UnicodeDecodeError, StatesureError):
                    # A torn or foreign line is skipped, never guessed at.
                    continue
        return observations
