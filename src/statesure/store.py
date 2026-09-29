"""Local label store (SQLite).

Samples, judge readings for every stage, and human labels live in one SQLite
file. Image bytes, when evaluation is on, are kept next to it with an expiry and
removed by ``purge_expired`` regardless of how the store is accessed.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .errors import StoreError
from .readings import Observation, Outcome, SignalReading
from .recipe import Recipe
from .sampling import PRIORITIES, STRATA, ReviewClass

SCHEMA_VERSION = 1
HUMAN_UNCERTAIN = "human_uncertain"
EVAL_STAGES = ("primary", "verified", "merged", "confirmed")
REVIEWER_CONFIDENCE = ("confident", "unsure")
_SAMPLE_ID = re.compile(r"^[0-9a-f]{32}$")

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS samples (
  sample_id TEXT PRIMARY KEY,
  source_id TEXT NOT NULL,
  install_fingerprint TEXT NOT NULL,
  captured_at TEXT NOT NULL,
  captured_at_utc TEXT NOT NULL,
  recipe_id TEXT NOT NULL,
  recipe_fingerprint TEXT NOT NULL,
  stratum TEXT NOT NULL CHECK (stratum IN {STRATA}),
  priority TEXT NOT NULL CHECK (priority IN {PRIORITIES}),
  inclusion_probability REAL NOT NULL
    CHECK (inclusion_probability > 0 AND inclusion_probability <= 1),
  save_all INTEGER NOT NULL,
  display_key REAL NOT NULL,
  image_path TEXT,
  image_retention_until TEXT,
  image_deleted_at TEXT
);
CREATE TABLE IF NOT EXISTS readings (
  sample_id TEXT NOT NULL REFERENCES samples(sample_id),
  judge_id TEXT NOT NULL,
  stage TEXT NOT NULL CHECK (stage IN {EVAL_STAGES}),
  signal TEXT NOT NULL,
  outcome TEXT NOT NULL,
  value_json TEXT,
  probability REAL,
  distribution_json TEXT,
  reason TEXT,
  PRIMARY KEY (sample_id, judge_id, stage, signal)
);
CREATE TABLE IF NOT EXISTS labels (
  sample_id TEXT NOT NULL REFERENCES samples(sample_id),
  signal TEXT NOT NULL,
  truth_json TEXT,
  assessable INTEGER NOT NULL,
  reviewer_confidence TEXT NOT NULL,
  labeled_at TEXT NOT NULL,
  label_source TEXT NOT NULL CHECK (label_source = 'human'),
  PRIMARY KEY (sample_id, signal)
);
"""


@dataclass(frozen=True)
class Label:
    truth: str | int | None
    """The true value, ``HUMAN_UNCERTAIN``, or None when not assessable."""
    assessable: bool
    reviewer_confidence: str
    labeled_at: datetime


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    source_id: str
    install_fingerprint: str
    captured_at: datetime
    recipe_id: str
    recipe_fingerprint: str
    stratum: str
    priority: str
    inclusion_probability: float
    display_key: float
    readings: Mapping[tuple[str, str, str], SignalReading]
    """(judge_id, stage, signal) -> reading"""
    labels: Mapping[str, Label]


class LabelStore:
    """SQLite store. Files are created with owner-only permissions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.frames_dir = self.path.parent / f"{self.path.stem}-frames"
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        existed = self.path.exists()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            elif row[0] != str(SCHEMA_VERSION):
                raise StoreError("unsupported_store_version")
        if not existed:
            os.chmod(self.path, 0o600)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            with conn:
                yield conn
        finally:
            conn.close()

    def add_sample(
        self,
        *,
        sample_id: str,
        source_id: str,
        install_fingerprint: str,
        captured_at: datetime,
        recipe: Recipe,
        review: ReviewClass,
        save_all: bool,
        image: bytes | None = None,
        now: datetime | None = None,
        retention: timedelta = timedelta(days=7),
    ) -> None:
        """Register a stored sample. Unselected samples must not be added."""
        if not _SAMPLE_ID.fullmatch(sample_id):
            raise StoreError("invalid_sample_id")
        if not review.stored:
            raise StoreError("sample_not_selected")
        if captured_at.tzinfo is None or (now is not None and now.tzinfo is None):
            raise StoreError("naive_timestamp")
        with self._connect() as conn:
            if conn.execute("SELECT 1 FROM samples WHERE sample_id=?", (sample_id,)).fetchone():
                # Checked before writing bytes so an existing frame is never replaced.
                raise StoreError("duplicate_or_invalid_sample")
        image_path = retention_until = None
        if image is not None:
            if now is None:
                raise StoreError("image_requires_now")
            image_path = self._write_frame(sample_id, image)
            retention_until = (now + retention).isoformat()
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL)",
                    (
                        sample_id,
                        source_id,
                        install_fingerprint,
                        captured_at.isoformat(),
                        _utc(captured_at),
                        recipe.id,
                        recipe.fingerprint,
                        review.stratum,
                        review.priority,
                        review.inclusion_probability,
                        int(save_all),
                        review.display_key,
                        image_path,
                        retention_until,
                    ),
                )
        except sqlite3.IntegrityError:
            if image_path is not None:
                (self.frames_dir / image_path).unlink(missing_ok=True)
            raise StoreError("duplicate_or_invalid_sample") from None

    def add_observation(self, observation: Observation) -> None:
        """Store the readings of a primary, verified or merged observation."""
        sample = self._sample_row(observation.sample_id)
        if (
            sample["install_fingerprint"] != observation.install_fingerprint
            or sample["recipe_id"] != observation.recipe_id
            or sample["recipe_fingerprint"] != observation.recipe_fingerprint
        ):
            raise StoreError("observation_sample_mismatch")
        self._put_readings(
            observation.sample_id, observation.judge_id, observation.stage, observation.signals
        )

    def add_confirmed(
        self, sample_id: str, judge_id: str, readings: Mapping[str, SignalReading]
    ) -> None:
        """Store the confirmed-state readings for a sample (see confirm.replay)."""
        self._sample_row(sample_id)
        self._put_readings(sample_id, judge_id, "confirmed", readings)

    def add_label(
        self,
        recipe: Recipe,
        sample_id: str,
        signal: str,
        truth: str | int | None,
        *,
        assessable: bool,
        reviewer_confidence: str,
        labeled_at: datetime,
        label_source: str = "human",
    ) -> None:
        """Record one human label. Model predictions are never accepted as labels."""
        self.add_labels(
            recipe,
            sample_id,
            {signal: truth},
            assessable=assessable,
            reviewer_confidence=reviewer_confidence,
            labeled_at=labeled_at,
            label_source=label_source,
        )

    def add_labels(
        self,
        recipe: Recipe,
        sample_id: str,
        truths: Mapping[str, str | int | None],
        *,
        assessable: bool,
        reviewer_confidence: str,
        labeled_at: datetime,
        label_source: str = "human",
    ) -> None:
        """Record several labels for one sample: all are validated first, then
        written in a single transaction, so a sample is never half-labeled."""
        if label_source != "human":
            raise StoreError("only_human_labels")
        sample = self._sample_row(sample_id)
        if sample["recipe_id"] != recipe.id or sample["recipe_fingerprint"] != recipe.fingerprint:
            # Labels are validated against the schema of the version that produced the sample.
            raise StoreError("label_recipe_mismatch")
        if reviewer_confidence not in REVIEWER_CONFIDENCE:
            raise StoreError("invalid_reviewer_confidence")
        if labeled_at.tzinfo is None:
            raise StoreError("naive_timestamp")
        if not truths:
            raise StoreError("no_labels")
        for signal, truth in truths.items():
            if signal not in recipe.signal_names:
                raise StoreError("unknown_signal")
            if assessable:
                if truth != HUMAN_UNCERTAIN and not recipe.signal(signal).accepts(truth):
                    raise StoreError("invalid_truth")
            elif truth is not None:
                raise StoreError("truth_without_assessable")
        if assessable:
            existing = {
                signal: label.truth
                for signal, label in self._labels_for(sample_id).items()
                if label.assessable
            }
            if recipe.conflicts({**existing, **truths}, ignore=frozenset({HUMAN_UNCERTAIN})):
                # e.g. "present" with location "none": ask the person to check again.
                raise StoreError("inconsistent_labels")
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?,?,?)",
                [
                    (
                        sample_id,
                        signal,
                        json.dumps(truth),
                        int(assessable),
                        reviewer_confidence,
                        labeled_at.isoformat(),
                        "human",
                    )
                    for signal, truth in truths.items()
                ],
            )

    def _labels_for(self, sample_id: str) -> dict[str, Label]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT signal, truth_json, assessable, reviewer_confidence, labeled_at "
                "FROM labels WHERE sample_id=?",
                (sample_id,),
            ).fetchall()
        return {
            signal: Label(
                truth=json.loads(truth),
                assessable=bool(assessable),
                reviewer_confidence=confidence,
                labeled_at=datetime.fromisoformat(labeled_at),
            )
            for signal, truth, assessable, confidence, labeled_at in rows
        }

    def purge_expired(self, now: datetime) -> int:
        """Delete image bytes past their retention time. Idempotent."""
        if now.tzinfo is None:
            raise StoreError("naive_timestamp")
        purged = 0
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT sample_id, image_path, image_retention_until FROM samples "
                "WHERE image_path IS NOT NULL AND image_deleted_at IS NULL"
            ).fetchall()
            for sample_id, image_path, until in rows:
                if datetime.fromisoformat(until) > now:
                    continue
                (self.frames_dir / image_path).unlink(missing_ok=True)
                conn.execute(
                    "UPDATE samples SET image_deleted_at=? WHERE sample_id=?",
                    (now.isoformat(), sample_id),
                )
                purged += 1
        return purged

    def image_path(self, sample_id: str, now: datetime | None = None) -> Path | None:
        """Path of a sample's image, or None if absent, deleted or (given ``now``) expired."""
        row = self._sample_row(sample_id)
        if row["image_path"] is None or row["image_deleted_at"] is not None:
            return None
        if now is not None and datetime.fromisoformat(row["image_retention_until"]) <= now:
            return None
        return self.frames_dir / row["image_path"]

    def records(self, recipe_id: str | None = None) -> list[SampleRecord]:
        """Load samples with all readings and labels, ordered by capture time."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            where, params = ("WHERE recipe_id=?", (recipe_id,)) if recipe_id else ("", ())
            samples = conn.execute(
                f"SELECT * FROM samples {where} ORDER BY captured_at_utc, sample_id", params
            ).fetchall()
            readings: dict[str, dict[tuple[str, str, str], SignalReading]] = {}
            for row in conn.execute("SELECT * FROM readings"):
                readings.setdefault(row["sample_id"], {})[
                    (row["judge_id"], row["stage"], row["signal"])
                ] = SignalReading(
                    Outcome(row["outcome"]),
                    json.loads(row["value_json"]) if row["value_json"] else None,
                    row["probability"],
                    json.loads(row["distribution_json"]) if row["distribution_json"] else None,
                    row["reason"],
                )
            labels: dict[str, dict[str, Label]] = {}
            for row in conn.execute("SELECT * FROM labels"):
                labels.setdefault(row["sample_id"], {})[row["signal"]] = Label(
                    truth=json.loads(row["truth_json"]),
                    assessable=bool(row["assessable"]),
                    reviewer_confidence=row["reviewer_confidence"],
                    labeled_at=datetime.fromisoformat(row["labeled_at"]),
                )
        return [
            SampleRecord(
                sample_id=row["sample_id"],
                source_id=row["source_id"],
                install_fingerprint=row["install_fingerprint"],
                captured_at=datetime.fromisoformat(row["captured_at"]),
                recipe_id=row["recipe_id"],
                recipe_fingerprint=row["recipe_fingerprint"],
                stratum=row["stratum"],
                priority=row["priority"],
                inclusion_probability=row["inclusion_probability"],
                display_key=row["display_key"],
                readings=readings.get(row["sample_id"], {}),
                labels=labels.get(row["sample_id"], {}),
            )
            for row in samples
        ]

    def _put_readings(
        self, sample_id: str, judge_id: str, stage: str, signals: Mapping[str, SignalReading]
    ) -> None:
        if stage not in EVAL_STAGES:
            raise StoreError("invalid_stage")
        with self._connect() as conn:
            for signal, reading in signals.items():
                conn.execute(
                    "INSERT OR REPLACE INTO readings VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        sample_id,
                        judge_id,
                        stage,
                        signal,
                        reading.outcome.value,
                        json.dumps(reading.value) if reading.value is not None else None,
                        reading.probability,
                        json.dumps(dict(reading.distribution)) if reading.distribution else None,
                        reading.reason,
                    ),
                )

    def _sample_row(self, sample_id: str) -> sqlite3.Row:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM samples WHERE sample_id=?", (sample_id,)).fetchone()
        if row is None:
            raise StoreError("unknown_sample")
        return row

    def _write_frame(self, sample_id: str, image: bytes) -> str:
        self.frames_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        name = f"{sample_id}.bin"
        with tempfile.NamedTemporaryFile("wb", dir=self.frames_dir, delete=False) as handle:
            handle.write(image)
            temporary = Path(handle.name)
        os.chmod(temporary, 0o600)
        temporary.replace(self.frames_dir / name)
        return name


def _utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()
