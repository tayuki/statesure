"""Evaluation cards: shareable results without images.

A card holds counts and intervals for one recipe version, one judge and one
installation. It never contains sample ids, source ids, installation details,
timestamps, images or hashes of images. ``validate_card`` enforces an exact
allowlist and is called before any card is returned.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .errors import CardError
from .metrics import Calibration, Ratio, Report, SignalMetrics
from .recipe import Recipe

CARD_VERSION = 1
CAMERA_KINDS = ("outdoor", "indoor", "unspecified")
JUDGE_KINDS = ("vlm_openai", "jev", "external")
STAGES = ("primary", "verified", "merged", "confirmed")

_TOP_KEYS = {
    "card_version",
    "recipe_id",
    "recipe_version",
    "recipe_fingerprint",
    "judge_kind",
    "model",
    "camera_kind",
    "evaluated_days",
    "stages",
    "calibration",
}
_SIGNAL_KEYS = {
    "scored",
    "correct",
    "wrong",
    "abstained",
    "failed",
    "coverage",
    "tp",
    "fp",
    "fn",
    "tn",
    "recall_all",
    "precision",
}
_CALIBRATION_KEYS = {"kind", "n", "ece", "brier", "bins"}
_BIN_KEYS = {"low", "high", "n", "hits"}
_SAFE_STRING = re.compile(r"^[a-z0-9_.:-]{1,64}$")
_DATE_LIKE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}")
_IP_LIKE = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
_HEX_RUN = re.compile(r"[0-9a-f]{12,}")


def normalize_model_name(model: str) -> str:
    """Lowercase a model name and replace ``/`` so it fits the card alphabet."""
    return model.strip().lower().replace("/", ":")


def build_card(
    recipe: Recipe,
    report: Report,
    *,
    install_fingerprint: str,
    judge_id: str,
    judge_kind: str,
    model: str,
    evaluated_days: int,
    camera_kind: str = "unspecified",
) -> dict[str, Any]:
    """Build and validate a card for one installation and judge."""
    stages: dict[str, dict[str, Any]] = {}
    calibration: dict[str, Any] = {}
    for metrics in report.signals:
        key = metrics.key
        if key.install_fingerprint != install_fingerprint or key.judge_id != judge_id:
            continue
        if key.recipe_fingerprint != recipe.fingerprint:
            continue
        stages.setdefault(key.stage, {})[key.signal] = _signal_entry(metrics)
        if key.stage == "primary" and metrics.calibration is not None:
            calibration[key.signal] = _calibration_entry(metrics.calibration)
    card = {
        "card_version": CARD_VERSION,
        "recipe_id": recipe.id,
        "recipe_version": recipe.version,
        "recipe_fingerprint": recipe.fingerprint,
        "judge_kind": judge_kind,
        "model": normalize_model_name(model),
        "camera_kind": camera_kind,
        "evaluated_days": evaluated_days,
        "stages": stages,
        "calibration": calibration,
    }
    validate_card(card, recipe)
    return card


def validate_card(card: object, recipe: Recipe) -> None:
    """Raise CardError unless ``card`` matches the allowlist exactly."""
    if not isinstance(card, Mapping) or set(card) != _TOP_KEYS:
        raise CardError("card_keys")
    if card["card_version"] != CARD_VERSION:
        raise CardError("card_version")
    if card["recipe_id"] != recipe.id or card["recipe_version"] != recipe.version:
        raise CardError("card_recipe")
    if card["recipe_fingerprint"] != recipe.fingerprint:
        raise CardError("card_recipe")
    if card["judge_kind"] not in JUDGE_KINDS:
        raise CardError("card_judge_kind")
    if card["camera_kind"] not in CAMERA_KINDS:
        raise CardError("card_camera_kind")
    _safe_string(card["model"])
    days = card["evaluated_days"]
    if type(days) is not int or not 0 <= days <= 3660:
        raise CardError("card_days")
    stages = card["stages"]
    if not isinstance(stages, Mapping) or not set(stages) <= set(STAGES):
        raise CardError("card_stages")
    signal_names = set(recipe.signal_names)
    for signals in stages.values():
        if not isinstance(signals, Mapping) or not set(signals) <= signal_names:
            raise CardError("card_signals")
        for entry in signals.values():
            _check_signal_entry(entry)
    calibration = card["calibration"]
    if not isinstance(calibration, Mapping) or not set(calibration) <= signal_names:
        raise CardError("card_calibration")
    for entry in calibration.values():
        _check_calibration_entry(entry)


def _signal_entry(metrics: SignalMetrics) -> dict[str, Any]:
    presence = metrics.presence
    return {
        "scored": metrics.scored,
        **{kind: metrics.counts[kind] for kind in ("correct", "wrong", "abstained", "failed")},
        "coverage": _interval(metrics.coverage),
        "tp": presence.tp if presence else None,
        "fp": presence.fp if presence else None,
        "fn": presence.fn if presence else None,
        "tn": presence.tn if presence else None,
        "recall_all": _interval(presence.recall_all) if presence else None,
        "precision": _interval(presence.precision) if presence else None,
    }


def _calibration_entry(calibration: Calibration) -> dict[str, Any]:
    return {
        "kind": calibration.kind,
        "n": calibration.n,
        "ece": round(calibration.ece, 6) if calibration.ece is not None else None,
        "brier": round(calibration.brier, 6) if calibration.brier is not None else None,
        "bins": [
            {"low": low, "high": high, "n": count, "hits": hits}
            for low, high, count, hits, _ in calibration.bins
        ],
    }


def _interval(value: Ratio) -> list[float] | None:
    if value.interval is None:
        return None
    return [round(value.interval[0], 6), round(value.interval[1], 6)]


def _check_signal_entry(entry: object) -> None:
    if not isinstance(entry, Mapping) or set(entry) != _SIGNAL_KEYS:
        raise CardError("card_signal_keys")
    for name in ("scored", "correct", "wrong", "abstained", "failed"):
        _count(entry[name])
    for name in ("tp", "fp", "fn", "tn"):
        if entry[name] is not None:
            _count(entry[name])
    for name in ("coverage", "recall_all", "precision"):
        _check_interval(entry[name])


def _check_calibration_entry(entry: object) -> None:
    if not isinstance(entry, Mapping) or set(entry) != _CALIBRATION_KEYS:
        raise CardError("card_calibration_keys")
    if entry["kind"] not in ("positive", "top1"):
        raise CardError("card_calibration_kind")
    _count(entry["n"])
    for name in ("ece", "brier"):
        value = entry[name]
        if value is not None and (type(value) not in (int, float) or not 0 <= value <= 2):
            raise CardError("card_number")
    bins = entry["bins"]
    if not isinstance(bins, list) or len(bins) > 20:
        raise CardError("card_bins")
    for item in bins:
        if not isinstance(item, Mapping) or set(item) != _BIN_KEYS:
            raise CardError("card_bin_keys")
        _count(item["n"])
        _count(item["hits"])
        for name in ("low", "high"):
            if type(item[name]) not in (int, float) or not 0 <= item[name] <= 1:
                raise CardError("card_number")


def _check_interval(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, list) or len(value) != 2:
        raise CardError("card_interval")
    for bound in value:
        if type(bound) not in (int, float) or not 0 <= bound <= 1:
            raise CardError("card_interval")


def _count(value: object) -> None:
    if type(value) is not int or value < 0:
        raise CardError("card_count")


def _safe_string(value: object) -> None:
    if not isinstance(value, str) or not _SAFE_STRING.fullmatch(value):
        raise CardError("card_string")
    if _DATE_LIKE.search(value) or _IP_LIKE.search(value) or _HEX_RUN.search(value):
        raise CardError("card_string")
    if value.startswith(("http", "www")):
        raise CardError("card_string")
