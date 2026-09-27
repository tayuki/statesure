"""Shared fixtures. All data is synthetic; no images are used."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml

from statesure import recipe as recipe_module
from statesure.readings import Observation, SignalReading, new_sample_id

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "recipes"
TZ = timezone(timedelta(hours=9))
T0 = datetime(2026, 1, 5, 7, 0, tzinfo=TZ)


@pytest.fixture
def package_doc() -> dict[str, Any]:
    return yaml.safe_load((RECIPES / "package_at_door.yaml").read_text("utf-8"))


@pytest.fixture
def package_recipe() -> recipe_module.Recipe:
    return recipe_module.load(RECIPES / "package_at_door.yaml")


def doc_copy(doc: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(doc)


def presence_obs(
    recipe: recipe_module.Recipe,
    value: str | None,
    at: datetime,
    *,
    stage: str = "primary",
    judge_id: str = "vlm_openai:test-model",
    install: str = "install-a",
    source: str = "cam-a",
    sample_id: str | None = None,
    reading: SignalReading | None = None,
) -> Observation:
    """An observation where only the presence signal matters."""
    signal = recipe.signals[0].name
    if reading is None:
        reading = (
            SignalReading.decided(value)
            if value is not None
            else SignalReading.abstained("model_uncertain")
        )
    return Observation(
        sample_id=sample_id or new_sample_id(),
        source_id=source,
        install_fingerprint=install,
        captured_at=at,
        recipe_id=recipe.id,
        recipe_fingerprint=recipe.fingerprint,
        judge_id=judge_id,
        stage=stage,
        signals={signal: reading},
    )
