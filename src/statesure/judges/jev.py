"""Judge through a Jev-style decision API.

The request carries a ``state`` (the rendered prompt), one question per signal
and the image as a data URL in ``images``. ``presence`` signals become ``noul``
questions and ``enum`` / ``count`` signals become ``choice`` questions whose
``criteria`` map each allowed value to a short description.

Compatible servers include djev (``/v1/systemone`` or ``/v1/request``) and the
hosted TypeSafe API. The hosted API sends images off-site; statesure prints a
warning for non-local judges.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..install import JudgeConfig
from ..normalize import normalize_jev_answers
from ..recipe import Recipe, SignalSpec
from .base import JudgeError, JudgeResult, Transport, data_url, post_json

# djev accepts one state image per request.
MAX_IMAGES = 1


class JevJudge:
    def __init__(self, config: JudgeConfig, transport: Transport) -> None:
        self.config = config
        self._transport = transport

    def judge(self, recipe: Recipe, prompt: str, images: Sequence[bytes]) -> JudgeResult:
        if len(images) != MAX_IMAGES:
            raise JudgeError("invalid_image_count")
        payload: dict[str, Any] = {
            "state": prompt,
            "questions": {spec.name: _question(spec) for spec in recipe.signals},
            "images": [data_url(images[0])],
        }
        if self.config.model:
            payload["model"] = self.config.model
        body = post_json(self._transport, self.config, payload)
        if self.config.model and body.get("model") not in (None, self.config.model):
            raise JudgeError("model_mismatch")
        answers = body.get("answers")
        if not isinstance(answers, dict):
            raise JudgeError("invalid_response_shape")
        signals = normalize_jev_answers(recipe, answers, min_probability=recipe.min_probability)
        return JudgeResult(signals=signals)


def _question(spec: SignalSpec) -> dict[str, Any]:
    instructions = spec.question.get("en") or f"What is the value of {spec.name}?"
    if spec.type == "presence":
        return {"type": "noul", "instructions": instructions}
    criteria = {str(value): _describe(spec, value) for value in spec.values}
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def _describe(spec: SignalSpec, value: str | int) -> str:
    if spec.type == "count":
        return f"exactly {value}" if value else "none"
    return str(value).replace("_", " ")
