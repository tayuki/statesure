"""Shared judge types and the default transport (see ``statesure.http``)."""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .. import http
from ..errors import StatesureError
from ..install import JudgeConfig
from ..readings import SignalReading
from ..recipe import Recipe

MAX_RESPONSE_BYTES = 256 * 1024

Transport = Callable[[str, Mapping[str, str], bytes, float], bytes]
"""(url, headers, body, timeout_s) -> response body"""


class JudgeError(StatesureError):
    """A judge call failed. The code is safe to log."""


@dataclass(frozen=True)
class JudgeResult:
    signals: Mapping[str, SignalReading]
    quality: Mapping[str, str] = field(default_factory=dict)


class Judge(Protocol):
    config: JudgeConfig

    def judge(self, recipe: Recipe, prompt: str, images: Sequence[bytes]) -> JudgeResult:
        """Judge JPEG ``images`` with ``prompt``. Raises JudgeError on failure."""
        ...


def http_post(url: str, headers: Mapping[str, str], body: bytes, timeout_s: float) -> bytes:
    """POST ``body`` and return the response body."""
    try:
        return http.request(
            "POST",
            url,
            headers=headers,
            body=body,
            timeout_s=timeout_s,
            max_bytes=MAX_RESPONSE_BYTES,
        )
    except http.HttpError as exc:
        raise JudgeError(exc.code) from None


def post_json(
    transport: Transport, config: JudgeConfig, payload: Mapping[str, Any]
) -> Mapping[str, Any]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if config.api_key_env:
        key = os.environ.get(config.api_key_env, "")
        if not key or any(ch.isspace() for ch in key):
            raise JudgeError("missing_api_key")
        headers["Authorization"] = f"Bearer {key}"
    body = json.dumps(payload, allow_nan=False).encode("utf-8")
    raw = transport(config.url, headers, body, config.timeout_s)
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise JudgeError("invalid_response_json") from None
    if not isinstance(value, dict):
        raise JudgeError("invalid_response_shape")
    return value


def data_url(image: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")


def build_judge(config: JudgeConfig, transport: Transport | None = None) -> Judge:
    """Create the adapter for ``config.kind``."""
    from .jev import JevJudge
    from .vlm_openai import OpenAIVisionJudge

    if config.kind == "vlm_openai":
        return OpenAIVisionJudge(config, transport or http_post)
    if config.kind == "jev":
        return JevJudge(config, transport or http_post)
    raise JudgeError("unknown_judge_kind")
