"""Shared judge types and a small, strict HTTP transport.

The transport uses only the standard library: no proxies from the environment,
no redirects, a response size limit, and errors that never include the request
body, the response body or credentials.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

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


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def http_post(url: str, headers: Mapping[str, str], body: bytes, timeout_s: float) -> bytes:
    """POST ``body`` and return the response body (standard library only)."""
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with _OPENER.open(request, timeout=timeout_s) as response:
            data = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise JudgeError(f"http_{exc.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise JudgeError("transport_failed") from None
    if len(data) > MAX_RESPONSE_BYTES:
        raise JudgeError("response_too_large")
    return data


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
    import base64

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
