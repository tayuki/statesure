"""Fetch one still image from a camera source.

Supported sources: Frigate's latest frame, a plain image URL, and an RTSP
stream (one frame through ffmpeg). RTSP credentials are read from environment
variables at run time and never logged.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from urllib.parse import quote, urlsplit, urlunsplit

from . import http
from .errors import StatesureError
from .install import SourceConfig

MAX_IMAGE_BYTES = 16 * 1024 * 1024
HTTP_TIMEOUT_S = 20.0
RTSP_TIMEOUT_S = 30.0

Fetcher = Callable[[SourceConfig], bytes]


class CaptureError(StatesureError):
    """No image could be captured. The code is safe to log."""


def capture(source: SourceConfig) -> bytes:
    """Return the raw bytes of one frame."""
    if source.kind == "frigate":
        return _http_get(f"{source.url}/api/{source.camera}/latest.jpg")
    if source.kind == "http_image":
        return _http_get(source.url)
    if source.kind == "rtsp":
        return _rtsp_frame(source)
    raise CaptureError("unknown_source_kind")


def _http_get(url: str) -> bytes:
    try:
        data = http.request(
            "GET",
            url,
            headers={"Accept": "image/jpeg,image/*"},
            timeout_s=HTTP_TIMEOUT_S,
            max_bytes=MAX_IMAGE_BYTES,
        )
    except http.HttpError as exc:
        raise CaptureError(exc.code) from None
    if not data:
        raise CaptureError("empty_image")
    return data


def rtsp_url_with_credentials(source: SourceConfig) -> str:
    """Insert credentials from the environment into the RTSP URL."""
    if not source.username_env:
        return source.url
    username = os.environ.get(source.username_env, "")
    password = os.environ.get(source.password_env or "", "")
    if not username:
        raise CaptureError("missing_rtsp_credentials")
    parts = urlsplit(source.url)
    userinfo = quote(username, safe="")
    if password:
        userinfo += ":" + quote(password, safe="")
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"  # IPv6 literals keep their brackets
    netloc = f"{userinfo}@{host}" + (f":{parts.port}" if parts.port else "")
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _rtsp_frame(source: SourceConfig) -> bytes:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise CaptureError("ffmpeg_not_found")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-i",
        rtsp_url_with_credentials(source),
        "-frames:v",
        "1",
        "-f",
        "image2",
        "-c:v",
        "mjpeg",
        "pipe:1",
    ]
    try:
        # stderr is discarded: ffmpeg may echo the URL, which can contain credentials.
        result = subprocess.run(  # noqa: S603 - fixed argument list, no shell
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=RTSP_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise CaptureError("rtsp_timeout") from None
    except OSError:
        raise CaptureError("rtsp_failed") from None
    if result.returncode != 0 or not result.stdout:
        raise CaptureError("rtsp_failed")
    if len(result.stdout) > MAX_IMAGE_BYTES:
        raise CaptureError("image_too_large")
    return result.stdout
