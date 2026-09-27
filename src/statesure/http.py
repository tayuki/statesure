"""A small, strict HTTP client built on the standard library.

No proxies from the environment, no redirects, a size limit on every response,
and error codes that never include request or response bodies or credentials.
"""

from __future__ import annotations

import http.client
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

from .errors import StatesureError


class HttpError(StatesureError):
    """An HTTP request failed. The code is safe to log."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def request(
    method: str,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    body: bytes | None = None,
    timeout_s: float,
    max_bytes: int,
) -> bytes:
    """Send one request and return the response body."""
    req = urllib.request.Request(url, data=body, headers=dict(headers or {}), method=method)
    try:
        with _OPENER.open(req, timeout=timeout_s) as response:
            data = response.read(max_bytes + 1)
    except urllib.error.HTTPError as exc:
        raise HttpError(f"http_{exc.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise HttpError("transport_failed") from None
    except (http.client.HTTPException, ValueError):
        # Malformed URLs or responses must stay on the HttpError path.
        raise HttpError("transport_failed") from None
    if len(data) > max_bytes:
        raise HttpError("response_too_large")
    return data
