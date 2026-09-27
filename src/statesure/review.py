"""Local labeling UI.

Binds to loopback only (use an SSH tunnel from another device). The page never
shows the judge's answer, so labels are not anchored to it. Every form carries
a per-process token, Host and Origin headers are checked, and responses carry a
strict Content-Security-Policy. Images past their retention time are never
served, even before the purge deletes them.
"""

from __future__ import annotations

import html
import ipaddress
import re
import secrets
import threading
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .errors import StatesureError
from .recipe import Recipe
from .recipe import load as load_recipe
from .store import HUMAN_UNCERTAIN, PRIORITIES, LabelStore, SampleRecord

MAX_FORM_BYTES = 16 * 1024
PAGE_SIZE = 20
PURGE_INTERVAL_S = 3600
_SAMPLE_ID_PATH = r"[0-9a-f]{32}"

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; "
        "form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "X-Frame-Options": "DENY",
}


class ReviewApp:
    """Label store plus the recipes needed to render and validate forms."""

    def __init__(
        self,
        store: LabelStore,
        recipes_dir: Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.store = store
        self.recipes_dir = recipes_dir
        self.token = secrets.token_urlsafe(32)
        self._clock = clock
        self._recipes: dict[str, tuple[tuple[int, int], Recipe]] = {}

    def now(self) -> datetime:
        return self._clock() if self._clock else datetime.now(UTC)

    def recipe_for(self, record: SampleRecord) -> Recipe | None:
        """The current recipe file, only if it is the version that produced the sample.

        The file is re-read whenever it changes, so edits made while the page is
        running are honored; failures are not cached.
        """
        path = self.recipes_dir / f"{record.recipe_id}.yaml"
        try:
            stat = path.stat()
        except OSError:
            self._recipes.pop(record.recipe_id, None)
            return None
        version = (stat.st_mtime_ns, stat.st_size)
        cached = self._recipes.get(record.recipe_id)
        if cached is None or cached[0] != version:
            try:
                cached = (version, load_recipe(path))
            except StatesureError:
                self._recipes.pop(record.recipe_id, None)
                return None
            self._recipes[record.recipe_id] = cached
        recipe = cached[1]
        if recipe.fingerprint != record.recipe_fingerprint:
            return None
        return recipe

    def queue(self) -> list[tuple[SampleRecord, Recipe]]:
        """Unlabeled samples with a viewable image, in review priority order."""
        order = {name: index for index, name in enumerate(PRIORITIES)}
        now = self.now()
        items: list[tuple[SampleRecord, Recipe]] = []
        for record in self.store.records():
            if record.labels:
                continue
            recipe = self.recipe_for(record)
            if recipe is None or self.store.image_path(record.sample_id, now) is None:
                continue
            items.append((record, recipe))
        items.sort(key=lambda item: (order.get(item[0].priority, 9), item[0].display_key))
        return items

    def submit(self, sample_id: str, form: Mapping[str, list[str]]) -> None:
        record = next((r for r in self.store.records() if r.sample_id == sample_id), None)
        if record is None:
            raise StatesureError("unknown_sample")
        recipe = self.recipe_for(record)
        if recipe is None:
            raise StatesureError("recipe_version_unavailable")
        confidence = _one(form, "reviewer_confidence")
        assessable = _one(form, "assessable") != "no"
        truths: dict[str, str | int | None] = {}
        for spec in recipe.signals:
            if not assessable:
                truths[spec.name] = None
                continue
            raw = _one(form, f"signal:{spec.name}")
            if raw == HUMAN_UNCERTAIN:
                truths[spec.name] = HUMAN_UNCERTAIN
            elif spec.type == "count" and raw.isdigit():
                truths[spec.name] = int(raw)
            else:
                truths[spec.name] = raw
        self.store.add_labels(
            recipe,
            sample_id,
            truths,
            assessable=assessable,
            reviewer_confidence=confidence,
            labeled_at=self.now(),
        )


def serve(app: ReviewApp, host: str = "127.0.0.1", port: int = 18120) -> ThreadingHTTPServer:
    """Create the server (not started). Only loopback addresses are accepted."""
    if not ipaddress.ip_address(host).is_loopback:
        raise StatesureError("review_must_bind_loopback")
    server = ThreadingHTTPServer((host, port), _handler_for(app))
    _start_purger(app, server)
    return server


def _start_purger(app: ReviewApp, server: ThreadingHTTPServer) -> None:
    stop = threading.Event()

    def loop() -> None:
        while not stop.is_set():
            app.store.purge_expired(app.now())
            stop.wait(PURGE_INTERVAL_S)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    original = server.server_close

    def close() -> None:
        stop.set()
        original()

    server.server_close = close  # type: ignore[method-assign]


def _handler_for(app: ReviewApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "statesure-review"
        sys_version = ""

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            # Request lines contain sample ids only; nothing is logged.
            return

        def _allowed_host(self) -> bool:
            host = (self.headers.get("Host") or "").lower()
            port = self.server.server_address[1]
            return host in {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for name, value in SECURITY_HEADERS.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _page(self, status: HTTPStatus, body: str) -> None:
            self._send(status, _layout(body).encode("utf-8"), "text/html; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            if not self._allowed_host():
                return self._page(HTTPStatus.FORBIDDEN, "<p>Forbidden host.</p>")
            path = urlsplit(self.path).path
            if path == "/":
                return self._page(HTTPStatus.OK, _queue_page(app))
            if path.startswith("/image/"):
                sample_id = path.removeprefix("/image/")
                if not re.fullmatch(_SAMPLE_ID_PATH, sample_id):
                    return self._page(HTTPStatus.NOT_FOUND, "<p>Not found.</p>")
                try:
                    image = app.store.image_path(sample_id, app.now())
                except StatesureError:
                    image = None
                if image is None or not image.exists():
                    return self._page(HTTPStatus.NOT_FOUND, "<p>Image expired or missing.</p>")
                return self._send(HTTPStatus.OK, image.read_bytes(), "image/jpeg")
            return self._page(HTTPStatus.NOT_FOUND, "<p>Not found.</p>")

        def do_POST(self) -> None:  # noqa: N802
            if not self._allowed_host():
                return self._page(HTTPStatus.FORBIDDEN, "<p>Forbidden host.</p>")
            origin = self.headers.get("Origin")
            if origin is not None and origin != f"http://{self.headers.get('Host')}":
                return self._page(HTTPStatus.FORBIDDEN, "<p>Forbidden origin.</p>")
            path = urlsplit(self.path).path
            match = re.fullmatch(rf"/label/({_SAMPLE_ID_PATH})", path)
            length = int(self.headers.get("Content-Length") or 0)
            if match is None or not 0 < length <= MAX_FORM_BYTES:
                return self._page(HTTPStatus.BAD_REQUEST, "<p>Bad request.</p>")
            form = parse_qs(self.rfile.read(length).decode("utf-8", "replace"))
            if not secrets.compare_digest(_one(form, "token"), app.token):
                return self._page(HTTPStatus.FORBIDDEN, "<p>Expired form. Reload.</p>")
            try:
                app.submit(match.group(1), form)
            except StatesureError as exc:
                return self._page(
                    HTTPStatus.BAD_REQUEST, f"<p>Rejected: {html.escape(exc.code)}</p>"
                )
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", "/")
            for name, value in SECURITY_HEADERS.items():
                self.send_header(name, value)
            self.send_header("Content-Length", "0")
            self.end_headers()

    return Handler


def _one(form: Mapping[str, list[str]], key: str) -> str:
    values = form.get(key) or [""]
    return values[0]


def _queue_page(app: ReviewApp) -> str:
    items = app.queue()
    if not items:
        return "<h1>Nothing to label</h1><p>All stored samples with images are labeled.</p>"
    parts = [f"<h1>Label samples</h1><p>{len(items)} waiting. Showing up to {PAGE_SIZE}.</p>"]
    for record, recipe in items[:PAGE_SIZE]:
        parts.append(_sample_form(app, record, recipe))
    return "".join(parts)


def _sample_form(app: ReviewApp, record: SampleRecord, recipe: Recipe) -> str:
    sid = html.escape(record.sample_id)
    title = html.escape(recipe.title.get("en", recipe.id))
    rows = []
    for spec in recipe.signals:
        options = [f'<option value="{HUMAN_UNCERTAIN}">cannot tell</option>'] + [
            f'<option value="{html.escape(str(v))}">{html.escape(str(v))}</option>'
            for v in spec.values
        ]
        question = html.escape(spec.question.get("en", spec.name))
        rows.append(
            f'<label>{question}<br><select name="signal:{html.escape(spec.name)}" required>'
            f'<option value="" selected disabled>choose</option>{"".join(options)}'
            "</select></label>"
        )
    return (
        f'<section><h2>{title}</h2><p class="meta">{html.escape(record.priority)}</p>'
        f'<img src="/image/{sid}" alt="stored sample">'
        f'<form method="post" action="/label/{sid}">'
        f'<input type="hidden" name="token" value="{html.escape(app.token)}">'
        f"{''.join(rows)}"
        '<label>Your confidence <select name="reviewer_confidence">'
        '<option value="confident">confident</option><option value="unsure">unsure</option>'
        "</select></label>"
        '<button type="submit">Save</button> '
        '<button type="submit" name="assessable" value="no" formnovalidate>'
        "Cannot judge this image</button>"
        "</form></section>"
    )


def _layout(body: str) -> str:
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>statesure review</title><style>"
        "body{font-family:system-ui,sans-serif;max-width:960px;margin:0 auto;padding:16px}"
        "section{border-top:1px solid #8884;padding:16px 0}"
        "img{max-width:100%;height:auto;display:block;margin:8px 0}"
        "label{display:block;margin:8px 0}.meta{color:#666;font-size:.9em}"
        "button{margin-top:8px;padding:6px 12px}"
        "</style></head><body>" + body + "</body></html>"
    )
