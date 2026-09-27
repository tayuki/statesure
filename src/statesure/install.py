"""Installation config: where images come from, which judge answers, and the
site-specific wording that recipes leave out.

The config file stays on the user's machine. Nothing in it is written to
evaluation cards. API keys are never stored in the file; a judge names an
environment variable instead.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

from .errors import InputError
from .recipe import Recipe

SCHEMA = "statesure.install/v1"
JUDGE_KINDS = ("vlm_openai", "jev")
SOURCE_KINDS = ("frigate", "rtsp", "http_image")
CAMERA_KINDS = ("outdoor", "indoor", "unspecified")

_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,47}$")
_ENV = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_TEXT_LIMIT = 400


@dataclass(frozen=True)
class JudgeConfig:
    name: str
    kind: str
    url: str
    model: str | None
    api_key_env: str | None
    timeout_s: float
    max_tokens: int
    json_schema: bool

    @property
    def is_local(self) -> bool:
        """True when the URL points at loopback, a private address or a LAN name."""
        return is_local_url(self.url)

    @property
    def judge_id(self) -> str:
        """Stable id used in the label store: kind, model and a config hash."""
        digest = _digest(
            {"kind": self.kind, "url": self.url, "model": self.model, "schema": self.json_schema}
        )
        return f"{self.kind}:{self.model or 'default'}@{digest[:8]}"


@dataclass(frozen=True)
class SourceConfig:
    kind: str
    url: str
    camera: str | None = None


@dataclass(frozen=True)
class Installation:
    name: str
    recipe_id: str
    judge: str
    source: SourceConfig
    zone_description: str
    target_hint: str
    zones: Mapping[str, str]
    roi: tuple[tuple[float, float, float, float], ...]
    always_verify: bool
    camera_kind: str

    def fingerprint(self) -> str:
        """Changes whenever the camera, crop regions or site wording change."""
        return _digest(
            {
                "name": self.name,
                "source": {
                    "kind": self.source.kind,
                    "url": self.source.url,
                    "camera": self.source.camera,
                },
                "zone": self.zone_description,
                "target": self.target_hint,
                "zones": dict(sorted(self.zones.items())),
                "roi": [list(box) for box in self.roi],
                "always_verify": self.always_verify,
            }
        )[:16]


@dataclass(frozen=True)
class EvaluationConfig:
    save_all: bool
    retention_days: int
    routine_rate: float


@dataclass(frozen=True)
class InstallConfig:
    timezone: ZoneInfo
    store: Path
    recipes_dir: Path
    evaluation: EvaluationConfig
    judges: Mapping[str, JudgeConfig]
    installations: Mapping[str, Installation]


def load_install(path: str | Path) -> InstallConfig:
    """Load and validate an installation config file."""
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text("utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        raise InputError("unreadable_install_config") from None
    return parse_install(data, base_dir=path.parent)


def parse_install(data: object, *, base_dir: Path = Path(".")) -> InstallConfig:
    doc = _mapping(data)
    _only(
        doc,
        {"schema", "timezone", "store", "judges", "installations"},
        {"recipes_dir", "evaluation"},
    )
    if doc["schema"] != SCHEMA:
        raise InputError("unsupported_install_schema")
    try:
        timezone = ZoneInfo(str(doc["timezone"]))
    except (ZoneInfoNotFoundError, ValueError):
        raise InputError("unknown_time_zone") from None

    judges = {name: _judge(name, body) for name, body in _mapping(doc["judges"]).items()}
    if not judges:
        raise InputError("no_judges")
    installations = {
        name: _installation(name, body, judges)
        for name, body in _mapping(doc["installations"]).items()
    }
    if not installations:
        raise InputError("no_installations")
    return InstallConfig(
        timezone=timezone,
        store=(base_dir / str(doc["store"])).resolve(),
        recipes_dir=(base_dir / str(doc.get("recipes_dir", "recipes"))).resolve(),
        evaluation=_evaluation(doc.get("evaluation", {})),
        judges=judges,
        installations=installations,
    )


def render_prompt(recipe: Recipe, installation: Installation, lang: str = "en") -> str:
    """Fill the recipe prompt with the installation's wording."""
    template = recipe.prompt.get(lang) or recipe.prompt["en"]
    text = re.sub(r"\{\{\s*zone\.description\s*\}\}", installation.zone_description, template)
    text = re.sub(r"\{\{\s*target\.hint\s*\}\}", installation.target_hint, text)
    zone_lines = [
        f"- {value}: {installation.zones[value]}"
        for spec in recipe.signals
        if spec.zones_from == "install"
        for value in spec.values
        if value in installation.zones
    ]
    if zone_lines:
        text = text.rstrip() + "\nLocation values:\n" + "\n".join(zone_lines)
    return text.strip()


def is_local_url(url: str) -> bool:
    host = urlsplit(url).hostname or ""
    if host in ("localhost",) or host.endswith((".local", ".lan", ".internal", ".home.arpa")):
        return True
    if "." not in host and ":" not in host:
        # Single-label names (e.g. a Kubernetes service or container name).
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host.endswith(".svc") or ".svc." in host
    return address.is_private or address.is_loopback or address.is_link_local


def _judge(name: str, body: object) -> JudgeConfig:
    if not _NAME.fullmatch(str(name)):
        raise InputError("invalid_judge_name")
    doc = _mapping(body)
    _only(doc, {"kind", "url"}, {"model", "api_key_env", "timeout_s", "max_tokens", "json_schema"})
    kind = doc["kind"]
    if kind not in JUDGE_KINDS:
        raise InputError("invalid_judge_kind")
    url = _http_url(doc["url"])
    model = doc.get("model")
    if model is not None and (not isinstance(model, str) or not 0 < len(model) <= 128):
        raise InputError("invalid_model")
    if kind == "vlm_openai" and model is None:
        raise InputError("model_required")
    api_key_env = doc.get("api_key_env")
    if api_key_env is not None and not _ENV.fullmatch(str(api_key_env)):
        raise InputError("invalid_api_key_env")
    timeout_s = doc.get("timeout_s", 60)
    if type(timeout_s) not in (int, float) or not 1 <= timeout_s <= 600:
        raise InputError("invalid_timeout")
    max_tokens = doc.get("max_tokens", 400)
    if type(max_tokens) is not int or not 16 <= max_tokens <= 4096:
        raise InputError("invalid_max_tokens")
    json_schema = doc.get("json_schema", True)
    if type(json_schema) is not bool:
        raise InputError("invalid_json_schema_flag")
    return JudgeConfig(
        name=name,
        kind=kind,
        url=url,
        model=model,
        api_key_env=api_key_env,
        timeout_s=float(timeout_s),
        max_tokens=max_tokens,
        json_schema=json_schema,
    )


def _installation(name: str, body: object, judges: Mapping[str, JudgeConfig]) -> Installation:
    if not _NAME.fullmatch(str(name)):
        raise InputError("invalid_installation_name")
    doc = _mapping(body)
    _only(
        doc,
        {"source", "recipe", "judge"},
        {"zone", "target", "zones", "roi", "always_verify", "camera_kind"},
    )
    if doc["judge"] not in judges:
        raise InputError("unknown_judge")
    recipe_id = doc["recipe"]
    if not isinstance(recipe_id, str) or not re.fullmatch(r"[a-z][a-z0-9_]{2,47}", recipe_id):
        raise InputError("invalid_recipe_reference")
    zone = _mapping(doc.get("zone", {}))
    _only(zone, set(), {"description"})
    target = _mapping(doc.get("target", {}))
    _only(target, set(), {"hint"})
    zones = _mapping(doc.get("zones", {}))
    for value, text in zones.items():
        if not isinstance(value, str) or not _NAME.fullmatch(value):
            raise InputError("invalid_zone_value")
        _text(text)
    always_verify = doc.get("always_verify", False)
    if type(always_verify) is not bool:
        raise InputError("invalid_always_verify")
    camera_kind = doc.get("camera_kind", "unspecified")
    if camera_kind not in CAMERA_KINDS:
        raise InputError("invalid_camera_kind")
    return Installation(
        name=name,
        recipe_id=recipe_id,
        judge=doc["judge"],
        source=_source(doc["source"]),
        zone_description=_text(zone.get("description", "the target area")),
        target_hint=_text(target.get("hint", ""), allow_empty=True),
        zones={str(k): str(v) for k, v in zones.items()},
        roi=_roi(doc.get("roi", [])),
        always_verify=always_verify,
        camera_kind=camera_kind,
    )


def _source(body: object) -> SourceConfig:
    doc = _mapping(body)
    _only(doc, {"kind", "url"}, {"camera"})
    kind = doc["kind"]
    if kind not in SOURCE_KINDS:
        raise InputError("invalid_source_kind")
    url = str(doc["url"])
    if kind == "rtsp":
        parts = urlsplit(url)
        if parts.scheme not in ("rtsp", "rtsps") or not parts.hostname:
            raise InputError("invalid_rtsp_url")
        if parts.username or parts.password:
            raise InputError("credentials_in_url")
    else:
        url = _http_url(url)
    camera = doc.get("camera")
    if kind == "frigate":
        if not isinstance(camera, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", camera):
            raise InputError("frigate_camera_required")
    elif camera is not None:
        raise InputError("camera_only_for_frigate")
    return SourceConfig(kind=kind, url=url.rstrip("/"), camera=camera)


def _roi(raw: object) -> tuple[tuple[float, float, float, float], ...]:
    if not isinstance(raw, list) or len(raw) > 4:
        raise InputError("invalid_roi")
    boxes: list[tuple[float, float, float, float]] = []
    for box in raw:
        if not isinstance(box, list) or len(box) != 4:
            raise InputError("invalid_roi")
        if not all(type(v) in (int, float) and 0.0 <= v <= 1.0 for v in box):
            raise InputError("invalid_roi")
        x0, y0, x1, y1 = (float(v) for v in box)
        if x1 - x0 < 0.05 or y1 - y0 < 0.05:
            raise InputError("invalid_roi")
        boxes.append((x0, y0, x1, y1))
    return tuple(boxes)


def _evaluation(raw: object) -> EvaluationConfig:
    doc = _mapping(raw)
    _only(doc, set(), {"save_all", "retention_days", "routine_rate"})
    save_all = doc.get("save_all", False)
    retention = doc.get("retention_days", 7)
    rate = doc.get("routine_rate", 1 / 6)
    if type(save_all) is not bool:
        raise InputError("invalid_save_all")
    if type(retention) is not int or not 1 <= retention <= 90:
        raise InputError("invalid_retention_days")
    if type(rate) not in (int, float) or not 0 < rate <= 1:
        raise InputError("invalid_routine_rate")
    return EvaluationConfig(save_all=save_all, retention_days=retention, routine_rate=float(rate))


def _http_url(value: object) -> str:
    if not isinstance(value, str):
        raise InputError("invalid_url")
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise InputError("invalid_url")
    if parts.username or parts.password:
        # Credentials belong in environment variables, not in URLs.
        raise InputError("credentials_in_url")
    return value


def _text(value: object, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > _TEXT_LIMIT or "{{" in value:
        raise InputError("invalid_text")
    if not allow_empty and not value.strip():
        raise InputError("invalid_text")
    return " ".join(value.split())


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InputError("expected_mapping")
    return value


def _only(doc: Mapping[str, Any], required: set[str], optional: set[str]) -> None:
    keys = set(doc)
    if not required <= keys:
        raise InputError("missing_key")
    if keys - required - optional:
        raise InputError("unknown_key")
