from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Mapping
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from statesure.errors import InputError
from statesure.install import is_local_url, load_install, parse_install, render_prompt
from statesure.judges import JudgeError, build_judge
from statesure.judges.base import MAX_RESPONSE_BYTES, http_post
from statesure.judges.vlm_openai import format_instructions, response_schema
from statesure.readings import Outcome
from statesure.recipe import Recipe

IMAGE = b"\xff\xd8synthetic-jpeg\xff\xd9"


def _config(**overrides) -> dict:
    doc = {
        "schema": "statesure.install/v1",
        "timezone": "Asia/Tokyo",
        "store": "data/labels.sqlite",
        "evaluation": {"save_all": True, "retention_days": 7},
        "judges": {
            "local-vlm": {
                "kind": "vlm_openai",
                "url": "http://127.0.0.1:8000/v1/chat/completions",
                "model": "test-vlm",
            },
            "local-jev": {
                "kind": "jev",
                "url": "http://djev.lan:8000/v1/systemone",
                "model": "djev-0.1",
            },
        },
        "installations": {
            "front-door": {
                "source": {"kind": "frigate", "url": "http://frigate:5000", "camera": "front"},
                "recipe": "package_at_door",
                "judge": "local-vlm",
                "zone": {"description": "the step in front of the door"},
                "zones": {"doorstep": "on the step", "side": "beside the step"},
                "roi": [[0.0, 0.2, 0.68, 1.0], [0.32, 0.2, 1.0, 1.0]],
                "camera_kind": "outdoor",
            }
        },
    }
    doc.update(overrides)
    return doc


def test_parse_install(tmp_path: Path) -> None:
    config = parse_install(_config(), base_dir=tmp_path)
    assert config.store == (tmp_path / "data/labels.sqlite").resolve()
    install = config.installations["front-door"]
    assert install.roi[0] == (0.0, 0.2, 0.68, 1.0)
    assert config.judges["local-vlm"].is_local
    assert config.judges["local-vlm"].judge_id.startswith("vlm_openai:test-vlm@")


def test_load_install_file(tmp_path: Path) -> None:
    import yaml

    path = tmp_path / "install.yaml"
    path.write_text(yaml.safe_dump(_config()), "utf-8")
    assert "front-door" in load_install(path).installations


def test_install_fingerprint_tracks_site_changes() -> None:
    install = parse_install(_config()).installations["front-door"]
    base = install.fingerprint()
    assert replace(install, roi=((0.1, 0.1, 0.9, 0.9),)).fingerprint() != base
    assert replace(install, zone_description="the porch").fingerprint() != base
    assert replace(install, camera_kind="indoor").fingerprint() == base


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda d: d["judges"]["local-vlm"].update(url="ftp://x"), "invalid_url"),
        (
            lambda d: d["judges"]["local-vlm"].update(url="http://user:pw@host/v1"),
            "credentials_in_url",
        ),
        (lambda d: d["judges"]["local-vlm"].update(api_key_env="lower"), "invalid_api_key_env"),
        (lambda d: d["judges"]["local-vlm"].pop("model"), "model_required"),
        (lambda d: d["judges"]["local-vlm"].update(api_key="secret"), "unknown_key"),
        (
            lambda d: d["installations"]["front-door"].update(roi=[[0.5, 0.5, 0.51, 0.9]]),
            "invalid_roi",
        ),
        (
            lambda d: d["installations"]["front-door"].update(
                source={"kind": "rtsp", "url": "rtsp://admin:pw@cam/stream"}
            ),
            "credentials_in_url",
        ),
        (
            lambda d: d["installations"]["front-door"]["source"].pop("camera"),
            "frigate_camera_required",
        ),
        (lambda d: d["installations"]["front-door"].update(judge="ghost"), "unknown_judge"),
        (lambda d: d.update(timezone="Mars/Olympus"), "unknown_time_zone"),
        (
            lambda d: d["installations"]["front-door"].update(zone={"description": "{{ x }}"}),
            "invalid_text",
        ),
    ],
)
def test_install_rejects_invalid(mutate, code: str) -> None:
    doc = _config()
    mutate(doc)
    with pytest.raises(InputError) as info:
        parse_install(doc)
    assert info.value.code == code


@pytest.mark.parametrize(
    ("url", "local"),
    [
        ("http://127.0.0.1:8000", True),
        ("http://192.168.1.20/v1", True),
        ("http://frigate:5000", True),
        ("http://vlm.local", True),
        ("http://djev.default.svc.cluster.local", True),
        ("https://api.typesafe.ai/v1/systemone", False),
        ("https://8.8.8.8/v1", False),
    ],
)
def test_is_local_url(url: str, local: bool) -> None:
    assert is_local_url(url) is local


def test_render_prompt(package_recipe: Recipe) -> None:
    install = parse_install(_config()).installations["front-door"]
    text = render_prompt(package_recipe, install)
    assert "the step in front of the door" in text
    assert "{{" not in text
    assert "- doorstep: on the step" in text


class Recorder:
    """A fake transport that records requests and returns a canned reply."""

    def __init__(self, reply: object) -> None:
        self.reply = reply
        self.calls: list[tuple[str, Mapping[str, str], dict]] = []

    def __call__(self, url, headers, body, timeout_s) -> bytes:
        self.calls.append((url, headers, json.loads(body)))
        return json.dumps(self.reply).encode()


def _vlm_reply(content: dict) -> dict:
    return {"choices": [{"message": {"content": json.dumps(content)}}]}


def test_vlm_judge_request_and_reply(package_recipe: Recipe) -> None:
    config = parse_install(_config()).judges["local-vlm"]
    fake = Recorder(
        _vlm_reply(
            {
                "visibility": "good",
                "analysis_confidence": "high",
                "package_present": "present",
                "package_count": 2,
                "package_location": "doorstep",
            }
        )
    )
    result = build_judge(config, fake).judge(package_recipe, "PROMPT", [IMAGE, IMAGE])
    assert result.signals["package_count"].value == 2
    url, headers, body = fake.calls[0]
    assert url == config.url and "Authorization" not in headers
    parts = body["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["image_url", "image_url", "text"]
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert parts[2]["text"].startswith("PROMPT\n\nReply with one JSON object only")
    assert body["temperature"] == 0
    assert body["response_format"]["json_schema"]["schema"] == response_schema(package_recipe)


def test_vlm_judge_without_schema_and_bad_replies(package_recipe: Recipe) -> None:
    config = replace(parse_install(_config()).judges["local-vlm"], json_schema=False)
    fake = Recorder(_vlm_reply({"package_present": "present"}))
    result = build_judge(config, fake).judge(package_recipe, "P", [IMAGE])
    assert "response_format" not in fake.calls[0][2]
    # Missing quality fields: nothing is accepted.
    assert {r.outcome for r in result.signals.values()} == {Outcome.ABSTAINED}
    with pytest.raises(JudgeError) as info:
        build_judge(config, Recorder({"nope": 1})).judge(package_recipe, "P", [IMAGE])
    assert info.value.code == "invalid_response_shape"
    with pytest.raises(JudgeError):
        build_judge(config, Recorder({})).judge(package_recipe, "P", [])


def test_format_instructions_and_schema_cover_every_key(package_recipe: Recipe) -> None:
    text = format_instructions(package_recipe)
    schema = response_schema(package_recipe)
    for key in ("visibility", "analysis_confidence", *package_recipe.signal_names):
        assert f"- {key}:" in text
        assert key in schema["required"]
    assert schema["additionalProperties"] is False


def _jev_reply(model: str = "djev-0.1", present: float = 0.9) -> dict:
    count = {str(i): 0.0 for i in range(11)}
    count["1"] = 1.0
    location = dict.fromkeys(("none", "doorstep", "side", "walkway", "multiple"), 0.0)
    location["doorstep"] = 1.0
    return {
        "model": model,
        "answers": {
            "package_present": {"type": "noul", "noul": present},
            "package_count": {
                "type": "choice",
                "choice": "1",
                "probabilities": count,
                "confidence": 1.0,
            },
            "package_location": {
                "type": "choice",
                "choice": "doorstep",
                "probabilities": location,
                "confidence": 1.0,
            },
        },
    }


def test_jev_judge_request_and_reply(package_recipe: Recipe) -> None:
    config = parse_install(_config()).judges["local-jev"]
    fake = Recorder(_jev_reply())
    result = build_judge(config, fake).judge(package_recipe, "STATE", [IMAGE])
    assert result.signals["package_present"].probability == pytest.approx(0.9)
    _, _, body = fake.calls[0]
    assert body["state"] == "STATE" and body["model"] == "djev-0.1"
    assert body["questions"]["package_present"] == {
        "type": "noul",
        "instructions": "Is there a package in the target area?",
    }
    criteria = body["questions"]["package_count"]["criteria"]
    assert set(criteria) == {str(i) for i in range(11)}
    assert body["images"][0].startswith("data:image/jpeg;base64,")


def test_jev_judge_rejects_model_mismatch_and_image_count(package_recipe: Recipe) -> None:
    config = parse_install(_config()).judges["local-jev"]
    with pytest.raises(JudgeError) as info:
        build_judge(config, Recorder(_jev_reply(model="other"))).judge(package_recipe, "S", [IMAGE])
    assert info.value.code == "model_mismatch"
    with pytest.raises(JudgeError):
        build_judge(config, Recorder(_jev_reply())).judge(package_recipe, "S", [IMAGE, IMAGE])


def test_api_key_comes_from_environment(
    package_recipe: Recipe, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(parse_install(_config()).judges["local-jev"], api_key_env="STATESURE_KEY")
    with pytest.raises(JudgeError) as info:
        build_judge(config, Recorder(_jev_reply())).judge(package_recipe, "S", [IMAGE])
    assert info.value.code == "missing_api_key"
    monkeypatch.setenv("STATESURE_KEY", "test-token")
    fake = Recorder(_jev_reply())
    build_judge(config, fake).judge(package_recipe, "S", [IMAGE])
    assert fake.calls[0][1]["Authorization"] == "Bearer test-token"


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:1/elsewhere")
            self.end_headers()
        elif self.path == "/huge":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"x" * (MAX_RESPONSE_BYTES + 10))
        elif self.path == "/error":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"secret internal detail")
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"ok": true}')


@pytest.fixture
def server() -> Iterator[str]:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_http_post_against_loopback(server: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # An environment proxy must not be used.
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    assert http_post(server + "/ok", {}, b"{}", 5) == b'{"ok": true}'
    for path, code in (("/redirect", "http_302"), ("/huge", "response_too_large")):
        with pytest.raises(JudgeError) as info:
            http_post(server + path, {}, b"{}", 5)
        assert info.value.code == code
    with pytest.raises(JudgeError) as info:
        http_post(server + "/error", {}, b"{}", 5)
    assert info.value.code == "http_500"
    assert "secret" not in str(info.value)
