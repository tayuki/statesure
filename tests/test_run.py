from __future__ import annotations

import io
import json
import random
import subprocess
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import pytest
from PIL import Image

from statesure import images as image_ops
from statesure import sources
from statesure.errors import InputError
from statesure.install import SourceConfig, parse_install
from statesure.judges import JudgeError
from statesure.pipeline import ObservationLog, run_lock, run_once
from statesure.readings import Outcome
from statesure.store import LabelStore

from .conftest import RECIPES, T0
from .test_install_judges import _config, _jev_reply, _vlm_reply

HALF_HOUR = timedelta(minutes=30)


def synthetic_jpeg(size: tuple[int, int] = (1920, 1080), *, exif: bool = False) -> bytes:
    image = Image.new("RGB", size, (90, 120, 150))
    output = io.BytesIO()
    if exif:
        data = Image.Exif()
        data[0x010F] = "synthetic-camera"
        image.save(output, format="JPEG", exif=data)
    else:
        image.save(output, format="JPEG")
    return output.getvalue()


def _dims(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as image:
        return image.size


def test_views_are_resized_and_stripped() -> None:
    raw = synthetic_jpeg(exif=True)
    primary = image_ops.primary_view(raw)
    assert _dims(primary) == (960, 540)
    with Image.open(io.BytesIO(primary)) as image:
        assert not image.getexif()
    views = image_ops.verification_views(raw, [(0.0, 0.2, 0.68, 1.0), (0.32, 0.2, 1.0, 1.0)])
    assert len(views) == 3
    assert _dims(views[0]) == (1280, 720)


def test_small_crops_are_enlarged() -> None:
    raw = synthetic_jpeg((320, 240))
    (full, crop) = image_ops.verification_views(raw, [(0.0, 0.0, 0.5, 0.5)])
    assert _dims(crop)[0] > 160


def test_invalid_images_are_rejected() -> None:
    with pytest.raises(image_ops.ImageError) as info:
        image_ops.primary_view(b"not an image")
    assert info.value.code == "invalid_image"


def test_rtsp_credentials_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    source = SourceConfig(
        kind="rtsp",
        url="rtsp://camera.lan:554/stream1",
        username_env="CAM_USER",
        password_env="CAM_PASS",
    )
    with pytest.raises(sources.CaptureError) as info:
        sources.rtsp_url_with_credentials(source)
    assert info.value.code == "missing_rtsp_credentials"
    monkeypatch.setenv("CAM_USER", "viewer")
    monkeypatch.setenv("CAM_PASS", "p@ss:word")
    url = sources.rtsp_url_with_credentials(source)
    assert url == "rtsp://viewer:p%40ss%3Aword@camera.lan:554/stream1"


def test_rtsp_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    source = SourceConfig(kind="rtsp", url="rtsp://camera.lan/stream")
    monkeypatch.setattr(sources.shutil, "which", lambda name: None)
    with pytest.raises(sources.CaptureError) as info:
        sources.capture(source)
    assert info.value.code == "ffmpeg_not_found"

    monkeypatch.setattr(sources.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    seen: dict = {}

    def fake_run(command, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(command, 1, stdout=b"", stderr=None)

    monkeypatch.setattr(sources.subprocess, "run", fake_run)
    with pytest.raises(sources.CaptureError) as info:
        sources.capture(source)
    assert info.value.code == "rtsp_failed"
    # ffmpeg's stderr may echo the URL, so it is never captured.
    assert seen["stderr"] is subprocess.DEVNULL


class ScriptedTransport:
    """Returns queued replies (or raises queued errors) in order."""

    def __init__(self, *replies: object) -> None:
        self.replies = list(replies)
        self.bodies: list[dict] = []

    def __call__(self, url, headers, body, timeout_s) -> bytes:
        self.bodies.append(json.loads(body))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return json.dumps(reply).encode()


def _present_reply() -> dict:
    return _vlm_reply(
        {
            "visibility": "good",
            "analysis_confidence": "high",
            "package_present": "present",
            "package_count": 1,
            "package_location": "doorstep",
        }
    )


def _setup(tmp_path: Path, *, save_all: bool = True, judge: str = "local-vlm"):
    doc = _config()
    doc["recipes_dir"] = str(RECIPES)
    doc["evaluation"] = {"save_all": save_all, "retention_days": 7}
    doc["installations"]["front-door"]["judge"] = judge
    config = parse_install(doc, base_dir=tmp_path)
    return config


def _fetch(raw: bytes | None = None) -> Callable[[SourceConfig], bytes]:
    data = raw or synthetic_jpeg()
    return lambda source: data


def test_first_run_verifies_and_stores(tmp_path: Path) -> None:
    config = _setup(tmp_path)
    transport = ScriptedTransport(_present_reply(), _present_reply())
    result = run_once(config, "front-door", now=T0, fetch=_fetch(), transport=transport)
    assert result.status == "ok"
    assert result.reasons == ("baseline_unconfirmed",)
    assert result.agreement == "full"
    # Primary with one image, then the re-check with the full frame and two crops.
    counts = [
        sum(p["type"] == "image_url" for p in body["messages"][0]["content"])
        for body in transport.bodies
    ]
    assert counts == [1, 3]
    assert "focused re-check" in transport.bodies[1]["messages"][0]["content"][-1]["text"]
    store = LabelStore(config.store)
    (record,) = store.records("package_at_door")
    stages = {stage for (_, stage, _) in record.readings}
    assert stages == {"primary", "verified", "merged", "confirmed"}
    assert store.image_path(record.sample_id) is not None
    assert len(ObservationLog(config.observations).read()) == 1


def test_second_matching_run_confirms(tmp_path: Path) -> None:
    config = _setup(tmp_path)
    transport = ScriptedTransport(*[_present_reply()] * 4)
    run_once(config, "front-door", now=T0, fetch=_fetch(), transport=transport)
    result = run_once(config, "front-door", now=T0 + HALF_HOUR, fetch=_fetch(), transport=transport)
    types = {(e["signal"], e["event_type"]) for e in result.events}
    assert ("package_present", "baseline_confirmed") in types
    assert result.signals["package_present"]["confirmed"] == "present"


def test_judge_failure_is_kept_for_review(tmp_path: Path) -> None:
    """Regression (P1(a)): a failed judge call is stored as an error sample."""
    config = _setup(tmp_path)
    transport = ScriptedTransport(JudgeError("http_500"))
    result = run_once(config, "front-door", now=T0, fetch=_fetch(), transport=transport)
    assert result.judge_failed and result.stratum == "error" and result.stored
    assert len(transport.bodies) == 1  # no re-check after a failure
    (record,) = LabelStore(config.store).records("package_at_door")
    reading = next(r for (_, stage, _), r in record.readings.items() if stage == "primary")
    assert reading.outcome is Outcome.ERROR
    assert LabelStore(config.store).image_path(record.sample_id) is not None


def test_capture_failure_records_nothing(tmp_path: Path) -> None:
    config = _setup(tmp_path)

    def broken(source):
        raise sources.CaptureError("http_404")

    result = run_once(config, "front-door", now=T0, fetch=broken, transport=ScriptedTransport())
    assert result.status == "http_404"
    assert not config.observations.exists()


def test_unstored_samples_still_count_for_confirmation(tmp_path: Path) -> None:
    config = _setup(tmp_path, save_all=False)
    transport = ScriptedTransport(*[_present_reply()] * 4)

    class Never(random.Random):
        def random(self) -> float:
            return 0.99

    first = run_once(config, "front-door", now=T0, fetch=_fetch(), transport=transport, rng=Never())
    second = run_once(
        config,
        "front-door",
        now=T0 + HALF_HOUR,
        fetch=_fetch(),
        transport=transport,
        rng=Never(),
    )
    # "baseline_unconfirmed" triggers a re-check but is not a review stratum, so these
    # ordinary samples are not selected; confirmation still uses both of them.
    assert first.stored is False and second.stored is False
    assert any(e["event_type"] == "baseline_confirmed" for e in second.events)
    assert len(ObservationLog(config.observations).read()) == 2


def test_jev_recheck_uses_one_crop(tmp_path: Path) -> None:
    config = _setup(tmp_path, judge="local-jev")
    transport = ScriptedTransport(_jev_reply(), _jev_reply())
    result = run_once(config, "front-door", now=T0, fetch=_fetch(), transport=transport)
    assert result.status == "ok"
    assert [len(body["images"]) for body in transport.bodies] == [1, 1]
    assert "enlarged crop" in transport.bodies[1]["state"]


def test_expired_images_are_purged_by_later_runs(tmp_path: Path) -> None:
    config = _setup(tmp_path)
    # Both runs are re-checked: the baseline is still unconfirmed after one sample.
    transport = ScriptedTransport(*[_present_reply()] * 4)
    first = run_once(config, "front-door", now=T0, fetch=_fetch(), transport=transport)
    store = LabelStore(config.store)
    assert store.image_path(first.sample_id) is not None
    run_once(config, "front-door", now=T0 + timedelta(days=8), fetch=_fetch(), transport=transport)
    assert store.image_path(first.sample_id) is None


def test_run_lock_is_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "store.sqlite"
    with run_lock(path) as first:
        assert first
        with run_lock(path) as second:
            assert not second


def test_observation_log_skips_torn_lines(tmp_path: Path) -> None:
    config = _setup(tmp_path)
    transport = ScriptedTransport(_present_reply(), _present_reply())
    run_once(config, "front-door", now=T0, fetch=_fetch(), transport=transport)
    with config.observations.open("a", encoding="utf-8") as handle:
        handle.write('{"sample_id": "tor')
    assert len(ObservationLog(config.observations).read()) == 1


def test_unknown_installation(tmp_path: Path) -> None:
    with pytest.raises(InputError):
        run_once(_setup(tmp_path), "garage", now=T0)
