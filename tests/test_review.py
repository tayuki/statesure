from __future__ import annotations

import http.client
import threading
from collections.abc import Iterator
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlencode

import pytest

from statesure.errors import StatesureError
from statesure.readings import SignalReading, new_sample_id
from statesure.recipe import Recipe
from statesure.review import ReviewApp, serve
from statesure.sampling import ReviewClass
from statesure.store import LabelStore

from .conftest import RECIPES, T0, presence_obs
from .test_run import synthetic_jpeg

JUDGE = "vlm_openai:secret-judge-name"


def _sample(store: LabelStore, recipe: Recipe, priority: str = "routine") -> str:
    sample_id = new_sample_id()
    store.add_sample(
        sample_id=sample_id,
        source_id="cam-a",
        install_fingerprint="install-a",
        captured_at=T0,
        recipe=recipe,
        review=ReviewClass("ordinary", priority, True, 1.0, 0.5),
        save_all=True,
        image=synthetic_jpeg((64, 48)),
        now=T0,
        retention=timedelta(days=7),
    )
    store.add_observation(
        presence_obs(
            recipe,
            None,
            T0,
            sample_id=sample_id,
            judge_id=JUDGE,
            reading=SignalReading.decided("present"),
        )
    )
    return sample_id


class Clock:
    def __init__(self) -> None:
        self.now = T0 + timedelta(hours=1)

    def __call__(self):
        return self.now


@pytest.fixture
def running(tmp_path: Path, package_recipe: Recipe) -> Iterator[tuple]:
    store = LabelStore(tmp_path / "store.sqlite")
    clock = Clock()
    app = ReviewApp(store, RECIPES, clock=clock)
    server = serve(app, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield app, store, clock, server.server_address[1]
    server.shutdown()
    server.server_close()


def _request(port: int, method: str, path: str, *, body: str = "", headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    all_headers = {"Host": f"127.0.0.1:{port}"}
    if body:
        all_headers["Content-Type"] = "application/x-www-form-urlencoded"
    all_headers.update(headers or {})
    conn.request(method, path, body=body.encode() if body else None, headers=all_headers)
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return response, data


def test_queue_page_hides_the_judge_answer(running, package_recipe: Recipe) -> None:
    app, store, _, port = running
    sample_id = _sample(store, package_recipe)
    response, data = _request(port, "GET", "/")
    page = data.decode()
    assert response.status == 200
    assert sample_id in page
    assert "secret-judge-name" not in page
    assert "default-src 'none'" in response.getheader("Content-Security-Policy")
    image, body = _request(port, "GET", f"/image/{sample_id}")
    assert image.status == 200 and image.getheader("Content-Type") == "image/jpeg"
    assert body.startswith(b"\xff\xd8")


def test_forbidden_host_and_bad_paths(running) -> None:
    _, _, _, port = running
    response, _ = _request(port, "GET", "/", headers={"Host": "evil.example"})
    assert response.status == 403
    response, _ = _request(port, "GET", "/image/../../etc/passwd")
    assert response.status == 404


def test_label_submission(running, package_recipe: Recipe) -> None:
    app, store, _, port = running
    sample_id = _sample(store, package_recipe)
    fields = {
        "token": app.token,
        "signal:package_present": "present",
        "signal:package_count": "2",
        "signal:package_location": "human_uncertain",
        "reviewer_confidence": "confident",
    }
    response, _ = _request(port, "POST", f"/label/{sample_id}", body=urlencode(fields))
    assert response.status == 303
    (record,) = store.records()
    assert record.labels["package_present"].truth == "present"
    assert record.labels["package_count"].truth == 2
    assert record.labels["package_location"].truth == "human_uncertain"
    # Labeled samples leave the queue.
    assert app.queue() == []


def test_not_assessable_submission(running, package_recipe: Recipe) -> None:
    app, store, _, port = running
    sample_id = _sample(store, package_recipe)
    fields = {"token": app.token, "assessable": "no", "reviewer_confidence": "unsure"}
    response, _ = _request(port, "POST", f"/label/{sample_id}", body=urlencode(fields))
    assert response.status == 303
    (record,) = store.records()
    assert all(not label.assessable for label in record.labels.values())


def test_token_and_origin_are_required(running, package_recipe: Recipe) -> None:
    app, store, _, port = running
    sample_id = _sample(store, package_recipe)
    fields = {"signal:package_present": "present", "reviewer_confidence": "confident"}
    response, _ = _request(port, "POST", f"/label/{sample_id}", body=urlencode(fields))
    assert response.status == 403
    fields["token"] = app.token
    response, _ = _request(
        port,
        "POST",
        f"/label/{sample_id}",
        body=urlencode(fields),
        headers={"Origin": "http://evil.example"},
    )
    assert response.status == 403
    assert store.records()[0].labels == {}


def test_invalid_label_values_are_rejected(running, package_recipe: Recipe) -> None:
    app, store, _, port = running
    sample_id = _sample(store, package_recipe)
    fields = {
        "token": app.token,
        "signal:package_present": "maybe",
        "reviewer_confidence": "confident",
    }
    response, data = _request(port, "POST", f"/label/{sample_id}", body=urlencode(fields))
    assert response.status == 400
    assert b"invalid_truth" in data


def test_invalid_field_leaves_no_partial_labels(running, package_recipe: Recipe) -> None:
    """Regression (#4 review): all labels of a sample are written together."""
    app, store, _, port = running
    sample_id = _sample(store, package_recipe)
    fields = {
        "token": app.token,
        "signal:package_present": "present",
        "signal:package_count": "99",
        "signal:package_location": "doorstep",
        "reviewer_confidence": "confident",
    }
    response, _ = _request(port, "POST", f"/label/{sample_id}", body=urlencode(fields))
    assert response.status == 400
    assert store.records()[0].labels == {}
    assert len(app.queue()) == 1


def test_recipe_changes_are_picked_up(tmp_path: Path, package_doc: dict) -> None:
    """Regression (#4 review): the version guard follows the current file."""
    import os

    import yaml

    from statesure import recipe as recipe_module

    recipes = tmp_path / "recipes"
    recipes.mkdir()
    path = recipes / "package_at_door.yaml"
    path.write_text(yaml.safe_dump(package_doc), "utf-8")
    recipe = recipe_module.load(path)
    store = LabelStore(tmp_path / "store.sqlite")
    _sample(store, recipe)
    app = ReviewApp(store, recipes, clock=Clock())
    assert len(app.queue()) == 1
    package_doc["version"] = 2
    path.write_text(yaml.safe_dump(package_doc), "utf-8")
    os.utime(path, ns=(1, 1))
    assert app.queue() == []
    path.unlink()
    assert app.queue() == []
    package_doc["version"] = 1
    path.write_text(yaml.safe_dump(package_doc), "utf-8")
    assert len(app.queue()) == 1


def test_expired_images_are_not_served(running, package_recipe: Recipe) -> None:
    app, store, clock, port = running
    sample_id = _sample(store, package_recipe)
    clock.now = T0 + timedelta(days=7, seconds=1)
    response, _ = _request(port, "GET", f"/image/{sample_id}")
    assert response.status == 404
    assert app.queue() == []


def test_old_recipe_versions_are_not_labeled(running, package_recipe: Recipe) -> None:
    app, store, _, _ = running
    _sample(store, replace(package_recipe, fingerprint="0" * 16))
    assert app.queue() == []


def test_queue_order_follows_priority(running, package_recipe: Recipe) -> None:
    app, store, _, _ = running
    routine = _sample(store, package_recipe, "routine")
    critical = _sample(store, package_recipe, "critical")
    assert [record.sample_id for record, _ in app.queue()] == [critical, routine]


def test_serve_refuses_non_loopback(tmp_path: Path) -> None:
    app = ReviewApp(LabelStore(tmp_path / "s.sqlite"), RECIPES)
    with pytest.raises(StatesureError):
        serve(app, host="0.0.0.0", port=0)


def test_page_escapes_recipe_text(tmp_path: Path, package_doc: dict) -> None:
    import yaml

    from statesure import recipe as recipe_module
    from statesure.review import _queue_page

    package_doc["title"] = {"en": "<script>alert(1)</script>"}
    recipes = tmp_path / "recipes"
    recipes.mkdir()
    (recipes / "package_at_door.yaml").write_text(yaml.safe_dump(package_doc), "utf-8")
    recipe = recipe_module.load(recipes / "package_at_door.yaml")
    store = LabelStore(tmp_path / "store.sqlite")
    _sample(store, recipe)
    page = _queue_page(ReviewApp(store, recipes, clock=Clock()))
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
