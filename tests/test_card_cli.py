from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from statesure.card import build_card, normalize_model_name, validate_card
from statesure.cli import main
from statesure.errors import CardError
from statesure.metrics import build_report
from statesure.readings import SignalReading
from statesure.recipe import Recipe

from .conftest import RECIPES, T0, presence_obs
from .test_metrics import JUDGE, Builder

D = SignalReading.decided
PACKAGE = str(RECIPES / "package_at_door.yaml")


def _card(tmp_path: Path, recipe: Recipe) -> dict:
    b = Builder(tmp_path, recipe)
    for index in range(5):
        b.add("present" if index % 2 else "absent", {"primary": D("present")})
    report = build_report(recipe, b.store.records(recipe.id))
    return build_card(
        recipe,
        report,
        install_fingerprint="install-a",
        judge_id=JUDGE,
        judge_kind="vlm_openai",
        model="Qwen/Qwen3-VL-8B",
        evaluated_days=1,
        camera_kind="outdoor",
    )


def test_card_contains_counts_only(tmp_path: Path, package_recipe: Recipe) -> None:
    card = _card(tmp_path, package_recipe)
    text = json.dumps(card)
    assert card["model"] == "qwen:qwen3-vl-8b"
    assert card["stages"]["primary"]["package_present"]["scored"] == 5
    for forbidden in ("install-a", "cam-a", "2026-01-05", JUDGE):
        assert forbidden not in text


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c.update(source_id="cam-a"),
        lambda c: c.update(model="2026-01-05t07:00"),
        lambda c: c.update(model="192.168.1.20"),
        lambda c: c.update(model="http:host"),
        lambda c: c.update(camera_kind="front_door"),
        lambda c: c["stages"]["primary"]["package_present"].update(sample_id="x"),
        lambda c: c["stages"].update(raw={}),
    ],
)
def test_card_rejects_extra_data(tmp_path: Path, package_recipe: Recipe, mutation) -> None:
    card = _card(tmp_path, package_recipe)
    mutation(card)
    with pytest.raises(CardError):
        validate_card(card, package_recipe)


def test_model_name_normalization() -> None:
    assert normalize_model_name(" Org/Model-7B ") == "org:model-7b"


def test_cli_validate(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert main(["validate", PACKAGE]) == 0
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema: nope\n", "utf-8")
    assert main(["validate", str(bad)]) == 1
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[0]["ok"] is True and lines[1]["error"] == "missing_key"


def test_cli_replay(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, package_recipe: Recipe
) -> None:
    observations = tmp_path / "obs.jsonl"
    rows = [
        presence_obs(package_recipe, value, T0 + timedelta(minutes=30 * i)).to_dict()
        for i, value in enumerate(["absent", "absent", "present", "present"])
    ]
    observations.write_text("\n".join(json.dumps(row) for row in rows) + "\n", "utf-8")
    states = tmp_path / "states.json"
    code = main(
        [
            "replay",
            "--observations",
            str(observations),
            "--recipe",
            PACKAGE,
            "--states",
            str(states),
        ]
    )
    assert code == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [e["event_type"] for e in events] == ["baseline_confirmed", "appeared"]
    assert json.loads(states.read_text("utf-8"))


def test_cli_labels_report_card_purge(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, package_recipe: Recipe
) -> None:
    b = Builder(tmp_path, package_recipe)
    sample_ids = [b.add(None, {"primary": D("present")}) for _ in range(3)]
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        "\n".join(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "signal": "package_present",
                    "truth": "present",
                    "assessable": True,
                    "reviewer_confidence": "confident",
                    "labeled_at": T0.isoformat(),
                    "label_source": "human",
                }
            )
            for sample_id in sample_ids
        ),
        "utf-8",
    )
    db = str(b.store.path)
    assert main(["import-labels", "--db", db, "--recipe", PACKAGE, str(labels)]) == 0
    assert main(["report", "--db", db, "--recipe", PACKAGE, "--tz", "Asia/Tokyo"]) == 0
    assert (
        main(
            [
                "card",
                "--db",
                db,
                "--recipe",
                PACKAGE,
                "--judge",
                JUDGE,
                "--judge-kind",
                "vlm_openai",
                "--model",
                "test-model",
            ]
        )
        == 0
    )
    assert main(["purge", "--db", db]) == 0
    out = capsys.readouterr().out
    assert '"imported": 3' in out
    assert '"purged": 0' in out


def test_cli_rejects_teacher_labels(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, package_recipe: Recipe
) -> None:
    b = Builder(tmp_path, package_recipe)
    sample_id = b.add(None, {"primary": D("present")})
    labels = tmp_path / "labels.jsonl"
    row = {
        "sample_id": sample_id,
        "signal": "package_present",
        "truth": "present",
        "assessable": True,
        "reviewer_confidence": "confident",
        "labeled_at": T0.isoformat(),
        "label_source": "teacher",
    }
    labels.write_text(json.dumps(row), "utf-8")
    db = str(b.store.path)
    assert main(["import-labels", "--db", db, "--recipe", PACKAGE, str(labels)]) == 2
    err = capsys.readouterr().err
    assert json.loads(err) == {"error": "only_human_labels"}


def test_cli_missing_store(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    db = str(tmp_path / "missing.sqlite")
    assert main(["purge", "--db", db]) == 2
    assert json.loads(capsys.readouterr().err) == {"error": "store_not_found"}


def test_cli_review_reports_port_in_use(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    import socket

    import yaml

    from .test_install_judges import _config

    doc = _config()
    doc["recipes_dir"] = str(RECIPES)
    config = tmp_path / "install.yaml"
    config.write_text(yaml.safe_dump(doc), "utf-8")
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        assert main(["review", "--config", str(config), "--port", str(port)]) == 2
    assert json.loads(capsys.readouterr().err) == {"error": "port_in_use"}
