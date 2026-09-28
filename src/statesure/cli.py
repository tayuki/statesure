"""Command line interface.

Only ``run-once`` opens network connections (to the configured sources and
judges). Errors are reported as fixed codes; input content is never echoed.
"""

from __future__ import annotations

import argparse
import errno
import json
import sys
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from . import recipe as recipe_module
from .card import CAMERA_KINDS, JUDGE_KINDS, build_card
from .confirm import replay
from .errors import InputError, StatesureError
from .install import is_local_url, load_install
from .metrics import build_report
from .readings import Observation
from .store import LabelStore

MAX_LINE_BYTES = 64 * 1024
MAX_ROWS = 100_000


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except StatesureError as exc:
        _emit({"error": exc.code}, sys.stderr)
        return 2
    except (OSError, ValueError):
        _emit({"error": "invalid_input"}, sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="statesure", description=__doc__)
    commands = parser.add_subparsers(required=True)

    validate = commands.add_parser("validate", help="validate recipe files")
    validate.add_argument("recipes", nargs="+", type=Path)
    validate.set_defaults(handler=_validate)

    replay_cmd = commands.add_parser("replay", help="rebuild confirmed events from observations")
    replay_cmd.add_argument("--observations", required=True, type=Path)
    replay_cmd.add_argument("--recipe", action="append", required=True, type=Path)
    replay_cmd.add_argument("--states", type=Path, help="write final states as JSON")
    replay_cmd.set_defaults(handler=_replay)

    report = commands.add_parser("report", help="metrics from a label store")
    report.add_argument("--db", required=True, type=Path)
    report.add_argument("--recipe", required=True, type=Path)
    report.add_argument("--judge")
    report.add_argument("--tz", help="IANA time zone used to group days (e.g. Asia/Tokyo)")
    report.set_defaults(handler=_report)

    card = commands.add_parser("card", help="evaluation card without images")
    card.add_argument("--db", required=True, type=Path)
    card.add_argument("--recipe", required=True, type=Path)
    card.add_argument("--judge", required=True)
    card.add_argument("--judge-kind", required=True, choices=JUDGE_KINDS)
    card.add_argument("--model", required=True)
    card.add_argument("--camera-kind", default="unspecified", choices=CAMERA_KINDS)
    card.add_argument("--install", help="installation fingerprint (required if several)")
    card.add_argument("--tz", default="UTC")
    card.set_defaults(handler=_card)

    run = commands.add_parser("run-once", help="capture, judge and record one sample")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--installation", help="run only this installation")
    run.set_defaults(handler=_run_once)

    review = commands.add_parser("review", help="label stored samples in a local web page")
    review.add_argument("--config", required=True, type=Path)
    review.add_argument("--port", type=int, default=18120)
    review.set_defaults(handler=_review)

    purge = commands.add_parser("purge", help="delete expired images")
    purge.add_argument("--db", required=True, type=Path)
    purge.set_defaults(handler=_purge)

    labels = commands.add_parser("import-labels", help="import human labels from JSONL")
    labels.add_argument("--db", required=True, type=Path)
    labels.add_argument("--recipe", required=True, type=Path)
    labels.add_argument("labels", type=Path)
    labels.set_defaults(handler=_import_labels)
    return parser


def _validate(args: argparse.Namespace) -> int:
    failed = False
    for path in args.recipes:
        try:
            loaded = recipe_module.load(path)
            result = {"recipe": path.name, "ok": True, "id": loaded.id}
            _emit({**result, "fingerprint": loaded.fingerprint})
        except StatesureError as exc:
            failed = True
            _emit({"recipe": path.name, "ok": False, "error": exc.code})
    return 1 if failed else 0


def _replay(args: argparse.Namespace) -> int:
    recipes = [recipe_module.load(path) for path in args.recipe]
    configs = {item.fingerprint: item.confirmation for item in recipes}
    observations = [Observation.from_dict(row) for row in _read_jsonl(args.observations)]
    result = replay(observations, configs)
    for event in result.events:
        _emit(event.to_dict())
    if args.states:
        states = {
            "|".join(key): {
                "confirmed_value": state.confirmed_value,
                "confirmed_since": _iso(state.confirmed_since),
                "confirmed_last_seen_at": _iso(state.confirmed_last_seen_at),
                "candidate_value": state.candidate_value,
                "candidate_samples": state.candidate_samples,
                "last_sample_at": _iso(state.last_sample_at),
            }
            for key, state in result.states.items()
        }
        args.states.write_text(json.dumps(states, indent=2, sort_keys=True) + "\n", "utf-8")
    return 0


def _report(args: argparse.Namespace) -> int:
    loaded = recipe_module.load(args.recipe)
    store = _open_store(args.db)
    tz = _zone(args.tz) if args.tz else None
    report = build_report(loaded, store.records(loaded.id), tz=tz, judge_id=args.judge)
    _emit(report.to_dict(), indent=2)
    return 0


def _card(args: argparse.Namespace) -> int:
    loaded = recipe_module.load(args.recipe)
    store = _open_store(args.db)
    records = store.records(loaded.id)
    installs = sorted({r.install_fingerprint for r in records})
    install = args.install
    if install is None:
        if len(installs) != 1:
            raise InputError("install_required")
        install = installs[0]
    elif install not in installs:
        raise InputError("unknown_install")
    tz = _zone(args.tz)
    selected = [r for r in records if r.install_fingerprint == install]
    days = len({r.captured_at.astimezone(tz).date() for r in selected})
    report = build_report(loaded, records, tz=tz, judge_id=args.judge)
    card = build_card(
        loaded,
        report,
        install_fingerprint=install,
        judge_id=args.judge,
        judge_kind=args.judge_kind,
        model=args.model,
        evaluated_days=days,
        camera_kind=args.camera_kind,
    )
    _emit(card, indent=2)
    return 0


def _run_once(args: argparse.Namespace) -> int:
    from .pipeline import run_lock, run_once

    config = load_install(args.config)
    names = [args.installation] if args.installation else sorted(config.installations)
    for name in names:
        if name not in config.installations:
            raise InputError("unknown_installation")
    for name in names:
        installation = config.installations[name]
        judge = config.judges[installation.judge]
        if not judge.is_local:
            _emit({"warning": "judge_not_local", "installation": name}, sys.stderr)
        if not is_local_url(installation.source.url):
            _emit({"warning": "source_not_local", "installation": name}, sys.stderr)
    with run_lock(config.store) as acquired:
        if not acquired:
            _emit({"status": "busy"})
            return 0
        for name in names:
            _emit(run_once(config, name, now=datetime.now(config.timezone)).to_dict())
    return 0


def _review(args: argparse.Namespace) -> int:
    from .review import ReviewApp, serve

    config = load_install(args.config)
    try:
        server = serve(ReviewApp(LabelStore(config.store), config.recipes_dir), port=args.port)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            # Usually an earlier review page is still running.
            raise InputError("port_in_use") from None
        raise
    _emit({"review": f"http://127.0.0.1:{server.server_address[1]}/"}, sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _purge(args: argparse.Namespace) -> int:
    store = _open_store(args.db)
    _emit({"purged": store.purge_expired(datetime.now(UTC))})
    return 0


def _import_labels(args: argparse.Namespace) -> int:
    loaded = recipe_module.load(args.recipe)
    store = _open_store(args.db)
    keys = {
        "sample_id",
        "signal",
        "truth",
        "assessable",
        "reviewer_confidence",
        "labeled_at",
        "label_source",
    }
    imported = 0
    for row in _read_jsonl(args.labels):
        if not isinstance(row, dict) or set(row) != keys:
            raise InputError("invalid_label_fields")
        if type(row["assessable"]) is not bool:
            raise InputError("invalid_label_fields")
        store.add_label(
            loaded,
            row["sample_id"],
            row["signal"],
            row["truth"],
            assessable=row["assessable"],
            reviewer_confidence=row["reviewer_confidence"],
            labeled_at=datetime.fromisoformat(row["labeled_at"]),
            label_source=row["label_source"],
        )
        imported += 1
    _emit({"imported": imported})
    return 0


def _open_store(path: Path) -> LabelStore:
    if not path.exists():
        raise InputError("store_not_found")
    return LabelStore(path)


def _read_jsonl(path: Path) -> Iterator[Any]:
    with path.open("rb") as handle:
        for count, line in enumerate(iter(lambda: handle.readline(MAX_LINE_BYTES + 1), b"")):
            if count >= MAX_ROWS:
                raise InputError("too_many_rows")
            if len(line) > MAX_LINE_BYTES:
                raise InputError("line_too_long")
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise InputError("invalid_jsonl") from None


def _zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        raise InputError("unknown_time_zone") from None


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _emit(value: Any, stream: TextIO | None = None, indent: int | None = None) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=indent, allow_nan=False, default=str)
    print(text, file=stream or sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
