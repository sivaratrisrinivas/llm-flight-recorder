from __future__ import annotations

import json
from pathlib import Path

from pytest import CaptureFixture

from llmfr.cli import run
from llmfr.core.schema import dumps_json
from llmfr.core.version import SCHEMA_VERSION, __version__
from tests.factories import make_trace


def test_cli_version(capsys: CaptureFixture[str]) -> None:
    assert run(["version"]) == 0
    out = capsys.readouterr().out
    assert __version__ in out
    assert SCHEMA_VERSION in out


def test_cli_validate_and_topk(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    path = tmp_path / "trace.json"
    path.write_text(dumps_json(make_trace()), encoding="utf-8")
    assert run(["validate", str(path)]) == 0
    validate_out = capsys.readouterr().out
    assert "ok " in validate_out
    assert "events=2" in validate_out

    assert run(["topk", str(path)]) == 0
    topk_out = capsys.readouterr().out
    assert "top-k" in topk_out
    assert "step 0" in topk_out


def test_cli_validate_missing_file(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    missing = tmp_path / "nope.json"
    assert run(["validate", str(missing)]) == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "Traceback" not in err


def test_cli_validate_bad_json(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not-json", encoding="utf-8")
    assert run(["validate", str(path)]) == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "Traceback" not in err


def test_cli_validate_invalid_trace(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    path = tmp_path / "invalid.json"
    payload = json.loads(dumps_json(make_trace()))
    payload["events"][0]["step"] = 9
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert run(["validate", str(path)]) == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "Traceback" not in err


def test_cli_topk_unsupported_schema(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    path = tmp_path / "old.json"
    payload = json.loads(dumps_json(make_trace()))
    payload["schema_version"] = "0.0.1"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert run(["topk", str(path)]) == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "schema_version" in err
    assert "Traceback" not in err
