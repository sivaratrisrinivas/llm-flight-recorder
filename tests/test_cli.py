from __future__ import annotations

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
