from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest import CaptureFixture

from llmfr.adapters.huggingface import HuggingFaceCausalLMAdapter, HuggingFaceExtraMissingError
from llmfr.cli import build_parser, run
from llmfr.core.schema import LogitsCapture, dumps_json
from llmfr.core.version import SCHEMA_VERSION, __version__
from llmfr.storage import TraceStore
from tests.factories import make_trace
from tests.fakes import FakeCausalLMAdapter


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


def test_cli_record_is_listed() -> None:
    help_text = build_parser().format_help()
    assert "record" in help_text
    assert "compare" in help_text
    assert "Compare commands are not available yet." not in help_text


def test_cli_record_prints_trace_id(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = FakeCausalLMAdapter(prompt_ids=[1, 2])
    monkeypatch.setattr("llmfr.cli._build_hf_adapter", lambda **_kwargs: adapter)
    code = run(["record", "hello", "--store", str(tmp_path), "--max-new-tokens", "2", "--greedy"])
    assert code == 0
    out = capsys.readouterr().out.strip()
    loaded = TraceStore(tmp_path).get(out)
    assert len(loaded.events) == 2
    assert loaded.run_metadata.source == "cli"
    assert loaded.run_metadata.prompt == "hello"


def test_cli_record_temperature_zero_stores_do_sample_false(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = FakeCausalLMAdapter(prompt_ids=[1, 2])
    monkeypatch.setattr("llmfr.cli._build_hf_adapter", lambda **_kwargs: adapter)
    code = run(
        [
            "record",
            "hello",
            "--store",
            str(tmp_path),
            "--max-new-tokens",
            "1",
            "--temperature",
            "0",
        ]
    )
    assert code == 0
    loaded = TraceStore(tmp_path).get(capsys.readouterr().out.strip())
    assert loaded.generation_config.do_sample is False
    assert loaded.generation_config.temperature == 0.0


def test_cli_record_validation_error(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "llmfr.cli._build_hf_adapter",
        lambda **_kwargs: FakeCausalLMAdapter(prompt_ids=[1]),
    )

    def _invalid_trace(*_args: object, **_kwargs: object) -> None:
        LogitsCapture(mode="topk")

    monkeypatch.setattr("llmfr.cli.record_generation", _invalid_trace)
    assert run(["record", "hello", "--store", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "Traceback" not in err


def test_cli_record_missing_hf_extra(
    monkeypatch: pytest.MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    def _boom(self: HuggingFaceCausalLMAdapter, *args: object, **kwargs: object) -> None:
        raise HuggingFaceExtraMissingError()

    monkeypatch.setattr(HuggingFaceCausalLMAdapter, "__init__", _boom)
    assert run(["record", "hello"]) == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "download.pytorch.org/whl/cpu" in err
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
