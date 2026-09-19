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
    assert "replay" in help_text
    assert "compare" in help_text
    assert "inspect" in help_text
    assert "Exit codes:" in help_text
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


def test_cli_inspect_step(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    path = tmp_path / "trace.json"
    path.write_text(dumps_json(make_trace(visible_limit=2)), encoding="utf-8")
    assert run(["inspect", str(path), "--step", "1"]) == 0
    out = capsys.readouterr().out
    assert "step 1 of 2" in out
    assert "full_history" in out
    assert "model_visible_context" in out
    assert "sampled token" in out
    assert "top-k" in out
    assert "Traceback" not in out


def test_cli_inspect_overview_and_json(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    path = tmp_path / "trace.json"
    path.write_text(dumps_json(make_trace()), encoding="utf-8")
    assert run(["inspect", str(path)]) == 0
    out = capsys.readouterr().out
    assert "sampled tokens:" in out
    assert "--step N" in out

    assert run(["inspect", str(path), "--step", "0", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["step"] == 0
    assert payload["sampled_token"] == "Hello"
    assert payload["sampled_token_id"] == 101
    assert payload["full_history"]["token_ids"]
    assert payload["model_visible_context"]["token_ids"]
    assert payload["top_k"]
    assert payload["visible_equals_history"] is True


def test_cli_inspect_store_id(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    store = TraceStore(tmp_path)
    trace = make_trace()
    store.put(trace)
    code = run(
        [
            "inspect",
            str(trace.run_metadata.trace_id),
            "--store",
            str(tmp_path),
            "--step",
            "0",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert str(trace.run_metadata.trace_id) in out
    assert "sampled token" in out


def test_cli_inspect_step_out_of_range(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    path = tmp_path / "trace.json"
    path.write_text(dumps_json(make_trace()), encoding="utf-8")
    assert run(["inspect", str(path), "--step", "9"]) == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "out of range" in err
    assert "Traceback" not in err


def test_cli_inspect_unknown_id(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    TraceStore(tmp_path)
    code = run(["inspect", "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "--store", str(tmp_path)])
    assert code == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "unknown trace_id" in err
    assert "Traceback" not in err


def test_cli_usage_error_is_exit_2(capsys: CaptureFixture[str]) -> None:
    assert run(["record"]) == 2
    err = capsys.readouterr().err
    assert "error:" in err
    assert "Traceback" not in err


def test_cli_inspect_help_mentions_step(capsys: CaptureFixture[str]) -> None:
    assert run(["inspect", "--help"]) == 0
    out = capsys.readouterr().out
    assert "--step" in out
    assert "Exit 1" in out


def test_cli_record_replay_compare_help(capsys: CaptureFixture[str]) -> None:
    assert run(["record", "--help"]) == 0
    record_help = capsys.readouterr().out
    assert "--store" in record_help
    assert "--greedy" in record_help
    assert "Exit 0" in record_help

    assert run(["replay", "--help"]) == 0
    replay_help = capsys.readouterr().out
    assert "TRACE_ID" in replay_help or "trace_id" in replay_help
    assert "Exit 0" in replay_help

    assert run(["compare", "--help"]) == 0
    compare_help = capsys.readouterr().out
    assert "--json" in compare_help
    assert "Exit 0" in compare_help


def test_cli_record_twice_then_compare_identical(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "llmfr.cli._build_hf_adapter",
        lambda **_kwargs: FakeCausalLMAdapter(prompt_ids=[1, 2]),
    )
    store = str(tmp_path)
    assert run(["record", "hello", "--store", store, "--max-new-tokens", "2", "--greedy"]) == 0
    id_a = capsys.readouterr().out.strip()
    assert run(["record", "hello", "--store", store, "--max-new-tokens", "2", "--greedy"]) == 0
    id_b = capsys.readouterr().out.strip()
    assert id_a != id_b
    assert run(["inspect", id_a, "--store", store, "--step", "0"]) == 0
    inspect_out = capsys.readouterr().out
    assert "full_history" in inspect_out
    assert "model_visible_context" in inspect_out
    assert "sampled token" in inspect_out
    assert run(["compare", id_a, id_b, "--store", store]) == 0
    compare_out = capsys.readouterr().out
    assert "identical" in compare_out
    assert "FIRST BEHAVIORAL DIVERGENCE" in compare_out
