"""Privacy: redaction hooks and disable-persist. Local write stays the default."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest import CaptureFixture

from llmfr.cli import run
from llmfr.core.schema import GenerationConfig, Trace
from llmfr.privacy import DEFAULT_REDACT_FIELDS, REDACTED, redact_trace
from llmfr.record import record_generation
from llmfr.storage import TraceStore
from tests.factories import make_trace
from tests.fakes import FakeCausalLMAdapter


def test_redact_trace_replaces_default_text_fields() -> None:
    original = make_trace()
    redacted = redact_trace(original)
    assert redacted.run_metadata.prompt == REDACTED
    assert redacted.run_metadata.output_text == REDACTED
    assert redacted.events[0].full_history.text == REDACTED
    assert redacted.events[0].full_history.token_ids == original.events[0].full_history.token_ids
    assert redacted.events[0].sampled_token_id == original.events[0].sampled_token_id
    assert redacted.events[0].top_k == original.events[0].top_k
    assert original.run_metadata.prompt == "Say hello"


def test_redact_trace_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown redact field"):
        redact_trace(make_trace(), fields=("prompt", "secret_blob"))


def test_redact_trace_can_target_sampled_token_and_tags() -> None:
    redacted = redact_trace(make_trace(), fields=("sampled_token", "tags"))
    assert redacted.run_metadata.prompt == "Say hello"
    assert redacted.events[0].sampled_token == REDACTED
    assert redacted.events[0].sampled_token_id == 101
    assert redacted.run_metadata.tags == {"case": REDACTED}


def test_record_persist_false_skips_store(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    trace = record_generation(
        FakeCausalLMAdapter(prompt_ids=[1, 2]),
        "secret prompt",
        generation=GenerationConfig(max_new_tokens=1, do_sample=False),
        store=store,
        persist=False,
    )
    assert trace.run_metadata.prompt == "secret prompt"
    assert store.list() == []
    assert list(store.traces_dir.glob("*")) == []
    with pytest.raises(KeyError):
        store.get(str(trace.run_metadata.trace_id))


def test_record_redacts_before_persist(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    trace = record_generation(
        FakeCausalLMAdapter(prompt_ids=[1, 2]),
        "secret prompt",
        generation=GenerationConfig(max_new_tokens=1, do_sample=False),
        store=store,
        redact=redact_trace,
    )
    loaded = store.get(str(trace.run_metadata.trace_id))
    assert loaded.run_metadata.prompt == REDACTED
    assert loaded.run_metadata.output_text == REDACTED
    assert loaded.events[0].full_history.text == REDACTED
    assert loaded.events[0].full_history.token_ids == [1, 2]
    assert loaded.run_metadata.prompt_token_ids == [1, 2]


def test_record_custom_redactor_hook(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)

    def _hook(trace: Trace) -> Trace:
        return trace.model_copy(
            update={
                "run_metadata": trace.run_metadata.model_copy(update={"prompt": "cleared"}),
            }
        )

    trace = record_generation(
        FakeCausalLMAdapter(prompt_ids=[1]),
        "secret prompt",
        generation=GenerationConfig(max_new_tokens=1, do_sample=False),
        store=store,
        redact=_hook,
    )
    loaded = store.get(str(trace.run_metadata.trace_id))
    assert loaded.run_metadata.prompt == "cleared"
    assert "prompt" in DEFAULT_REDACT_FIELDS


def test_cli_record_no_persist(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "llmfr.cli._build_hf_adapter",
        lambda **_kwargs: FakeCausalLMAdapter(prompt_ids=[1, 2]),
    )
    code = run(
        [
            "record",
            "secret prompt",
            "--store",
            str(tmp_path),
            "--max-new-tokens",
            "1",
            "--greedy",
            "--no-persist",
        ]
    )
    assert code == 0
    capsys.readouterr()
    assert not (tmp_path / "index.sqlite").exists()
    assert list(tmp_path.glob("**/*")) == [] or not (tmp_path / "traces").exists()


def test_cli_record_redact_before_write(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "llmfr.cli._build_hf_adapter",
        lambda **_kwargs: FakeCausalLMAdapter(prompt_ids=[1, 2]),
    )
    code = run(
        [
            "record",
            "secret prompt",
            "--store",
            str(tmp_path),
            "--max-new-tokens",
            "1",
            "--greedy",
            "--redact",
        ]
    )
    assert code == 0
    trace_id = capsys.readouterr().out.strip()
    loaded = TraceStore(tmp_path).get(trace_id)
    assert loaded.run_metadata.prompt == REDACTED
    assert loaded.run_metadata.output_text == REDACTED
    assert "secret prompt" not in loaded.run_metadata.prompt
