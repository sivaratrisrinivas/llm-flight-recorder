"""Replay tests with a fake adapter (no Hugging Face)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest import CaptureFixture

from llmfr.cli import build_parser, run
from llmfr.core.schema import GenerationConfig, LogitsCapture
from llmfr.record import record_generation
from llmfr.replay import (
    BIT_IDENTICAL_CAVEAT,
    build_adapter_for_trace,
    infer_max_visible_tokens,
    replay_trace,
)
from llmfr.storage import TraceStore
from tests.factories import make_trace
from tests.fakes import FakeCausalLMAdapter


def _peaked_adapter(*, supports_replay: bool = True) -> FakeCausalLMAdapter:
    return FakeCausalLMAdapter(
        prompt_ids=[1],
        logits=(0.0, 0.0, 9.0, 0.0),
        logits_for_prefix={
            (1,): (0.0, 0.0, 9.0, 0.0),
            (1, 2): (9.0, 0.0, 0.0, 0.0),
        },
        supports_replay=supports_replay,
    )


def test_replay_reproduces_greedy_tokens_and_logits() -> None:
    adapter = _peaked_adapter()
    trace = record_generation(
        adapter,
        "x",
        generation=GenerationConfig(max_new_tokens=2, do_sample=False, seed=3),
    )
    result = replay_trace(trace, adapter=_peaked_adapter())
    assert result.status == "reproduced"
    assert result.token_ids_matched is True
    assert result.logits_bit_identical is True
    assert result.bit_identical is True
    assert result.matched_steps == 2
    assert result.total_steps == 2
    assert result.first_unmatched_step is None
    assert BIT_IDENTICAL_CAVEAT in result.notes
    assert [row.replayed_token_id for row in result.steps] == [2, 0]


def test_replay_uses_stored_seed_for_sampling() -> None:
    logits = (1.0, 1.0, 1.0, 1.0)
    adapter = FakeCausalLMAdapter(prompt_ids=[0], logits=logits)
    generation = GenerationConfig(
        max_new_tokens=4,
        do_sample=True,
        temperature=1.0,
        seed=42,
    )
    trace = record_generation(adapter, "x", generation=generation)
    same = replay_trace(
        trace,
        adapter=FakeCausalLMAdapter(prompt_ids=[0], logits=logits),
    )
    other_trace = record_generation(
        FakeCausalLMAdapter(prompt_ids=[0], logits=logits),
        "x",
        generation=generation.model_copy(update={"seed": 43}),
    )
    assert same.status == "reproduced"
    assert same.token_ids_matched is True
    assert [event.sampled_token_id for event in trace.events] != [
        event.sampled_token_id for event in other_trace.events
    ]


def test_replay_reports_partial_when_later_token_differs() -> None:
    recorded = record_generation(
        _peaked_adapter(),
        "x",
        generation=GenerationConfig(max_new_tokens=2, do_sample=False),
    )
    drifted = FakeCausalLMAdapter(
        prompt_ids=[1],
        logits=(0.0, 0.0, 9.0, 0.0),
        logits_for_prefix={
            (1,): (0.0, 0.0, 9.0, 0.0),
            (1, 2): (0.0, 9.0, 0.0, 0.0),
        },
    )
    result = replay_trace(recorded, adapter=drifted)
    assert result.status == "partially_reproduced"
    assert result.token_ids_matched is False
    assert result.bit_identical is False
    assert result.matched_steps == 1
    assert result.first_unmatched_step == 1
    assert result.steps[0].token_matched is True
    assert result.steps[1].token_matched is False


def test_token_match_without_logit_match_is_not_bit_identical() -> None:
    recorded = record_generation(
        _peaked_adapter(),
        "x",
        generation=GenerationConfig(max_new_tokens=2, do_sample=False),
    )
    same_argmax = FakeCausalLMAdapter(
        prompt_ids=[1],
        logits=(0.0, 0.0, 8.5, 0.0),
        logits_for_prefix={
            (1,): (0.0, 0.0, 8.5, 0.0),
            (1, 2): (8.5, 0.0, 0.0, 0.0),
        },
    )
    result = replay_trace(recorded, adapter=same_argmax)
    assert result.status == "reproduced"
    assert result.token_ids_matched is True
    assert result.logits_bit_identical is False
    assert result.bit_identical is False
    assert any("not a full numeric replay" in note for note in result.notes)


def test_prob_only_topk_is_not_bit_identical() -> None:
    recorded = record_generation(
        _peaked_adapter(),
        "x",
        generation=GenerationConfig(max_new_tokens=2, do_sample=False),
    )
    events = []
    for event in recorded.events:
        top_k = [
            candidate.model_copy(
                update={
                    "logit": None,
                    "logprob": None,
                    "prob": 0.5 if candidate.prob is None else candidate.prob,
                }
            )
            for candidate in event.top_k
        ]
        events.append(
            event.model_copy(
                update={
                    "sampled_logit": None,
                    "sampled_logprob": None,
                    "sampled_prob": 0.5,
                    "top_k": top_k,
                }
            )
        )
    stripped = recorded.model_copy(update={"events": events})
    result = replay_trace(stripped, adapter=_peaked_adapter())
    assert result.status == "reproduced"
    assert result.token_ids_matched is True
    assert all(row.token_matched for row in result.steps)
    assert result.logits_bit_identical is False
    assert result.bit_identical is False
    assert any("not a full numeric replay" in note for note in result.notes)


def test_logits_unavailable_trace_is_not_replayable() -> None:
    trace = make_trace(logits_mode="none")
    result = replay_trace(trace, adapter=FakeCausalLMAdapter(prompt_ids=[1, 2, 3]))
    assert result.status == "not_replayable"
    assert result.token_ids_matched is False
    assert result.bit_identical is False
    assert result.reason is not None
    assert "logits.mode=none" in result.reason


def test_sampled_trace_without_seed_is_not_replayable() -> None:
    adapter = FakeCausalLMAdapter(prompt_ids=[1], logits=(1.0, 1.0, 1.0, 1.0))
    trace = record_generation(
        adapter,
        "x",
        generation=GenerationConfig(max_new_tokens=2, do_sample=True, temperature=1.0, seed=7),
    )
    stripped = trace.model_copy(
        update={"generation_config": trace.generation_config.model_copy(update={"seed": None})}
    )
    result = replay_trace(stripped, adapter=FakeCausalLMAdapter(prompt_ids=[1]))
    assert result.status == "not_replayable"
    assert result.reason is not None
    assert "no seed" in result.reason


def test_missing_hub_revision_is_not_replayable() -> None:
    trace = make_trace()
    huggingface = trace.model_copy(
        update={
            "model": trace.model.model_copy(update={"provider": "huggingface", "revision": None})
        }
    )
    result = replay_trace(huggingface)
    assert result.status == "not_replayable"
    assert result.reason is not None
    assert "revision" in result.reason


def test_moving_hub_revision_does_not_download(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("from_pretrained must not run for a moving Hub ref")

    monkeypatch.setattr("llmfr.replay.replay.HuggingFaceCausalLMAdapter", _boom)
    huggingface = make_trace().model_copy(
        update={
            "model": make_trace().model.model_copy(
                update={"provider": "huggingface", "revision": "main"}
            )
        }
    )
    result = replay_trace(huggingface)
    assert result.status == "not_replayable"
    assert result.reason is not None
    assert "40-character" in result.reason
    assert "moving" in result.reason
    with pytest.raises(ValueError, match="moving"):
        build_adapter_for_trace(huggingface)


def test_adapter_without_replay_support_is_refused() -> None:
    adapter = _peaked_adapter()
    trace = record_generation(
        adapter,
        "x",
        generation=GenerationConfig(max_new_tokens=1, do_sample=False),
    )
    result = replay_trace(trace, adapter=_peaked_adapter(supports_replay=False))
    assert result.status == "not_replayable"
    assert result.reason is not None
    assert "does not support replay" in result.reason


def test_infer_max_visible_tokens_from_truncated_events() -> None:
    adapter = FakeCausalLMAdapter(prompt_ids=[0, 1, 2, 3, 4], max_visible_tokens=2)
    trace = record_generation(
        adapter,
        "long",
        generation=GenerationConfig(max_new_tokens=2, do_sample=False),
    )
    assert infer_max_visible_tokens(trace) == 2
    unclipped = record_generation(
        FakeCausalLMAdapter(prompt_ids=[1, 2], max_visible_tokens=16),
        "hi",
        generation=GenerationConfig(max_new_tokens=1, do_sample=False),
    )
    assert infer_max_visible_tokens(unclipped) is None


def test_cli_replay_is_listed() -> None:
    help_text = build_parser().format_help()
    assert "replay" in help_text
    assert "Compare commands are not available yet." in help_text
    assert "Replay and compare commands are not available yet." not in help_text


def test_cli_replay_prints_structured_result(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _peaked_adapter()
    store = TraceStore(tmp_path)
    trace = record_generation(
        adapter,
        "x",
        generation=GenerationConfig(max_new_tokens=2, do_sample=False, seed=3),
        store=store,
    )

    monkeypatch.setattr(
        "llmfr.cli._replay",
        lambda _trace: replay_trace(_trace, adapter=_peaked_adapter()),
    )
    code = run(["replay", str(trace.run_metadata.trace_id), "--store", str(tmp_path)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "reproduced"
    assert payload["token_ids_matched"] is True
    assert payload["logits_bit_identical"] is True
    assert payload["bit_identical"] is True
    assert payload["matched_steps"] == 2
    assert BIT_IDENTICAL_CAVEAT in payload["notes"]


def test_cli_replay_partial_is_nonzero_exit(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded = record_generation(
        _peaked_adapter(),
        "x",
        generation=GenerationConfig(max_new_tokens=2, do_sample=False),
        store=TraceStore(tmp_path),
    )
    drifted = FakeCausalLMAdapter(
        prompt_ids=[1],
        logits=(0.0, 0.0, 9.0, 0.0),
        logits_for_prefix={
            (1,): (0.0, 0.0, 9.0, 0.0),
            (1, 2): (0.0, 9.0, 0.0, 0.0),
        },
    )
    monkeypatch.setattr(
        "llmfr.cli._replay",
        lambda _trace: replay_trace(_trace, adapter=drifted),
    )
    code = run(["replay", str(recorded.run_metadata.trace_id), "--store", str(tmp_path)])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "partially_reproduced"
    assert payload["first_unmatched_step"] == 1


def test_cli_replay_unknown_trace_id(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    TraceStore(tmp_path)
    code = run(["replay", "11111111-1111-4111-8111-111111111111", "--store", str(tmp_path)])
    assert code == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "unknown trace_id" in err
    assert "Traceback" not in err


def test_replay_forward_failure_is_not_replayable() -> None:
    recorded = record_generation(
        _peaked_adapter(),
        "x",
        generation=GenerationConfig(max_new_tokens=2, do_sample=False),
    )
    boom = _peaked_adapter()

    def _fail(_context: object) -> None:
        raise RuntimeError("forward failed")

    boom.next_token_logits = _fail  # type: ignore[method-assign]
    result = replay_trace(recorded, adapter=boom)
    assert result.status == "not_replayable"
    assert result.token_ids_matched is False
    assert result.bit_identical is False
    assert result.reason is not None
    assert "forward failed" in result.reason
    assert any("replay stopped" in note for note in result.notes)


def test_cli_replay_runtime_failure_prints_json(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded = record_generation(
        _peaked_adapter(),
        "x",
        generation=GenerationConfig(max_new_tokens=2, do_sample=False),
        store=TraceStore(tmp_path),
    )

    def _boom(_trace: object) -> None:
        raise RuntimeError("forward failed")

    monkeypatch.setattr("llmfr.cli._replay", _boom)
    code = run(["replay", str(recorded.run_metadata.trace_id), "--store", str(tmp_path)])
    assert code == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    payload = json.loads(captured.out)
    assert payload["status"] == "not_replayable"
    assert payload["token_ids_matched"] is False
    assert payload["bit_identical"] is False
    assert "forward failed" in payload["reason"]
    assert any("replay failed" in note for note in payload["notes"])


def test_cli_replay_not_replayable_validation_error(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    store = TraceStore(tmp_path)
    store.put(make_trace(logits_mode="none"))

    def _invalid(_trace: object) -> None:
        LogitsCapture(mode="topk")

    monkeypatch.setattr("llmfr.cli._replay", _invalid)
    assert run(["replay", str(make_trace().run_metadata.trace_id), "--store", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    payload = json.loads(captured.out)
    assert payload["status"] == "not_replayable"
    assert payload["token_ids_matched"] is False
    assert payload["reason"]
