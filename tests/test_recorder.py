"""Recorder loop tests with a fake adapter (no Hugging Face)."""

from __future__ import annotations

from pathlib import Path

import pytest

from llmfr.core.schema import GenerationConfig
from llmfr.record import RECORDER_PIPELINE, RecordableAdapter, record_generation
from llmfr.storage import TraceStore
from tests.fakes import FakeCausalLMAdapter


def test_recorder_pipeline_constant_matches_acceptance() -> None:
    assert RECORDER_PIPELINE == (
        "full_history",
        "model_visible_context",
        "raw_logits",
        "temperature_logits",
        "probabilities",
        "sample",
        "append",
    )


def test_recorder_walks_prefix_then_logits_then_append() -> None:
    adapter = FakeCausalLMAdapter(
        prompt_ids=[1],
        logits=(0.0, 0.0, 9.0, 0.0),
        logits_for_prefix={
            (1,): (0.0, 0.0, 9.0, 0.0),
            (1, 2): (9.0, 0.0, 0.0, 0.0),
        },
    )
    assert isinstance(adapter, RecordableAdapter)
    trace = record_generation(
        adapter,
        "x",
        generation=GenerationConfig(max_new_tokens=2, do_sample=False, temperature=0.0),
    )
    assert adapter.prefixes == [(1,), (1, 2)]
    assert [event.sampled_token_id for event in trace.events] == [2, 0]
    assert trace.events[0].full_history.token_ids == [1]
    assert trace.events[1].full_history.token_ids == [1, 2]
    assert trace.events[0].full_history.token_ids + [2] == trace.events[1].full_history.token_ids
    assert trace.run_metadata.prompt_token_ids == [1]
    assert trace.run_metadata.output_text == "t2t0"
    assert all(event.top_k[0].logit is not None for event in trace.events)


def test_history_differs_from_visible_context_when_windowed() -> None:
    prompt_ids = [0, 1, 2, 3, 4]
    adapter = FakeCausalLMAdapter(prompt_ids=prompt_ids, max_visible_tokens=2)
    trace = record_generation(
        adapter,
        "long-prompt",
        generation=GenerationConfig(max_new_tokens=2, do_sample=False),
    )
    first = trace.events[0]
    assert first.full_history.token_ids == prompt_ids
    assert first.model_visible_context.token_ids == prompt_ids[-2:]
    assert first.model_visible_context.truncated is True
    assert first.full_history.token_ids != first.model_visible_context.token_ids
    second = trace.events[1]
    expected_history = prompt_ids + [first.sampled_token_id]
    assert second.full_history.token_ids == expected_history
    assert second.model_visible_context.token_ids == expected_history[-2:]
    assert second.model_visible_context.truncated is True


def test_untruncated_visible_equals_full_history() -> None:
    adapter = FakeCausalLMAdapter(prompt_ids=[1, 2], max_visible_tokens=16)
    trace = record_generation(
        adapter,
        "hi",
        generation=GenerationConfig(max_new_tokens=1, do_sample=False),
    )
    event = trace.events[0]
    assert event.model_visible_context.truncated is False
    assert event.model_visible_context.token_ids == event.full_history.token_ids


def test_recorder_persists_via_trace_store(tmp_path: Path) -> None:
    adapter = FakeCausalLMAdapter(prompt_ids=[1, 2])
    store = TraceStore(tmp_path)
    trace = record_generation(
        adapter,
        "hello",
        generation=GenerationConfig(max_new_tokens=2, do_sample=False, seed=3),
        store=store,
        capture_k=3,
        fmt="jsonl",
        source="test",
    )
    loaded = store.get(str(trace.run_metadata.trace_id))
    assert loaded == trace
    assert loaded.run_metadata.trace_id == trace.run_metadata.trace_id
    assert loaded.run_metadata.logits.mode == "topk"
    assert loaded.run_metadata.logits.k == 3
    assert len(loaded.events[0].top_k) == 3
    assert loaded.generation_config.seed == 3
    entry = store.get_index(str(trace.run_metadata.trace_id))
    assert entry.format == "jsonl"
    assert entry.event_count == 2
    assert (tmp_path / "index.sqlite").is_file()


def test_recorder_stores_effective_greedy_config(tmp_path: Path) -> None:
    adapter = FakeCausalLMAdapter(prompt_ids=[1, 2])
    store = TraceStore(tmp_path)
    requested = GenerationConfig(
        max_new_tokens=2,
        do_sample=True,
        temperature=0.0,
        seed=3,
    )
    trace = record_generation(
        adapter,
        "hello",
        generation=requested,
        store=store,
    )
    assert requested.do_sample is True
    assert trace.generation_config.do_sample is False
    assert trace.generation_config.temperature == 0.0
    assert trace.generation_config.seed == 3
    loaded = store.get(str(trace.run_metadata.trace_id))
    assert loaded.generation_config.do_sample is False
    assert loaded.generation_config.temperature == 0.0


def test_seed_refused_when_adapter_lacks_support() -> None:
    adapter = FakeCausalLMAdapter(supports_seed=False)
    with pytest.raises(ValueError, match="does not support seed"):
        record_generation(
            adapter,
            "hi",
            generation=GenerationConfig(max_new_tokens=1, do_sample=True, seed=1, temperature=1.0),
        )


def test_logits_unavailable_adapter_is_refused() -> None:
    adapter = FakeCausalLMAdapter(supports_logits=False)
    with pytest.raises(ValueError, match="refusing to invent"):
        record_generation(
            adapter,
            "hi",
            generation=GenerationConfig(max_new_tokens=1, do_sample=False),
        )


def test_top_p_and_repetition_penalty_are_rejected() -> None:
    adapter = FakeCausalLMAdapter()
    with pytest.raises(ValueError, match="top_p"):
        record_generation(
            adapter,
            "hi",
            generation=GenerationConfig(max_new_tokens=1, top_p=0.9, do_sample=False),
        )
    with pytest.raises(ValueError, match="repetition_penalty"):
        record_generation(
            adapter,
            "hi",
            generation=GenerationConfig(max_new_tokens=1, repetition_penalty=1.1, do_sample=False),
        )


def test_empty_prompt_is_rejected() -> None:
    adapter = FakeCausalLMAdapter(prompt_ids=[])
    with pytest.raises(ValueError, match="no tokens"):
        record_generation(
            adapter,
            "",
            generation=GenerationConfig(max_new_tokens=1, do_sample=False),
        )


def test_stop_sequence_halts_after_match() -> None:
    adapter = FakeCausalLMAdapter(
        prompt_ids=[1],
        logits=(0.0, 0.0, 9.0, 0.0),
    )
    trace = record_generation(
        adapter,
        "x",
        generation=GenerationConfig(
            max_new_tokens=8,
            do_sample=False,
            stop=["t2"],
        ),
    )
    assert len(trace.events) == 1
    assert trace.events[0].sampled_token == "t2"
