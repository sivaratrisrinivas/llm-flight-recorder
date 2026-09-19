"""Recorder tests for the OpenAI hosted path. Mocked responses only."""

from __future__ import annotations

from pathlib import Path

from llmfr.adapters.openai import OpenAIChatAdapter
from llmfr.core.schema import GenerationConfig
from llmfr.privacy import REDACTED, redact_trace
from llmfr.record import record_generation
from llmfr.storage import TraceStore
from tests.openai_fakes import (
    FakeEncoding,
    FakeOpenAIClient,
    hello_logprob_tokens,
    make_chat_response,
)


def _adapter(client: FakeOpenAIClient) -> OpenAIChatAdapter:
    return OpenAIChatAdapter("gpt-4o-mini", client=client, encoding=FakeEncoding())


def test_record_openai_stores_top_logprobs_without_logits(tmp_path: Path) -> None:
    client = FakeOpenAIClient([make_chat_response(hello_logprob_tokens())])
    store = TraceStore(tmp_path)
    trace = record_generation(
        _adapter(client),
        "Hi",
        generation=GenerationConfig(max_new_tokens=2, do_sample=False, temperature=0.0),
        store=store,
        capture_k=3,
        source="test",
    )
    assert trace.model.provider == "openai"
    assert trace.run_metadata.logits.mode == "topk"
    assert trace.run_metadata.logits.k == 3
    assert [event.sampled_token for event in trace.events] == ["He", "llo"]
    assert all(event.sampled_logit is None for event in trace.events)
    assert all(event.sampled_logprob is not None for event in trace.events)
    for event in trace.events:
        assert event.top_k
        assert all(candidate.logit is None for candidate in event.top_k)
        assert all(candidate.logprob is not None for candidate in event.top_k)
    assert trace.events[0].full_history.token_ids == trace.run_metadata.prompt_token_ids
    assert trace.events[0].model_visible_context.truncated is False
    loaded = store.get(str(trace.run_metadata.trace_id))
    assert loaded.run_metadata.logits.mode == "topk"
    assert loaded.events[0].top_k[0].logit is None


def test_record_openai_without_logprobs_is_mode_none() -> None:
    client = FakeOpenAIClient(
        [make_chat_response(hello_logprob_tokens(), output_text="Hello", include_logprobs=False)]
    )
    trace = record_generation(
        _adapter(client),
        "Hi",
        generation=GenerationConfig(max_new_tokens=2, do_sample=False),
    )
    assert trace.run_metadata.logits.mode == "none"
    assert trace.run_metadata.logits.unavailable_reason is not None
    assert "invent" in trace.run_metadata.logits.unavailable_reason
    assert all(not event.top_k for event in trace.events)
    assert all(event.sampled_logit is None for event in trace.events)
    assert trace.events


def test_record_openai_redact_and_no_persist(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    unredacted = record_generation(
        _adapter(FakeOpenAIClient([make_chat_response(hello_logprob_tokens())])),
        "secret prompt",
        generation=GenerationConfig(max_new_tokens=2, do_sample=False),
        persist=False,
        store=store,
    )
    assert store.list() == []
    assert unredacted.run_metadata.prompt == "secret prompt"

    redacted = record_generation(
        _adapter(FakeOpenAIClient([make_chat_response(hello_logprob_tokens())])),
        "secret prompt",
        generation=GenerationConfig(max_new_tokens=2, do_sample=False),
        store=store,
        redact=redact_trace,
    )
    loaded = store.get(str(redacted.run_metadata.trace_id))
    assert loaded.run_metadata.prompt == REDACTED
    assert loaded.events[0].sampled_token == REDACTED
    assert all(candidate.token == REDACTED for event in loaded.events for candidate in event.top_k)
    assert loaded.events[0].sampled_logprob == unredacted.events[0].sampled_logprob
    assert loaded.events[0].top_k[0].logprob == unredacted.events[0].top_k[0].logprob
    assert loaded.events[0].top_k[0].logit is None
