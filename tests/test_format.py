from __future__ import annotations

from llmfr.core.format import (
    format_event_topk,
    format_inspect_overview,
    format_inspect_step,
    format_trace_topk,
)
from tests.factories import make_trace


def test_format_is_human_readable_and_ranked() -> None:
    trace = make_trace(k=3)
    text = format_trace_topk(trace)
    assert "logits top-k (k=3)" in text
    assert "step 0  sampled 'Hello'" in text
    assert "1. 'Hello'" in text
    assert "logit=" in text
    assert "prob=" in text
    assert "id=" in text
    assert "full_history=" in text
    assert "model_visible=" in text
    assert "tensor" not in text
    assert "vocab" not in text
    assert format_event_topk(trace.events[0]).startswith("step 0")


def test_format_unavailable_logits() -> None:
    text = format_trace_topk(make_trace(logits_mode="none"))
    assert "logits unavailable" in text
    assert "refusing to invent" in text
    assert "step 0" not in text


def test_format_marks_truncated_context() -> None:
    text = format_event_topk(make_trace(visible_limit=2).events[1])
    assert "(truncated)" in text


def test_inspect_overview_lists_sampled_tokens() -> None:
    text = format_inspect_overview(make_trace())
    assert "sampled tokens:" in text
    assert "0: 'Hello' (id=101)" in text
    assert "--step N" in text
    assert "events 2" in text


def test_inspect_step_shows_history_vs_visible_and_topk() -> None:
    trace = make_trace(visible_limit=2)
    text = format_inspect_step(trace, 1)
    assert "step 1 of 2" in text
    assert "sampled token" in text
    assert "' world' (id=102)" in text
    assert "full_history" in text
    assert "model_visible_context" in text
    assert "truncated=true" in text
    assert "left-truncated suffix of full_history" in text
    assert "top-k" in text
    assert "id=102" in text


def test_inspect_step_unavailable_logits_does_not_invent() -> None:
    text = format_inspect_step(make_trace(logits_mode="none"), 0)
    assert "logits unavailable" in text
    assert "no invented logits" in text


def test_inspect_step_empty_topk_is_honest() -> None:
    trace = make_trace(k=3)
    events = [event.model_copy(update={"top_k": []}) for event in trace.events]
    stripped = trace.model_copy(update={"events": events})
    text = format_inspect_step(stripped, 0)
    assert "(no top-k logits)" in text
    assert "invent" not in text


def test_inspect_overview_openai_does_not_claim_raw_logits() -> None:
    from tests.openai_fakes import make_openai_trace

    text = format_inspect_overview(make_openai_trace())
    assert "openai top_logprobs are not full-vocab raw logits" in text
    assert "replay is not bit-identical" in text
    assert "gpt-4o-mini" in text
