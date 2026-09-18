from __future__ import annotations

from llmfr.core.format import format_event_topk, format_trace_topk
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
