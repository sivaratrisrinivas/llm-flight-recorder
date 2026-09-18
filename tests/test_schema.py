from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from llmfr.core.migrate import UnsupportedSchemaVersionError, migrate_payload
from llmfr.core.schema import (
    Event,
    LogitsCapture,
    TokenContext,
    TopKCandidate,
    Trace,
    dumps_json,
    loads_json,
)
from llmfr.core.version import MAX_TOP_K, SCHEMA_VERSION
from tests.factories import make_trace


def test_schema_version_constant_is_v1() -> None:
    assert SCHEMA_VERSION == "1.0.0"
    trace = make_trace()
    assert trace.schema_version == SCHEMA_VERSION


def test_required_trace_and_event_fields() -> None:
    payload = json.loads(dumps_json(make_trace()))
    assert payload.keys() >= {
        "schema_version",
        "run_metadata",
        "environment",
        "model_config",
        "generation_config",
        "events",
    }
    event = payload["events"][0]
    assert event.keys() >= {
        "sampled_token_id",
        "sampled_token",
        "sampled_logit",
        "sampled_prob",
        "sampled_logprob",
        "sampled_rank",
        "top_k",
        "full_history",
        "model_visible_context",
    }
    assert event["full_history"]["token_ids"]
    assert event["model_visible_context"]["token_ids"]
    assert event["top_k"][0].keys() >= {"rank", "token_id", "token", "logit", "prob", "logprob"}


def test_unknown_schema_version_is_rejected() -> None:
    payload = json.loads(dumps_json(make_trace()))
    payload["schema_version"] = "2.0.0"
    with pytest.raises(UnsupportedSchemaVersionError, match="v1 only"):
        migrate_payload(payload)
    with pytest.raises(UnsupportedSchemaVersionError):
        loads_json(json.dumps(payload))


def test_migrate_stub_is_identity_for_v1() -> None:
    payload = json.loads(dumps_json(make_trace()))
    assert migrate_payload(payload) is payload


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        TopKCandidate(rank=1, token_id=1, token="a", logit=0.0, extra="nope")  # type: ignore[call-arg]


def test_full_vocab_size_is_rejected() -> None:
    huge = [
        TopKCandidate(rank=i, token_id=i, token=f"tok{i}", logit=0.0)
        for i in range(1, MAX_TOP_K + 1)
    ]
    huge.append(TopKCandidate(rank=MAX_TOP_K, token_id=999, token="overflow", logit=0.0))
    ctx = TokenContext(token_ids=[1], text="a")
    with pytest.raises(ValidationError):
        Event(
            step=0,
            sampled_token_id=1,
            sampled_token="a",
            top_k=huge,
            full_history=ctx,
            model_visible_context=ctx,
        )


def test_topk_json_does_not_embed_full_vocab() -> None:
    vocab_size = 32_000
    trace = make_trace(k=5)
    blob = dumps_json(trace)
    assert blob.count('"rank"') == 10  # 2 events * 5
    assert f"tok{vocab_size}" not in blob
    assert "full_vocab" not in blob
    assert len(blob.encode("utf-8")) < 20_000


def test_hosted_api_logits_none_is_honest() -> None:
    trace = make_trace(logits_mode="none")
    assert trace.run_metadata.logits.mode == "none"
    assert "invent" in (trace.run_metadata.logits.unavailable_reason or "")
    assert all(not event.top_k for event in trace.events)


def test_none_mode_cannot_carry_topk() -> None:
    payload = json.loads(dumps_json(make_trace()))
    payload["run_metadata"]["logits"] = {
        "mode": "none",
        "unavailable_reason": "hosted API",
    }
    with pytest.raises(ValidationError):
        Trace.model_validate(payload)


def test_noncontiguous_steps_rejected() -> None:
    payload = json.loads(dumps_json(make_trace()))
    payload["events"][1]["step"] = 5
    with pytest.raises(ValidationError):
        Trace.model_validate(payload)


def test_logits_topk_requires_k() -> None:
    with pytest.raises(ValidationError):
        LogitsCapture(mode="topk")


def test_truncated_visible_context_is_suffix() -> None:
    trace = make_trace(visible_limit=2)
    event = trace.events[1]
    assert event.model_visible_context.truncated is True
    assert event.full_history.token_ids[-2:] == event.model_visible_context.token_ids
    assert event.full_history.token_ids != event.model_visible_context.token_ids


def test_untruncated_mismatch_is_rejected() -> None:
    payload = json.loads(dumps_json(make_trace()))
    payload["events"][0]["model_visible_context"]["token_ids"] = [99]
    payload["events"][0]["model_visible_context"]["truncated"] = False
    with pytest.raises(ValidationError, match="must equal full_history"):
        Trace.model_validate(payload)
