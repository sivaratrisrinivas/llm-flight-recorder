from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from llmfr.migrate import UnsupportedSchemaVersionError, migrate_payload
from llmfr.schema import Event, LogitsCapture, TopKCandidate, Trace, dumps_json, loads_json
from llmfr.version import MAX_TOP_K, SCHEMA_VERSION
from tests.factories import make_trace


def test_schema_version_constant_is_v1() -> None:
    assert SCHEMA_VERSION == "1.0.0"
    trace = make_trace()
    assert trace.schema_version == SCHEMA_VERSION


def test_unknown_schema_version_is_rejected() -> None:
    payload = make_trace().model_dump(mode="json")
    payload["schema_version"] = "2.0.0"
    with pytest.raises(UnsupportedSchemaVersionError, match="v1 only"):
        migrate_payload(payload)
    with pytest.raises(UnsupportedSchemaVersionError):
        loads_json(json.dumps(payload))


def test_migrate_stub_is_identity_for_v1() -> None:
    payload = make_trace().model_dump(mode="json")
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
    with pytest.raises(ValidationError):
        Event(step=0, token_id=1, token="a", top_k=huge)


def test_topk_json_does_not_embed_full_vocab() -> None:
    vocab_size = 32_000
    trace = make_trace(k=5)
    blob = dumps_json(trace)
    assert blob.count('"rank"') == 10  # 2 events * 5
    assert f"tok{vocab_size}" not in blob
    assert "full_vocab" not in blob
    assert len(blob.encode("utf-8")) < 8_000


def test_hosted_api_logits_none_is_honest() -> None:
    trace = make_trace(logits_mode="none")
    assert trace.logits.mode == "none"
    assert "invent" in (trace.logits.unavailable_reason or "")
    assert all(not event.top_k for event in trace.events)


def test_none_mode_cannot_carry_topk() -> None:
    trace = make_trace()
    with pytest.raises(ValidationError):
        Trace(
            **{
                **trace.model_dump(mode="json"),
                "logits": {
                    "mode": "none",
                    "unavailable_reason": "hosted API",
                },
            }
        )


def test_noncontiguous_steps_rejected() -> None:
    trace = make_trace()
    payload = trace.model_dump(mode="json")
    payload["events"][1]["step"] = 5
    with pytest.raises(ValidationError):
        Trace.model_validate(payload)


def test_logits_topk_requires_k() -> None:
    with pytest.raises(ValidationError):
        LogitsCapture(mode="topk")
