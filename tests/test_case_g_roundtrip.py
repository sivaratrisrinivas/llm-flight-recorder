"""Case G: round-trip serialize and deserialize for Trace + Event."""

from __future__ import annotations

import json
from pathlib import Path

from llmfr.schema import dumps_json, dumps_jsonl, loads_json, loads_jsonl
from llmfr.store import TraceStore
from llmfr.version import SCHEMA_VERSION
from tests.factories import make_trace


def test_case_g_json_roundtrip() -> None:
    original = make_trace()
    blob = dumps_json(original)
    restored = loads_json(blob)
    assert restored == original
    payload = json.loads(blob)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert restored.model_dump(mode="json") == original.model_dump(mode="json")


def test_case_g_jsonl_roundtrip() -> None:
    original = make_trace()
    blob = dumps_jsonl(original)
    restored = loads_jsonl(blob)
    assert restored == original
    lines = [line for line in blob.splitlines() if line]
    header = json.loads(lines[0])
    assert header["record"] == "header"
    assert header["schema_version"] == SCHEMA_VERSION
    assert "events" not in header
    assert json.loads(lines[1])["record"] == "event"


def test_case_g_store_json_and_jsonl_roundtrip(tmp_path: Path) -> None:
    original = make_trace()
    store = TraceStore(tmp_path)

    json_path = store.put(original, fmt="json")
    jsonl_path = store.put(original, fmt="jsonl")

    assert json_path.exists()
    assert jsonl_path.exists()
    assert (tmp_path / "index.sqlite").exists()

    from_json = store.get(str(original.trace_id))
    # last put was jsonl; get uses the index format
    assert jsonl_path.suffix == ".jsonl"
    assert from_json == original

    store_json_only = TraceStore(tmp_path / "json-only")
    store_json_only.put(original, fmt="json")
    assert store_json_only.get(str(original.trace_id)) == original

    listed = store.list()
    assert len(listed) == 1
    assert listed[0].trace_id == str(original.trace_id)
    assert listed[0].schema_version == SCHEMA_VERSION
    assert listed[0].event_count == 2
    assert listed[0].logits_mode == "topk"
    assert listed[0].format == "jsonl"
