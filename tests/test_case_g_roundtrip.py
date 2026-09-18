"""Case G: round-trip serialize and deserialize for Trace + Event."""

from __future__ import annotations

import json
from pathlib import Path

from llmfr.core.schema import dumps_json, dumps_jsonl, loads_json, loads_jsonl
from llmfr.core.version import SCHEMA_VERSION
from llmfr.storage.store import TraceStore
from tests.factories import TRACE_ID, make_trace


def test_case_g_json_roundtrip() -> None:
    original = make_trace()
    blob = dumps_json(original)
    restored = loads_json(blob)
    assert restored == original
    payload = json.loads(blob)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["run_metadata"]["trace_id"] == str(TRACE_ID)
    assert restored.run_metadata.trace_id == original.run_metadata.trace_id
    assert restored.model_dump(mode="json", by_alias=True) == original.model_dump(
        mode="json", by_alias=True
    )


def test_case_g_jsonl_roundtrip() -> None:
    original = make_trace()
    blob = dumps_jsonl(original)
    restored = loads_jsonl(blob)
    assert restored == original
    assert restored.run_metadata.trace_id == TRACE_ID
    lines = [line for line in blob.splitlines() if line]
    header = json.loads(lines[0])
    assert header["record"] == "header"
    assert header["schema_version"] == SCHEMA_VERSION
    assert "events" not in header
    assert "model_config" in header
    assert json.loads(lines[1])["record"] == "event"
    assert "full_history" in json.loads(lines[1])
    assert "model_visible_context" in json.loads(lines[1])


def test_case_g_store_json_and_jsonl_roundtrip(tmp_path: Path) -> None:
    original = make_trace()
    store = TraceStore(tmp_path)

    json_path = store.put(original, fmt="json")
    jsonl_path = store.put(original, fmt="jsonl", overwrite=True)

    assert not json_path.exists()
    assert jsonl_path.exists()
    assert (tmp_path / "index.sqlite").exists()

    loaded = store.get(str(original.run_metadata.trace_id))
    assert loaded == original
    assert loaded.run_metadata.trace_id == original.run_metadata.trace_id

    store_json_only = TraceStore(tmp_path / "json-only")
    store_json_only.put(original, fmt="json")
    assert store_json_only.get(str(original.run_metadata.trace_id)) == original

    listed = store.list()
    assert len(listed) == 1
    assert listed[0].trace_id == str(TRACE_ID)
    assert listed[0].schema_version == SCHEMA_VERSION
    assert listed[0].event_count == 2
    assert listed[0].logits_mode == "topk"
    assert listed[0].format == "jsonl"
