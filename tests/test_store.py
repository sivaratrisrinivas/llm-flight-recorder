from __future__ import annotations

from pathlib import Path

import pytest

from llmfr.storage.store import DuplicateTraceIdError, TraceStore
from tests.factories import make_trace


def test_store_lists_and_loads_index(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    trace = make_trace()
    store.put(trace, fmt="jsonl")
    entries = store.list()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.trace_id == str(trace.run_metadata.trace_id)
    assert entry.model_name == "tiny-model"
    assert entry.prompt_preview.startswith("Say hello")
    assert store.get_index(entry.trace_id).event_count == 2


def test_store_unknown_id(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    with pytest.raises(KeyError):
        store.get("missing")


def test_store_refuses_duplicate_id_without_overwrite(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    first = make_trace()
    store.put(first, fmt="json")
    with pytest.raises(DuplicateTraceIdError):
        store.put(first, fmt="json")


def test_store_overwrite_keeps_stable_trace_id(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    first = make_trace()
    store.put(first, fmt="json")
    second = first.model_copy(
        update={
            "run_metadata": first.run_metadata.model_copy(update={"output_text": "Hello world!"})
        }
    )
    store.put(second, fmt="json", overwrite=True)
    loaded = store.get(str(first.run_metadata.trace_id))
    assert loaded.run_metadata.trace_id == first.run_metadata.trace_id
    assert loaded.run_metadata.output_text == "Hello world!"
    assert len(store.list()) == 1
