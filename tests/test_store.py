from __future__ import annotations

from pathlib import Path

import pytest

from llmfr.store import TraceStore
from tests.factories import make_trace


def test_store_lists_and_loads_index(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    trace = make_trace()
    store.put(trace, fmt="jsonl")
    entries = store.list()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.model_name == "tiny-model"
    assert entry.prompt_preview.startswith("Say hello")
    assert store.get_index(str(trace.trace_id)).event_count == 2


def test_store_unknown_id(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    with pytest.raises(KeyError):
        store.get("missing")


def test_store_overwrites_same_id(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    first = make_trace()
    store.put(first, fmt="json")
    second = first.model_copy(update={"output_text": "Hello world!"})
    store.put(second, fmt="json")
    assert store.get(str(first.trace_id)).output_text == "Hello world!"
    assert len(store.list()) == 1
