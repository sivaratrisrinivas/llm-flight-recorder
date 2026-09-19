from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from llmfr.storage.store import DuplicateTraceIdError, TracePathError, TraceStore, _index_error
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


def test_failed_format_change_keeps_old_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = TraceStore(tmp_path)
    trace = make_trace()
    json_path = store.put(trace, fmt="json")

    def boom(_trace: object) -> str:
        raise RuntimeError("dump fail")

    monkeypatch.setattr("llmfr.storage.store.dumps_jsonl", boom)
    with pytest.raises(RuntimeError, match="dump fail"):
        store.put(trace, fmt="jsonl", overwrite=True)

    assert json_path.is_file()
    loaded = store.get(str(trace.run_metadata.trace_id))
    assert loaded == trace


def test_put_insert_conflict_maps_to_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = TraceStore(tmp_path)
    trace = make_trace()
    jsonl_path = store.put(trace, fmt="jsonl")
    monkeypatch.setattr(store, "_index_row", lambda _trace_id: None)
    with pytest.raises(DuplicateTraceIdError):
        store.put(trace, fmt="json")
    monkeypatch.undo()
    json_path = jsonl_path.with_suffix(".json")
    assert jsonl_path.is_file()
    assert not json_path.exists()
    assert list(store.traces_dir.glob("*.tmp")) == []
    loaded = store.get(str(trace.run_metadata.trace_id))
    assert loaded == trace


def test_get_rejects_relpath_outside_traces(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    secret = tmp_path / "secret.json"
    secret.write_text("{}", encoding="utf-8")
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            """
            INSERT INTO traces (
                trace_id, schema_version, created_at, model_name, model_provider,
                prompt_preview, output_preview, event_count, logits_mode, relpath, format
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "evil",
                "1.0.0",
                "2026-01-01T00:00:00+00:00",
                "m",
                "p",
                "x",
                "y",
                0,
                "none",
                "../secret.json",
                "json",
            ),
        )
        conn.commit()
    with pytest.raises(TracePathError, match="escapes store"):
        store.get("evil")


def test_open_existing_missing_store_path(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-store"
    with pytest.raises(FileNotFoundError, match="trace store not found"):
        TraceStore(missing, create=False)


def test_open_existing_missing_index(tmp_path: Path) -> None:
    root = tmp_path / "empty-root"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="trace store index missing"):
        TraceStore(root, create=False)


def test_get_missing_trace_file(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    trace = make_trace()
    path = store.put(trace, fmt="jsonl")
    path.unlink()
    with pytest.raises(FileNotFoundError, match="trace file missing from store"):
        store.get(str(trace.run_metadata.trace_id))


def test_get_corrupt_trace_file(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    trace = make_trace()
    path = store.put(trace, fmt="jsonl")
    path.write_text('{"record":"header"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt or partial"):
        store.get(str(trace.run_metadata.trace_id))


def test_get_permission_error_is_not_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = TraceStore(tmp_path)
    trace = make_trace()
    store.put(trace, fmt="jsonl")

    def _denied(_self: Path, *_args: object, **_kwargs: object) -> str:
        raise PermissionError("permission denied")

    monkeypatch.setattr(Path, "read_text", _denied)
    with pytest.raises(PermissionError, match="permission denied") as excinfo:
        store.get(str(trace.run_metadata.trace_id))
    assert "corrupt" not in str(excinfo.value)


def test_corrupt_index_is_clear_error(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    (root / "traces").mkdir()
    (root / "index.sqlite").write_text("not a sqlite database", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt trace store index"):
        TraceStore(root, create=False).list()


def test_index_lock_is_not_labeled_corrupt() -> None:
    err = _index_error(sqlite3.OperationalError("database is locked"))
    assert str(err).startswith("trace store index error:")
    assert "corrupt" not in str(err)


def test_index_io_error_is_not_labeled_corrupt() -> None:
    err = _index_error(sqlite3.OperationalError("disk I/O error"))
    assert str(err).startswith("trace store index error:")
    assert "corrupt" not in str(err)
