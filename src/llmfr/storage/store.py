"""SQLite index plus JSON/JSONL trace files keyed by a stable trace_id."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from llmfr.core.schema import Trace, dumps_json, dumps_jsonl, loads_json, loads_jsonl

FormatName = Literal["json", "jsonl"]

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_provider TEXT NOT NULL,
    prompt_preview TEXT NOT NULL,
    output_preview TEXT NOT NULL,
    event_count INTEGER NOT NULL,
    logits_mode TEXT NOT NULL,
    relpath TEXT NOT NULL,
    format TEXT NOT NULL CHECK (format IN ('json', 'jsonl'))
);

CREATE INDEX IF NOT EXISTS idx_traces_created_at ON traces(created_at);
CREATE INDEX IF NOT EXISTS idx_traces_model ON traces(model_name);
"""

_PREVIEW_CHARS = 200


class DuplicateTraceIdError(ValueError):
    """Raised when putting a trace_id that already exists without overwrite=True."""


class TraceIndexEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    schema_version: str
    created_at: str
    model_name: str
    model_provider: str
    prompt_preview: str
    output_preview: str
    event_count: int
    logits_mode: str
    relpath: str
    format: FormatName


class TraceStore:
    """Directory of traces: index.sqlite plus traces/<stable-id>.json[l]."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.traces_dir = root / "traces"
        self.db_path = root / "index.sqlite"
        self.root.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    def put(self, trace: Trace, fmt: FormatName = "jsonl", *, overwrite: bool = False) -> Path:
        trace_id = str(trace.run_metadata.trace_id)
        filename = f"{trace_id}.{fmt}"
        dest = self.traces_dir / filename
        existing = self._index_row(trace_id)
        if existing is not None or dest.exists():
            if not overwrite:
                raise DuplicateTraceIdError(f"trace_id already exists: {trace_id}")
            if existing is not None:
                old_path = self.root / existing.relpath
                if old_path != dest and old_path.exists():
                    old_path.unlink()
        text = dumps_jsonl(trace) if fmt == "jsonl" else dumps_json(trace)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(dest)
        entry = _entry_from_trace(trace, relpath=f"traces/{filename}", fmt=fmt)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO traces (
                    trace_id, schema_version, created_at, model_name, model_provider,
                    prompt_preview, output_preview, event_count, logits_mode, relpath, format
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.trace_id,
                    entry.schema_version,
                    entry.created_at,
                    entry.model_name,
                    entry.model_provider,
                    entry.prompt_preview,
                    entry.output_preview,
                    entry.event_count,
                    entry.logits_mode,
                    entry.relpath,
                    entry.format,
                ),
            )
        return dest

    def get(self, trace_id: str) -> Trace:
        entry = self.get_index(trace_id)
        path = self.root / entry.relpath
        text = path.read_text(encoding="utf-8")
        if entry.format == "jsonl":
            return loads_jsonl(text)
        return loads_json(text)

    def get_index(self, trace_id: str) -> TraceIndexEntry:
        row = self._index_row(trace_id)
        if row is None:
            raise KeyError(f"unknown trace_id {trace_id}")
        return row

    def list(self) -> list[TraceIndexEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM traces ORDER BY created_at DESC, trace_id ASC"
            ).fetchall()
        return [_row_to_entry(row) for row in rows]

    def _index_row(self, trace_id: str) -> TraceIndexEntry | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM traces WHERE trace_id = ?",
                (trace_id,),
            ).fetchone()
        if row is None:
            return None
        return _row_to_entry(row)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def _preview(text: str) -> str:
    if len(text) <= _PREVIEW_CHARS:
        return text
    return text[:_PREVIEW_CHARS]


def _entry_from_trace(trace: Trace, relpath: str, fmt: FormatName) -> TraceIndexEntry:
    dumped = trace.model_dump(mode="json")
    meta = trace.run_metadata
    return TraceIndexEntry(
        trace_id=str(meta.trace_id),
        schema_version=trace.schema_version,
        created_at=dumped["run_metadata"]["created_at"],
        model_name=trace.model.name,
        model_provider=trace.model.provider,
        prompt_preview=_preview(meta.prompt),
        output_preview=_preview(meta.output_text),
        event_count=len(trace.events),
        logits_mode=meta.logits.mode,
        relpath=relpath,
        format=fmt,
    )


def _row_to_entry(row: sqlite3.Row) -> TraceIndexEntry:
    return TraceIndexEntry(
        trace_id=row["trace_id"],
        schema_version=row["schema_version"],
        created_at=row["created_at"],
        model_name=row["model_name"],
        model_provider=row["model_provider"],
        prompt_preview=row["prompt_preview"],
        output_preview=row["output_preview"],
        event_count=row["event_count"],
        logits_mode=row["logits_mode"],
        relpath=row["relpath"],
        format=row["format"],
    )
