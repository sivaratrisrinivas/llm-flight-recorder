# ADR 0002: v1 storage (SQLite index plus JSON/JSONL)

Status: accepted  
Date: 2026-09-18

## Context

Traces need two access patterns. Listing recent runs should not parse every
file. Loading one run should return the exact Pydantic Trace that was written,
including events, so Case G round-trips hold.

v1 is a local CLI package. Kafka, Redis, Postgres, and object stores add
moving parts without helping a developer inspect two JSON files.

`trace_id` has to be stable. If serialize, write, read, or a later overwrite
minted a new UUID, compare and replay handles would dangle.

## Decision

Store a directory:

```
<root>/
  index.sqlite
  traces/<trace_id>.json
  traces/<trace_id>.jsonl
```

The SQLite database is an index only. Columns: `trace_id` (primary key),
`schema_version`, `created_at`, model provider/name, prompt and output
previews, event count, logits mode, relative path, and format. Trace bodies
are not SQLite blobs.

JSON is one Trace object (pretty-printed). JSONL is a header record (the Trace
without `events`) followed by one event record per line. Both go through
`schema_version` migration (identity in v1) and Pydantic validation on read.

`trace_id` comes from `run_metadata.trace_id`. The filename is that UUID. The
index primary key is that UUID. `TraceStore.put` does not assign a new id.
Putting an existing id requires `overwrite=True`; otherwise
`DuplicateTraceIdError`. Overwrite keeps the same `trace_id` and replaces the
file plus index row. Switching JSON to JSONL deletes the old file.

Unknown `schema_version` values never land in the index: validation runs on
the payload before callers typically `put`, and `get` validates again.

## Consequences

- Tests can assert `store.get(trace_id).run_metadata.trace_id` equals the id
  that was written.
- M3 can append JSONL event lines later without changing the index shape. M1
  still writes complete files; there is no live recorder in this milestone.
- Backup is "copy the directory". No server.

## Alternatives considered

- Postgres or Redis: out of scope for v1.
- One SQLite table holding full JSON: harder to diff in git, harder to skim.
- Filesystem only, no index: listing becomes a full parse of every header.
- Auto-regenerating UUIDs on write: breaks stable handles.
- Pickle: not inspectable, not versioned in a useful way.
