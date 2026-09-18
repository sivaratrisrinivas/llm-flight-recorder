# ADR 0001: Schema v1 only

Status: accepted  
Date: 2026-09-18

## Context

Milestone 1 needs a versioned Trace and Event schema, on-disk traces, and an
index. Later milestones will add adapters, recording, and comparison. Shipping a
migration framework now would be speculative.

## Decision

1. Freeze the on-disk schema at `SCHEMA_VERSION = "1.0.0"`. The constant lives
   in `llmfr.version.SCHEMA_VERSION`.
2. Reject payloads with any other `schema_version`. `llmfr.migrate.migrate_payload`
   is a stub: identity for `1.0.0`, error otherwise. There is no upgrade path
   in v1.
3. Store traces as JSON (one object) or JSONL (header line, then one event per
   line). Keep a SQLite file as an index only. Trace bodies are not stored as
   blobs inside SQLite.
4. Capture at most top-k logits per step in a human-readable candidate list
   (`rank`, `token_id`, `token`, `logit`, optional `logprob`). Do not serialize
   full vocabulary tensors.
5. If a backend cannot provide real logits or logprobs (typical hosted HTTP
   APIs), set `logits.mode = "none"` and an `unavailable_reason`. Do not fill
   zeros, uniforms, or other synthetic scores.

## Consequences

- Readers can trust that a `1.0.0` file matches this package.
- A future schema bump must land with a real migrator and a new ADR. Until then,
  mixed-version stores fail closed.
- Compare (M5) can assume contiguous `event.step` values starting at 0.

## Alternatives considered

- Postgres / Redis: out of scope for v1; extra moving parts for a local CLI.
- Pickle or raw `.pt` logit dumps: not inspectable, easy to smuggle a full vocab.
- Silent schema coercion: hides incompatibility until compare results lie.
