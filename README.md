# llmfr (LLM Flight Recorder)

CLI name: `llmfr`.

## Problem

Two LLM generations can look different in the final string while splitting for
very different reasons. The first sampled token may differ. The model-visible
window may have been truncated. Everything after the split may just be
downstream fallout. A string diff cannot tell those apart.

llmfr records a generation so later tools can. A v1 trace keeps run metadata,
environment, model config, generation config, per-step sampled tokens, top-k
logits or probabilities, and both `full_history` and `model_visible_context`.
Hosted APIs that do not return real logprobs are stored as logits-unavailable.
This project will not invent fake logits.

Python-first. Local-only for v1: SQLite index plus on-disk JSON/JSONL, keyed by
a stable `trace_id`. No Kafka, Kubernetes, Redis, or Postgres. No LangChain.

## Milestone 1 (done)

- Versioned Pydantic Trace and Event schema (`SCHEMA_VERSION = 1.0.0`)
- Package layout: `llmfr.core` (schema) and `llmfr.storage` (SQLite + files)
- Human-readable top-k (no full-vocab blob)
- Round-trip serialize/deserialize tests (Case G)
- ADRs for the v1 schema and storage choices

Not in M1: Hugging Face adapter, recorder loop, replay, compare, or a `record`
CLI command. `llmfr version`, `validate`, and `topk` only inspect stored traces.

Install and test:

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
ruff format --check src tests
mypy
```

## Roadmap

- [x] **M1** Schema and storage
- [ ] **M2** Hugging Face adapter
- [ ] **M3** Recorder loop
- [ ] **M5** Compare / first-divergence UI

Design notes: `docs/adr/0001-v1-trace-schema.md` and `docs/adr/0002-v1-storage.md`.
