# llmfr (LLM Flight Recorder)

CLI name: `llmfr`.

Record, replay, and compare LLM generations. When two generations differ, later
milestones will show the first divergence versus downstream effects.

Python-first. Local-only for v1 (SQLite index plus on-disk JSON/JSONL traces).
No Kafka, Kubernetes, Redis, or Postgres. No LangChain. Hosted APIs that do not
return real logprobs are recorded as logits-unavailable. This project will not
invent fake logits.

## Milestone 1 (this branch)

Done in this milestone:

- Versioned Pydantic `Trace` and `Event` schema (`SCHEMA_VERSION = 1.0.0`)
- SQLite index plus JSON / JSONL trace files
- Human-readable top-k logits (no full-vocab blob)
- Round-trip serialize/deserialize tests (Case G)
- Schema version constant and a v1-only ADR (unknown versions are rejected)

Install:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
ruff check src tests
ruff format --check src tests
mypy
```

CLI (schema/storage only in M1):

```bash
llmfr version
llmfr validate path/to/trace.json
llmfr topk path/to/trace.jsonl
```

## Roadmap

- **M1** Schema and storage (this branch)
- **M2** Hugging Face adapter
- **M3** Recorder loop
- **M5** Compare / first-divergence UI

## Design notes

See `docs/adr/0001-schema-v1-only.md`.
