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

Inspect CLI: `llmfr version`, `validate`, and `topk`.

## Milestone 2 (done)

- `ModelAdapter` protocol with capability flags: `supports_logits`,
  `supports_logprobs`, `supports_attention`, `supports_hidden_states`,
  `supports_seed`, `supports_replay`
- Hugging Face causal LM adapter: feed model-visible token ids, get the
  model's real next-token logits and the ids that were actually fed
- CPU default `sshleifer/tiny-gpt2` for CI; portfolio demo model is
  `distilbert/distilgpt2` (see `docs/adr/0003-hf-adapter-default-model.md`)
- Optional extra `hf` so a core M1 install does not pull torch

## Milestone 3 (done on this branch)

- Token-by-token generation recorder. Each step, in order: full history,
  model-visible context, raw logits, temperature-adjusted logits,
  probabilities, sampling, chosen token, append
- Distinguishes `full_history` vs `model_visible_context` at every step
  (left-windowing matches the M1 suffix/truncation rules)
- Persists via existing `TraceStore` (SQLite index plus JSON/JSONL, top-k)
- `supports_seed=True` on the HF adapter. Seed drives an isolated
  `random.Random` in the recorder, not `torch.manual_seed` (see
  `docs/adr/0004-recorder-loop.md`)
- Minimal `llmfr record` that writes a store and prints `trace_id`

Not in M3: replay, compare, or a polished record CLI.

## Install and test

Core (schema, storage, inspect CLI, recorder unit tests with a fake adapter):

```bash
pip install -e ".[dev]"
pytest tests/test_schema.py tests/test_store.py tests/test_sample.py tests/test_recorder.py tests/test_cli.py
ruff check src tests
ruff format --check src tests
mypy
```

Hugging Face adapter and recorder (CPU, downloads `sshleifer/tiny-gpt2` on
first run). Install the CPU torch wheel first, matching CI. A bare
`pip install -e ".[dev,hf]"` can pull the CUDA-default PyPI torch build.

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install --upgrade-strategy only-if-needed -e ".[dev,hf]"
pytest
```

Recorder-only with the live model:

```bash
pytest tests/test_recorder_hf.py tests/test_huggingface_adapter.py tests/test_adapter_protocol.py
```

`tests/test_huggingface_adapter.py` and `tests/test_recorder_hf.py` are marked
`slow` and are skipped unless `torch` and `transformers` are installed. They
do not stub logits.

Minimal record CLI (prints `trace_id`):

```bash
llmfr record "Hello" --store .llmfr --max-new-tokens 8 --greedy
```

## Roadmap

- [x] **M1** Schema and storage
- [x] **M2** Hugging Face adapter
- [x] **M3** Recorder loop
- [ ] **M4** Replay
- [ ] **M5** Compare / first-divergence UI

Design notes: `docs/adr/0001-v1-trace-schema.md`, `docs/adr/0002-v1-storage.md`,
`docs/adr/0003-hf-adapter-default-model.md`, and
`docs/adr/0004-recorder-loop.md`.
