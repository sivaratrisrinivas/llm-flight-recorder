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

Inspect CLI: `llmfr version`, `validate`, and `topk` (file traces). Milestone 7
adds `llmfr inspect` for store ids and `--step` views.

## Milestone 2 (done)

- `ModelAdapter` protocol with capability flags: `supports_logits`,
  `supports_logprobs`, `supports_attention`, `supports_hidden_states`,
  `supports_seed`, `supports_replay`
- Hugging Face causal LM adapter: feed model-visible token ids, get the
  model's real next-token logits and the ids that were actually fed
- CPU default `sshleifer/tiny-gpt2` for CI; portfolio demo model is
  `distilbert/distilgpt2` (see `docs/adr/0003-hf-adapter-default-model.md`)
- Optional extra `hf` so a core M1 install does not pull torch

## Milestone 3 (done)

- Token-by-token generation recorder. Each step, in order: full history,
  model-visible context, raw logits, temperature-adjusted logits,
  probabilities, sampling, chosen token, append
- Distinguishes `full_history` vs `model_visible_context` at every step
  (left-windowing matches the M1 suffix/truncation rules)
- Persists via existing `TraceStore` (SQLite index plus JSON/JSONL, top-k)
- `supports_seed=True` on the HF adapter. Seed drives an isolated
  `random.Random` in the recorder, not `torch.manual_seed` (see
  `docs/adr/0004-recorder-loop.md`)
- `llmfr record` writes a store and prints `trace_id`

## Milestone 4 (done)

- Replay a stored Trace as far as the runtime permits, using the recorded
  seed, effective `generation_config`, and Hub revision pin
  (`ModelConfig.revision`)
- Re-runs the M3 sampling path (`LocalRNG` + `choose_token`) against the
  same adapter weights. Does not invent logits to force a match
- Structured `ReplayResult` for CLI and tests: `reproduced`,
  `partially_reproduced`, `not_reproduced`, `not_replayable`, plus separate
  flags for token match vs bit-identical logits
- ADR for what cannot be bit-identical across GPU/CPU/PyTorch/kernels
  (`docs/adr/0005-deterministic-replay.md`)
- `llmfr replay TRACE_ID` against the existing TraceStore layout

## Milestone 5 (done)

- Two-trace compare. Config difference report covers seed, temperature,
  model id/revision, prompt/history, tokenizer, and other
  `generation_config` / `run_metadata` fields that matter
- Walks both traces token-by-token and classifies the first causal split
  as one of: prompt/history, tokenizer, model-visible context,
  model/version, raw-logit, decoding config, probability distribution,
  sampling, unknown/runtime
- Later context and logit diffs after that point are tagged **downstream**,
  not a new root cause (Case E)
- `llmfr compare TRACE_A TRACE_B` prints a short report (`--json` for
  structured `CompareResult`). TRACE_A/B may be files or TraceStore ids
- Core classification tests use stored traces and the fake adapter. They
  do not invent logits for hosted `logits.mode=none` traces

## Milestone 6 (done)

- Compare report names root cause vs downstream effects in a fixed
  section order: CONFIG DIFFERENCE, EXECUTION (where paths still match),
  FIRST BEHAVIORAL DIVERGENCE, LIKELY ENABLING CONFIG, then Steps N+:
  downstream effects
- LIKELY ENABLING CONFIG is the subset of config diffs that likely
  enabled the first split (seed for sampling, sampler fields for
  decoding config). Unrelated diffs stay in CONFIG DIFFERENCE only
- Later context, logit, and token diffs are labeled not a new root
  cause. Case E stays correct: those later diffs are not a second class
- Extends `llmfr compare` formatting and `CompareResult` fields
  (`matched_prefix_steps`, `likely_enabling_config`). No new command

## Milestone 7 (done on this branch)

- Production Typer CLI: `llmfr record`, `llmfr replay`, `llmfr compare`,
  and `llmfr inspect --step N`
- `inspect` shows history vs visible context, sampled token, and top-k
  for one recorded step. `validate` and `topk` still work on trace files
- `--help` documents commands and exit codes. Expected failures print
  `error:` without a traceback
- Happy path: record twice (or load two traces), inspect a step, compare.
  See [Definition of Done](#definition-of-done) below

## Commands

| Command | What it does |
| --- | --- |
| `llmfr record PROMPT` | Record a generation into `--store` and print `trace_id` |
| `llmfr replay TRACE_ID` | Replay from TraceStore; JSON `ReplayResult` |
| `llmfr compare A B` | First divergence vs downstream effects (files or ids) |
| `llmfr inspect TRACE --step N` | One step: history vs visible context, top-k, sampled token |
| `llmfr validate PATH` | Load a JSON/JSONL file and confirm the v1 schema |
| `llmfr topk PATH` | Full-trace top-k dump for a JSON/JSONL file |
| `llmfr version` | Package and schema versions |

`llmfr --help` and `llmfr COMMAND --help` describe flags. Store commands
default to `--store .llmfr`. Compare and inspect accept a file path or a
`trace_id`. Replay takes a store id only.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Success. `compare`: traces identical. `replay`: status `reproduced`. |
| 1 | Expected failure: missing or invalid input, unknown `trace_id`, out-of-range `--step`, compare diverged, replay not reproduced. |
| 2 | Usage error: unknown command or invalid options. |

Expected failures write `error: ...` to stderr and do not print a traceback.

## Definition of Done

Record two traces, inspect a step, then compare. Needs the `hf` extra and a
CPU torch wheel (see install below). `--greedy` keeps the two records on the
same argmax path so compare can exit 0.

```bash
A=$(llmfr record "Hello" --store .llmfr --max-new-tokens 8 --greedy)
B=$(llmfr record "Hello" --store .llmfr --max-new-tokens 8 --greedy)
llmfr inspect "$A" --store .llmfr --step 0
llmfr compare "$A" "$B" --store .llmfr
```

Different seeds are the split case. Compare then exits 1 and names sampling
when the first sampled token differs:

```bash
A=$(llmfr record "Hello" --store .llmfr --max-new-tokens 8 --seed 1)
B=$(llmfr record "Hello" --store .llmfr --max-new-tokens 8 --seed 2)
llmfr compare "$A" "$B" --store .llmfr
```

File traces work the same way:

```bash
llmfr compare path/a.jsonl path/b.jsonl
llmfr inspect path/a.jsonl --step 0
```

Replay a stored id (JSON `ReplayResult`; exit 0 only when reproduced):

```bash
llmfr replay "$A" --store .llmfr
```

## Install and test

Core (schema, storage, CLI, recorder unit tests with a fake adapter):

```bash
pip install -e ".[dev]"
pytest tests/test_schema.py tests/test_store.py tests/test_sample.py tests/test_recorder.py tests/test_replay.py tests/test_compare.py tests/test_compare_report.py tests/test_cli.py
ruff check src tests
ruff format --check src tests
mypy
```

Hugging Face adapter, recorder, and replay (CPU, downloads `sshleifer/tiny-gpt2`
on first run). Install the CPU torch wheel first, matching CI. A bare
`pip install -e ".[dev,hf]"` can pull the CUDA-default PyPI torch build.

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install --upgrade-strategy only-if-needed -e ".[dev,hf]"
pytest
```

Recorder and replay with the live model:

```bash
pytest tests/test_recorder_hf.py tests/test_replay_hf.py tests/test_huggingface_adapter.py tests/test_adapter_protocol.py
```

`tests/test_huggingface_adapter.py`, `tests/test_recorder_hf.py`, and
`tests/test_replay_hf.py` are marked `slow` and are skipped unless `torch`
and `transformers` are installed. They do not stub logits.

## Roadmap

- [x] **M1** Schema and storage
- [x] **M2** Hugging Face adapter
- [x] **M3** Recorder loop
- [x] **M4** Replay
- [x] **M5** Compare / first-divergence
- [x] **M6** Downstream-effects / root-cause narrative
- [x] **M7** Polished CLI UX (this branch)

Design notes: `docs/adr/0001-v1-trace-schema.md`, `docs/adr/0002-v1-storage.md`,
`docs/adr/0003-hf-adapter-default-model.md`, `docs/adr/0004-recorder-loop.md`,
`docs/adr/0005-deterministic-replay.md`,
`docs/adr/0006-first-divergence-compare.md`,
`docs/adr/0007-root-cause-report.md`, and
`docs/adr/0008-production-cli.md`.
