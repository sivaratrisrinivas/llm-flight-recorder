# llmfr

## What

`llmfr` records a local LLM generation step by step so you can replay it and compare two runs: where they first split, and what is only fallout after that.

## Why

Prompt and final-string logs cannot tell those apart. Two outputs can differ because the first sampled token differed, because the model-visible window was truncated, or because everything after an earlier split is downstream. A string diff treats all of that as one blob. llmfr stores per-step tokens, top-k scores when the model actually returned them, and both full history and model-visible context so compare can name the first divergence and label later diffs as not a new root cause.

## How

Core package (schema, storage, CLI, tests with a fake adapter):

```bash
pip install -e ".[dev]"
```

`record` and live `replay` need the Hugging Face extra. Install a CPU torch wheel first so pip does not pull the CUDA-default PyPI build:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install --upgrade-strategy only-if-needed -e ".[dev,hf]"
```

Default model is `sshleifer/tiny-gpt2`. Store directory defaults to `.llmfr`.

```bash
A=$(llmfr record "Hello" --store .llmfr --max-new-tokens 8 --greedy)
B=$(llmfr record "Hello" --store .llmfr --max-new-tokens 8 --greedy)
llmfr inspect "$A" --store .llmfr --step 0
llmfr compare "$A" "$B" --store .llmfr
llmfr replay "$A" --store .llmfr
```

`--greedy` keeps both records on the same argmax path. Different `--seed` values are the usual split case. Compare and inspect also take JSON/JSONL files:

```bash
llmfr compare path/a.jsonl path/b.jsonl
llmfr inspect path/a.jsonl --step 0
```

`replay` takes a store `trace_id` only. Exit 0 means compare traces identical, or replay status `reproduced`. Exit 1 is an expected failure (including a diverged compare). Exit 2 is a usage error. `llmfr --help` lists the rest (`validate`, `topk`, `version`).

## Essentials

- Token replay on a pinned CPU checkpoint is not a promise of bit-identical logits across GPU, dtype, or PyTorch builds. See `docs/adr/0005-deterministic-replay.md`.
- v1 is local-only: SQLite index plus JSON/JSONL files keyed by `trace_id`. Nothing is sent to a hosted store.
- If a backend does not expose real logits or logprobs, the trace records that. llmfr does not invent scores.
- Design notes: `docs/adr/`.
