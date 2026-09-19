# llmfr

## What

`llmfr` records a local LLM generation step by step so you can replay it and compare two runs: where they first split, and what is only fallout after that.

## Why

Prompt and final-string logs cannot tell those apart. Two outputs can differ because the first sampled token differed, because the model-visible window was truncated, or because everything after an earlier split is downstream. A string diff treats all of that as one blob. llmfr stores per-step tokens, top-k scores when the model actually returned them, and both full history and model-visible context so compare can name the first divergence and label later diffs as not a new root cause.

## Architecture

Prompt goes to the recorder. The adapter encodes once, then supplies `next_token_logits` each step. TraceStore holds the result. Replay, compare, and inspect read stored traces. Compare does not call a model.

```mermaid
flowchart TD
  prompt[prompt] --> rec
  subgraph rec [recorder loop]
    direction LR
    hist[full history] --> vis[model-visible context]
    vis --> raw[raw logits]
    raw --> temp[temperature logits]
    temp --> probs[probs]
    probs --> sample[sample]
    sample --> append[append]
    adapter[adapter]
    adapter -->|"encode once"| hist
    adapter -->|"next_token_logits each step"| vis
  end
  rec --> store[TraceStore]
  store --> sqlite[SQLite index]
  store --> files[JSON / JSONL]
  store --> replay[replay]
  store --> compare[compare]
  store --> inspect[inspect]
```

Compare two traces: config fields that differ, then the first causal split, then later diffs tagged as downstream of that split.

```mermaid
flowchart LR
  a[trace A] --> cmp[compare]
  b[trace B] --> cmp
  cmp --> cfg[config diff]
  cmp --> first[first divergence]
  first --> down[downstream effects]
```

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

Default model is `sshleifer/tiny-gpt2`. Store directory defaults to `.llmfr`. Record twice, then compare:

```bash
A=$(llmfr record "Hello" --store .llmfr --max-new-tokens 6 --seed 1)
B=$(llmfr record "Hello" --store .llmfr --max-new-tokens 6 --seed 2)
llmfr compare "$A" "$B" --store .llmfr
```

That is Demo 1: same prompt, different `--seed`. Compare names the first split; later token, context, and logit diffs are downstream, not a second root cause. Demo 2 keeps the seed and changes temperature:

```bash
C=$(llmfr record "Hello" --store .llmfr --max-new-tokens 6 --seed 1 --temperature 0.7)
D=$(llmfr record "Hello" --store .llmfr --max-new-tokens 6 --seed 1 --temperature 1.2)
llmfr compare "$C" "$D" --store .llmfr
```

Captured `sshleifer/tiny-gpt2` output (real logits, not invented) is in `docs/demo.md`. Replay the exact reports without a model:

```bash
llmfr compare examples/demo/demo1_a.jsonl examples/demo/demo1_b.jsonl
llmfr inspect examples/demo/demo1_a.jsonl --step 0
```

`--greedy` on both records is the identical path (exit 0). `llmfr inspect TRACE --step N` shows history vs visible context plus that step's top-k. `llmfr replay TRACE_ID` needs a store id. Exit 0 means compare traces identical, or replay status `reproduced`. Exit 1 is an expected failure (including a diverged compare). Exit 2 is a usage error. `llmfr --help` lists the rest (`validate`, `topk`, `version`).

## Essentials

- Token replay on a pinned CPU checkpoint is not a promise of bit-identical logits across GPU, dtype, or PyTorch builds. See `docs/adr/0005-deterministic-replay.md`.
- v1 is local-only: SQLite index plus JSON/JSONL files keyed by `trace_id`. Nothing is sent to a hosted store.
- Callers can redact prompt, output, sampled-token strings, top-k token strings, and context text before write (`redact_trace`, `llmfr record --redact`) or skip the store (`persist=False`, `--no-persist`). Local persist stays the default.
- If a backend does not expose real logits or logprobs, the trace records that. llmfr does not invent scores.
- v1 does not claim a compare UI, hosted-API logits, LangChain, or a Kafka/Redis/Postgres store. Known limits and a short roadmap: `docs/demo.md`.
- Design notes (do not duplicate here): `docs/adr/`.
