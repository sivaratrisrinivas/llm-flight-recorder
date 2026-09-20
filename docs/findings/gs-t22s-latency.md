# GS-T22s: CLI wall-clock latency (p50/p99)

Status: measured
Date: 2026-09-20T05:47:59.156865+00:00

## Question

What is the wall-clock latency of `llmfr record`, `llmfr compare`, and
`llmfr study` on the portfolio Qwen CPU path? Report measured p50 and p99.
Do not invent timings. Fake-adapter and tiny-gpt2 smoke runs are not this table.

## Setup

- **Warmup:** 2 discarded invocations per command
  (not included in p50/p99).
- **N trials:** 11 timed invocations per command.
- **Percentile:** nearest-rank: rank = ceil(p/100 * n), value = sorted[rank-1]. With N=11, p99 is the maximum timed trial.
- **Clock:** `time.perf_counter` around `subprocess`
  `python -m llmfr ...` (same interpreter as `llmfr`).
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- **Library default / CI smoke:** `sshleifer/tiny-gpt2` and `--backend fake`
  are not used for this finding.
- **record:** Demo 1 prompt, `--max-new-tokens 16`
  `--seed 1`, persist to a fresh temp store each trial.
- **compare:** checked-in Demo 1 JSONL paths (no model). Exit 1 (diverged)
  is expected and counted as a successful timed trial.
- **study:** 1-item `examples/findings/gs-t22s/prompts.jsonl` (`sheep_trick`),
  `--max-new-tokens 16` (not the 8-item / 64-token
  study default). Four recorded sides; the model loads once per invocation.
- **Hardware:** Linux-6.12.94+-x86_64-with-glibc2.39; Python 3.12.3;
  cpus=4; cpu_model=Intel(R) Xeon(R) Processor.
- **Library versions:** {'llmfr': '0.1.0', 'torch': '2.14.0+cpu', 'transformers': '5.17.0'}
- **Backend:** `hf` (`subprocess`).

Re-run:

```bash
python scripts/gs_t22s_latency.py --backend hf --write-docs
```

CI smoke (fake adapter; must not overwrite this finding):

```bash
python scripts/gs_t22s_latency.py --backend fake --out /tmp/llmfr-gs-t22s-fake
```

Raw JSON: `docs/findings/gs-t22s-results.json`.
Study prompt: `examples/findings/gs-t22s/prompts.jsonl`.

## Result

Measured seconds. min/max are from the same timed trials as p50/p99.

| command | p50 (s) | p99 (s) | min (s) | max (s) |
|---|---:|---:|---:|---:|
| `llmfr record` | 6.969 | 7.335 | 6.762 | 7.335 |
| `llmfr compare` | 0.151 | 0.158 | 0.146 | 0.158 |
| `llmfr study` | 16.854 | 19.715 | 15.783 | 19.715 |

## Commands timed

- record: `llmfr record 'A farmer has 17 sheep. All but 9 run away. How many sheep are left? Think step by step, then give the final number.' --store .llmfr-gs-t22s --max-new-tokens 16 --seed 1 --capture-k 5 --model Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775`
- compare: `llmfr compare examples/demo/demo1_a.jsonl examples/demo/demo1_b.jsonl`
- study: `llmfr study examples/findings/gs-t22s/prompts.jsonl --store .llmfr-gs-t22s --max-new-tokens 16 --capture-k 5 --model Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775`

## Limits

- N is small on purpose. This is not an SLA and not a CI budget.
- Each record/study trial includes interpreter start, checkpoint load,
  generation, and persist. Compare does not call a model.
- 0.5B-class instruct model on CPU, no chat template (same raw-prompt
  style as the portfolio demo).
- The 1-item / 16-token study workload is not GS-T22q (N=30, 64 tokens).
- tiny-gpt2 answers and `--backend fake` timings are smoke. They are
  not the table above.
- Llama-3.2-1B-Instruct was not used (gated; no HF_TOKEN).
