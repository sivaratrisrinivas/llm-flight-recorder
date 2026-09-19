# Demo

Checked-in traces were recorded with the public CLI against `sshleifer/tiny-gpt2` (Hub commit `5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be`) on CPU, torch `2.14.0+cpu`, transformers `5.17.0`. Scores in the JSONL files are the model's captured top-k, not invented logits. Tiny-gpt2 is a 2-layer smoke checkpoint; sampled strings look like noise. That is expected. For readable English, pass `--model distilbert/distilgpt2` (not downloaded in CI). See `docs/adr/0003-hf-adapter-default-model.md`.

Live path (new `trace_id` values each time):

```bash
A=$(llmfr record "Hello" --store .llmfr --max-new-tokens 6 --seed 1)
B=$(llmfr record "Hello" --store .llmfr --max-new-tokens 6 --seed 2)
llmfr compare "$A" "$B" --store .llmfr
```

The reports below are replayable without a model from `examples/demo/`.

## Demo 1: same prompt, different seed

```bash
llmfr compare examples/demo/demo1_a.jsonl examples/demo/demo1_b.jsonl
```

Exit 1 (diverged). Output captured 2026-09-19:

```
compare a=7d2b9710-9131-470d-abf8-91cb5374b155 b=51594c38-fe66-4436-8bb3-bb41e39dfaf3

CONFIG DIFFERENCE
  generation_config.seed: 1 vs 2

EXECUTION
  recorded length: 6 vs 6 steps
  same steps: (none; paths split at step 0)

FIRST BEHAVIORAL DIVERGENCE
  step 0
  class: sampling
  differences: sampled_token
  sampled token: 'ether' (id=6750) vs ' titan' (id=48047)
  reason: same context, captured logits, and decoding config; sampled tokens differ

LIKELY ENABLING CONFIG
  generation_config.seed: 1 vs 2
  seed difference likely enabled this sampling split

Steps 1+: downstream effects
  later context, logit, and token diffs are not a new root cause
  step 1 (not root cause): full_history, model_visible_context, raw_logits, probabilities, sampled_token ('IENCE' (id=42589) vs ' humankind' (id=47634))
  step 2 (not root cause): full_history, model_visible_context, raw_logits, probabilities, sampled_token ('otomy' (id=38385) vs ' Mich' (id=2843))
  step 3 (not root cause): full_history, model_visible_context, raw_logits, probabilities, sampled_token (' Naz' (id=12819) vs 'ios' (id=4267))
  step 4 (not root cause): full_history, model_visible_context, raw_logits, probabilities, sampled_token ('pex' (id=24900) vs ' ascending' (id=41988))
  step 5 (not root cause): full_history, model_visible_context, raw_logits, probabilities, sampled_token (' peas' (id=22589) vs ' Bust' (id=36988))

notes
  Later context and logit diffs are downstream of the first divergence, not a new root cause.

diverged
```

Read it in this order:

1. CONFIG DIFFERENCE lists the seed change (`1` vs `2`).
2. FIRST BEHAVIORAL DIVERGENCE names step 0 as `sampling`. Captured top-k logits and decoding config still match; different draws are Case B, including a seed change. LIKELY ENABLING CONFIG names that seed. The stored `sampled_prob` of two different tokens is not a `probability distribution` split.
3. Steps 1+ are downstream. Later full_history, visible-context, logit, and token diffs are not a new root cause.

Step 0 inspect of `demo1_a` (`llmfr inspect examples/demo/demo1_a.jsonl --step 0`):

```
trace 7d2b9710-9131-470d-abf8-91cb5374b155  step 0 of 6

sampled token
  'ether' (id=6750)
  rank=17556  logit= 0.0107  prob= 0.0000  logprob=-10.8146

full_history
  tokens=1  truncated=false
  ids=[15496]
  text='Hello'

model_visible_context
  tokens=1  truncated=false
  ids=[15496]
  text='Hello'
  equals full_history (model saw the full prefix)

top-k
   1. ' stairs'        id=16046     logit= 0.1269  prob= 0.0000  logprob=-10.6985
   2. ' vendors'       id=17192     logit= 0.1252  prob= 0.0000  logprob=-10.7001
   3. ' intermittent'  id=38361     logit= 0.1149  prob= 0.0000  logprob=-10.7104
   4. ' hauled'        id=43423     logit= 0.1079  prob= 0.0000  logprob=-10.7174
   5. ' Brew'          id=9702      logit= 0.1065  prob= 0.0000  logprob=-10.7188
```

The sampled token is not in the stored top-5. Top-k is a snapshot, not the vocabulary. See `docs/adr/0001-v1-trace-schema.md`.

## Demo 2: same seed, different temperature

```bash
C=$(llmfr record "Hello" --store .llmfr --max-new-tokens 6 --seed 1 --temperature 0.7)
D=$(llmfr record "Hello" --store .llmfr --max-new-tokens 6 --seed 1 --temperature 1.2)
llmfr compare "$C" "$D" --store .llmfr
```

Fixture replay:

```bash
llmfr compare examples/demo/demo2_a.jsonl examples/demo/demo2_b.jsonl
```

Exit 1 (diverged). Output captured 2026-09-19:

```
compare a=94b4c5ea-08c7-48d5-8848-5d90df0d610c b=000982fd-cbfb-4ad1-8711-8c5edb6f9d7c

CONFIG DIFFERENCE
  generation_config.temperature: 0.7 vs 1.2

EXECUTION
  recorded length: 6 vs 6 steps
  same steps: (none; paths split at step 0)

FIRST BEHAVIORAL DIVERGENCE
  step 0
  class: decoding config
  differences: sampled_token
  sampled token: 'ician' (id=6749) vs 'ether' (id=6750)
  reason: captured logits match; temperature, do_sample, or other sampler settings differ

LIKELY ENABLING CONFIG
  generation_config.temperature: 0.7 vs 1.2
  sampler settings likely enabled this first split

Steps 1+: downstream effects
  later context, logit, and token diffs are not a new root cause
  step 1 (not root cause): full_history, model_visible_context, raw_logits, probabilities ('IENCE' (id=42589) vs 'IENCE' (id=42589))
  step 2 (not root cause): full_history, model_visible_context, raw_logits ('otomy' (id=38385) vs 'otomy' (id=38385))
  step 3 (not root cause): full_history, model_visible_context, raw_logits (' Naz' (id=12819) vs ' Naz' (id=12819))
  step 4 (not root cause): full_history, model_visible_context, raw_logits, sampled_token ('pex' (id=24900) vs 'buster' (id=24899))
  step 5 (not root cause): full_history, model_visible_context, raw_logits, probabilities (' peas' (id=22589) vs ' peas' (id=22589))

notes
  Later context and logit diffs are downstream of the first divergence, not a new root cause.

diverged
```

CONFIG DIFFERENCE and LIKELY ENABLING CONFIG both name temperature. The first split is `decoding config` because captured logits match and sampler settings do not. Later steps can share a sampled token (steps 1-3 and 5 here) and still be downstream: the prefix already diverged, so those logit and history diffs are fallout.

A config-only compare (seed differs, recorded tokens still match) names the config row and does not invent a first behavioral split. See `docs/adr/0006-first-divergence-compare.md`.

## Recapture

```bash
python scripts/capture_demo.py --backend hf
```

Writes `examples/demo/*.jsonl` and the sibling compare/inspect captures. Inspect and compare subprocesses must succeed (non-empty stdout); a failed inspect is not written. `--backend fake` never overwrites `examples/demo` (even with `--force`): that directory is the tiny-gpt2 CLI capture used by this page. For a scratch fake capture that also writes inspect:

```bash
python scripts/capture_demo.py --backend fake --out /tmp/llmfr-demo-fake
```

## Limitations

- Replay matching sampled token ids on this CPU pin is not bit-identical logits across GPU, dtype, or PyTorch builds (`docs/adr/0005-deterministic-replay.md`).
- Hosted APIs that omit real logprobs are recorded as `logits.mode=none`. llmfr does not fill zeros or a uniform vocab.
- Default `sshleifer/tiny-gpt2` is a CI smoke model, not an English demo.
- Compare classifies stored events. It does not call a live model, and it does not upgrade a later logit diff into a second root cause.
- v1 storage is a local directory (SQLite index plus JSON/JSONL). No Kafka, Redis, Postgres, or Kubernetes control plane (`docs/adr/0002-v1-storage.md`).
- No compare UI in this tree. No LangChain wrapper.

## Roadmap (not a promise)

v1 is record, inspect, token replay, and first-divergence compare on local traces. A later schema bump needs a real migrator and a new ADR. More adapters are in scope only when they expose real scores. A UI was scoped as a separate milestone and is not in this release.

## Design notes

Link only; the bodies live in `docs/adr/`.

- `docs/adr/0001-v1-trace-schema.md`
- `docs/adr/0002-v1-storage.md`
- `docs/adr/0003-hf-adapter-default-model.md`
- `docs/adr/0004-recorder-loop.md`
- `docs/adr/0005-deterministic-replay.md`
- `docs/adr/0006-first-divergence-compare.md`
- `docs/adr/0007-root-cause-report.md`
- `docs/adr/0008-production-cli.md`
