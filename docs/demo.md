# Demo

Checked-in traces were recorded with the public CLI against
`Qwen/Qwen2.5-0.5B-Instruct` (Hub commit
`7ae557604adf67be50417f59c2c2f167def9a775`) on CPU, torch `2.14.0+cpu`,
transformers `5.17.0`. Scores in the JSONL files are the model's captured
top-k, not invented logits. The library default remains `sshleifer/tiny-gpt2`
(CI/smoke). Omit `--model` for that path; CI does not download the 0.5B
weights. See `docs/adr/0003-hf-adapter-default-model.md`.

Live path (new `trace_id` values each time):

```bash
PROMPT="A farmer has 17 sheep. All but 9 run away. How many sheep are left? Think step by step, then give the final number."
A=$(llmfr record "$PROMPT" --model Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775 --store .llmfr --max-new-tokens 16 --seed 1)
B=$(llmfr record "$PROMPT" --model Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775 --store .llmfr --max-new-tokens 16 --seed 2)
llmfr compare "$A" "$B" --store .llmfr
```

The reports below are replayable without a model from `examples/demo/`.

## Demo 1: same prompt, different seed

```bash
llmfr compare examples/demo/demo1_a.jsonl examples/demo/demo1_b.jsonl
```

Exit 1 (diverged). Output captured 2026-09-19:

```
compare a=c9276fdb-be8b-4470-9b4a-d19c3ae7e589 b=d8ee1917-67f3-4b39-8cd4-f0acc7a60cdc

CONFIG DIFFERENCE
  generation_config.seed: 1 vs 2

EXECUTION
  recorded length: 16 vs 16 steps
  same steps: (none; paths split at step 0)

FIRST BEHAVIORAL DIVERGENCE
  step 0
  class: sampling
  differences: sampled_token
  sampled token: ' To' (id=2014) vs ' Step' (id=14822)
  reason: same context, captured logits, and decoding config; sampled tokens differ

LIKELY ENABLING CONFIG
  generation_config.seed: 1 vs 2
  seed difference likely enabled this sampling split

Steps 1+: downstream effects
  later context, logit, and token diffs are not a new root cause
  step 1 (not root cause): full_history, model_visible_context, raw_logits, probabilities, sampled_token (' determine' (id=8253) vs ' ' (id=220))
  step 2 (not root cause): full_history, model_visible_context, raw_logits, probabilities, sampled_token (' how' (id=1246) vs '1' (id=16))
  step 3 (not root cause): full_history, model_visible_context, raw_logits, probabilities, sampled_token (' many' (id=1657) vs ':' (id=25))
  step 4 (not root cause): full_history, model_visible_context, raw_logits, probabilities, sampled_token (' sheep' (id=31912) vs ' Identify' (id=64547))
  step 5 (not root cause): full_history, model_visible_context, raw_logits, probabilities, sampled_token (' are' (id=525) vs ' the' (id=279))
  step 6 (not root cause): full_history, model_visible_context, raw_logits, probabilities, sampled_token (' left' (id=2115) vs ' initial' (id=2856))
  step 7 (not root cause): full_history, model_visible_context, raw_logits, probabilities, sampled_token (' after' (id=1283) vs ' number' (id=1372))
  step 8 (not root cause): full_history, model_visible_context, raw_logits, probabilities, sampled_token (' all' (id=678) vs ' of' (id=315))
  step 9 (not root cause): full_history, model_visible_context, raw_logits, probabilities, sampled_token (' but' (id=714) vs ' sheep' (id=31912))
  step 10 (not root cause): full_history, model_visible_context, raw_logits, probabilities, sampled_token (' ' (id=220) vs '.\n' (id=624))
  step 11 (not root cause): full_history, model_visible_context, raw_logits, probabilities, sampled_token ('9' (id=24) vs 'The' (id=785))
  step 12 (not root cause): full_history, model_visible_context, raw_logits, probabilities, sampled_token (' run' (id=1598) vs ' farmer' (id=36400))
  step 13 (not root cause): full_history, model_visible_context, raw_logits, probabilities, sampled_token (' away' (id=3123) vs ' starts' (id=8471))
  step 14 (not root cause): full_history, model_visible_context, raw_logits, probabilities, sampled_token (',' (id=11) vs ' with' (id=448))
  step 15 (not root cause): full_history, model_visible_context, raw_logits, probabilities, sampled_token (' we' (id=582) vs ' a' (id=264))

notes
  Later context and logit diffs are downstream of the first divergence, not a new root cause.

diverged
```

Read it in this order:

1. CONFIG DIFFERENCE lists the seed change (`1` vs `2`).
2. FIRST BEHAVIORAL DIVERGENCE names step 0 as `sampling`. Captured top-k logits and decoding config still match; different draws are Case B, including a seed change. LIKELY ENABLING CONFIG names that seed. The stored `sampled_prob` of two different tokens is not a `probability distribution` split.
3. Steps 1+ are downstream. Later full_history, visible-context, logit, and token diffs are not a new root cause.

Seed 1 continues `To determine how many sheep are left after all but 9 run away, we`. Seed 2 continues `Step 1: Identify the initial number of sheep.` then `The farmer starts with a`. That is two readable solution paths, not tokenizer noise.

Step 0 inspect of `demo1_a` (`llmfr inspect examples/demo/demo1_a.jsonl --step 0`):

```
trace c9276fdb-be8b-4470-9b4a-d19c3ae7e589  step 0 of 16

sampled token
  ' To' (id=2014)
  rank=1  logit= 18.7956  prob= 0.5313  logprob=-0.6325

full_history
  tokens=32  truncated=false
  ids=[32, 36400, 702, 220, 16, 22, 31912, 13, 2009, 714, 220, 24, 1598, 3123, 13, 2585, 1657, 31912, 525, 2115, 30, 21149, 3019, 553, 3019, 11, 1221, 2968, 279, 1590, 1372, 13]
  text='A farmer has 17 sheep. All but 9 run away. How many sheep are left? Think step by step, then give the final number.'

model_visible_context
  tokens=32  truncated=false
  ids=[32, 36400, 702, 220, 16, 22, 31912, 13, 2009, 714, 220, 24, 1598, 3123, 13, 2585, 1657, 31912, 525, 2115, 30, 21149, 3019, 553, 3019, 11, 1221, 2968, 279, 1590, 1372, 13]
  text='A farmer has 17 sheep. All but 9 run away. How many sheep are left? Think step by step, then give the final number.'
  equals full_history (model saw the full prefix)

top-k
 * 1. ' To'            id=2014      logit= 18.7956  prob= 0.5313  logprob=-0.6325
   2. ' Let'           id=6771      logit= 17.6427  prob= 0.1677  logprob=-1.7854
   3. ' Step'          id=14822     logit= 17.2073  prob= 0.1085  logprob=-2.2207
   4. ' First'         id=5512      logit= 16.1834  prob= 0.0390  logprob=-3.2447
   5. ' The'           id=576       logit= 16.1077  prob= 0.0361  logprob=-3.3204
```

The sampled token is rank 1 in the stored top-5. Seed 2's `' Step'` is rank 3 in that same snapshot. Top-k is still not the vocabulary. See `docs/adr/0001-v1-trace-schema.md`.

## Demo 2: same seed, different temperature

```bash
C=$(llmfr record "$PROMPT" --model Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775 --store .llmfr --max-new-tokens 16 --seed 1 --temperature 0.7)
D=$(llmfr record "$PROMPT" --model Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775 --store .llmfr --max-new-tokens 16 --seed 1 --temperature 1.2)
llmfr compare "$C" "$D" --store .llmfr
```

Fixture replay:

```bash
llmfr compare examples/demo/demo2_a.jsonl examples/demo/demo2_b.jsonl
```

Exit 1 (diverged). Output captured 2026-09-19:

```
compare a=90c53473-a255-4df4-b4b6-0062b98f6971 b=bb1d2ce9-506b-49e3-b873-44419c18b41b

CONFIG DIFFERENCE
  generation_config.temperature: 0.7 vs 1.2

EXECUTION
  recorded length: 16 vs 16 steps
  same steps: 0-7 (sampled tokens, model-visible context, captured logits still match)

FIRST BEHAVIORAL DIVERGENCE
  step 8
  class: decoding config
  differences: sampled_token
  sampled token: ' all' (id=678) vs ' the' (id=279)
  reason: captured logits match; temperature, do_sample, or other sampler settings differ

LIKELY ENABLING CONFIG
  generation_config.temperature: 0.7 vs 1.2
  sampler settings likely enabled this first split

Steps 9+: downstream effects
  later context, logit, and token diffs are not a new root cause
  step 9 (not root cause): full_history, model_visible_context, raw_logits, probabilities, sampled_token (' but' (id=714) vs ' ' (id=220))
  step 10 (not root cause): full_history, model_visible_context, raw_logits, probabilities, sampled_token (' ' (id=220) vs '9' (id=24))
  step 11 (not root cause): full_history, model_visible_context, raw_logits, probabilities, sampled_token ('9' (id=24) vs ' sheep' (id=31912))
  step 12 (not root cause): full_history, model_visible_context, raw_logits, probabilities (' run' (id=1598) vs ' run' (id=1598))
  step 13 (not root cause): full_history, model_visible_context, raw_logits, probabilities, sampled_token (' away' (id=3123) vs ' off' (id=1007))
  step 14 (not root cause): full_history, model_visible_context, raw_logits, probabilities (',' (id=11) vs ',' (id=11))
  step 15 (not root cause): full_history, model_visible_context, raw_logits, probabilities (' we' (id=582) vs ' we' (id=582))

notes
  Later context and logit diffs are downstream of the first divergence, not a new root cause.

diverged
```

CONFIG DIFFERENCE and LIKELY ENABLING CONFIG both name temperature. Steps 0-7 still match (`To determine how many sheep are left after`). The first split is `decoding config` because captured logits match and sampler settings do not: `all but 9 run away` versus `the 9 sheep run off`. Later steps can share a sampled token (steps 12, 14, and 15 here) and still be downstream: the prefix already diverged, so those logit and history diffs are fallout.

A config-only compare (seed differs, recorded tokens still match) names the config row and does not invent a first behavioral split. See `docs/adr/0006-first-divergence-compare.md`.

## Recapture

```bash
python scripts/capture_demo.py --backend hf
```

Writes `examples/demo/*.jsonl` and the sibling compare/inspect captures. This path downloads `Qwen/Qwen2.5-0.5B-Instruct` at the pinned Hub revision. It is not a CI job. Inspect and compare subprocesses must succeed (non-empty stdout); a failed inspect is not written. `--backend fake` never overwrites `examples/demo` (even with `--force`): that directory is the portfolio CLI capture used by this page. For a scratch fake capture that also writes inspect:

```bash
python scripts/capture_demo.py --backend fake --out /tmp/llmfr-demo-fake
```

## Limitations

- Replay matching sampled token ids on this CPU pin is not bit-identical logits across GPU, dtype, or PyTorch builds (`docs/adr/0005-deterministic-replay.md`).
- Hosted traces that already store `logits.mode=none` stay honest; compare does not invent logits. OpenAI recording fails closed if the API omits per-token logprob content: it does not re-tokenize the completion into fake steps. Returned `top_logprobs` are stored as logprob-only top-k (`logit=None`). That is not a full-vocab logit vector, and OpenAI replay is `not_replayable`. llmfr does not fill zeros or a uniform vocab.
- Default `sshleifer/tiny-gpt2` is a CI smoke model, not an English demo.
  Portfolio Demo 1/2 captures use `Qwen/Qwen2.5-0.5B-Instruct` at
  `7ae557604adf67be50417f59c2c2f167def9a775`. CI does not download those
  weights.
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
- `docs/adr/0009-openai-adapter.md`
