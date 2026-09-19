# ADR 0005: Deterministic replay is token replay, not silent bit-identity

Status: accepted  
Date: 2026-09-19

## Context

Milestone 4 replays a stored v1 Trace. The recorder (ADR 0004) already walked
raw logits, temperature, probabilities, and sample, and it stored the
effective `generation_config` plus a Hub commit in `ModelConfig.revision`
(ADR 0003). Callers will want to ask whether a later runtime "got the same
generation."

That question has two answers, and they are not the same. Sampled token ids
can match on a supported CPU stack with a pinned checkpoint and the
recorder's isolated `LocalRNG`. The float logits behind those tokens are
products of a specific kernel, dtype, device, and PyTorch build. Treating a
token match as a bit-identical numeric replay would hide the second answer.

Compare / first-divergence UI (M5) and downstream-effects analysis (M6) are
out of scope here. Replay only has to re-run the sampling path as far as
this runtime permits and report what matched.

## Decision

1. **Replay re-runs the M3 sampling path.** `llmfr.replay.replay_trace` loads
   the Trace, rebuilds (or accepts) a `RecordableAdapter`, starts from stored
   `prompt_token_ids`, and at each step calls `next_token_logits` then
   `choose_token` with `LocalRNG(generation_config.seed)` and the stored
   effective `generation_config`. It does not call `torch.manual_seed`. It
   does not invent logits to force a match.

2. **Weights are the stored Hub pin.** Hugging Face replay constructs
   `HuggingFaceCausalLMAdapter(name, revision=ModelConfig.revision)` only
   after the revision matches the same 40-character commit SHA rule as
   `_COMMIT_SHA` in the adapter. A missing revision, or a moving ref such
   as `main`, is `not_replayable` before `from_pretrained`. The CLI is
   `llmfr replay TRACE_ID` against the existing TraceStore layout (ADR 0002).

3. **The result is structured and honest.** `ReplayResult` records
   `status`, `matched_steps` / `total_steps`, `token_ids_matched`,
   `logits_bit_identical`, and `bit_identical`. Status `reproduced` means
   every recorded sampled token id was produced again. It does not mean
   the logit vectors were bit-identical. `bit_identical` is true only when
   tokens matched and stored sampled / top-k logits compared equal as
   Python floats. Token-id agreement on a `logit=None` (prob-only) top-k
   row is not a numeric comparison. If tokens match and no stored logit
   floats were compared, or a compared float disagrees, the result stays
   `reproduced` with `logits_bit_identical=False` and a note. It never
   reports a full numeric match from tokens alone.

4. **What cannot be bit-identical.** Even with the same prompt and pin:

   - CPU vs GPU (and CUDA vs MPS) use different kernels.
   - PyTorch version, cuDNN, Intel MKL / OpenBLAS, and fused vs unfused
     matmuls change float32 rounding.
   - dtype changes (bf16, fp16, tf32) change values and sometimes argmax.
   - Nondeterministic CUDA algorithms can change values from run to run
     on the same GPU.
   - A recorded GPU trace replayed on CPU (or the reverse) is best-effort
     token replay, not a numeric identity check.

   Case A-ish coverage in this repo is: same supported CPU torch wheel,
   `sshleifer/tiny-gpt2` at the stored SHA, float32, eval mode, greedy or
   seeded `LocalRNG`. Tests may assert token identity, and on that stack
   they may also see equal stored logits. That is not a portability
   guarantee.

5. **Fail closed when replay would have to guess.** `logits.mode=none`,
   sampling without a seed, missing Hub revision, `supports_replay=False`,
   unimplemented sampler fields (`top_p`, `repetition_penalty`), and any
   `model.provider=openai` trace yield `not_replayable` rather than a fake
   success. OpenAI `top_logprobs` are not a vocabulary of raw logits, and
   hosted sampling is not a pinned HF CPU checkpoint.

6. **Out of scope:** first-divergence UI (M5), downstream-effects (M6),
   polished CLI (M7). Partial token match is reported as
   `partially_reproduced` with `first_unmatched_step`. That is a replay
   score, not a compare product.

## Consequences

- CLI and tests assert fields on `ReplayResult` instead of a boolean "ok".
- A green Case A CPU test does not license README claims of cross-device
  determinism.
- Later compare work can consume the same per-step token and logit flags
  without changing the v1 Trace schema.

## Alternatives considered

- Teacher-force every recorded prefix even after a sampled token differs:
  closer to M5 first-divergence, and it would hide that a free-running
  generation already left the recorded path.
- Require bit-identical logits for `status=reproduced`: would fail Case A
  token replay on any float drift and push users to ignore the result.
- Silently treat token match as full replay: the lie this ADR exists to
  prevent.
