# ADR 0004: Token-by-token recorder loop and isolated sampling RNG

Status: accepted  
Date: 2026-09-19

## Context

Milestone 3 has to record a generation as a sequence of M1 Events without
inventing logits. Milestone 2 already returns real last-position logits for a
model-visible prefix and left-clips to `max_visible_tokens`. Compare work (M5)
needs `full_history` and `model_visible_context` at every step, plus the token
that was actually sampled.

Sampling is not part of the Hugging Face forward pass. Calling
`torch.manual_seed` would reseed process-global PyTorch state, so other
tensors, DataLoaders, or libraries in the same process would share that seed
whether they meant to or not. Eval-mode logits for these causal LMs are
already deterministic for a given prefix.

## Decision

1. **Recorder owns the loop.** `llmfr.record.record_generation` walks, in
   order: full history, model-visible context (adapter left-clip), raw
   logits, temperature-adjusted logits, probabilities, sample, chosen token,
   append, next step. It writes a v1 `Trace` and may `TraceStore.put` it.

2. **Contexts follow M1 rules.** The recorder passes the complete prefix into
   `ModelAdapter.next_token_logits`. The adapter returns `requested_token_ids`
   (full history) and clipped `token_ids` (visible). `StepLogits.as_contexts`
   maps those onto `Event.full_history` and `Event.model_visible_context`.

3. **Stored scores are the model distribution.** Event `logit` / `logprob` /
   `prob` and top-k rows come from raw logits (and the adapter's
   `log_softmax`). Temperature and optional sampler `top_k` change only the
   draw, not the stored model scores.

4. **Seed uses `random.Random`, not `torch.manual_seed`.** `LocalRNG` wraps a
   private `random.Random(seed)` and inverse-CDF sampling. A seed affects this
   object's draws only. It does not reset `random.seed` or global torch RNG
   state. Reproducibility is: same adapter logits + same `LocalRNG` seed +
   same sampler settings. It is not "the whole process is seeded."

5. **`supports_seed=True` on the HF adapter.** M2 left the flag False because
   there was no sampling path. The flag now means the recorder will honor
   `GenerationConfig.seed` for this backend. If a later adapter cannot, it
   must keep the flag False; the recorder refuses a seed rather than ignoring
   it. Greedy (`do_sample=False` or temperature <= 0) stores the seed but
   does not draw.

6. **Stored `generation_config` is the effective policy.** If `is_greedy` is
   true, including a request with `do_sample=True` and temperature <= 0, the
   Trace stores `do_sample=False`. Temperature is left as requested so a zero
   or negative value remains the reason argmax was used. Replay (M4) must
   read this stored config. Never persist `do_sample=True` for an argmax run.

7. **Out of scope here:** `top_p`, `repetition_penalty`, replay (M4), compare
   (M5/M6), and a polished `record` CLI (M7). The CLI prints `trace_id`.

## Consequences

- A recorded JSON/JSONL file is a valid M1 Trace: SQLite index, top-k format,
  truncation suffix rule.
- Tests can prove seed divergence on a flat fake distribution without torch.
- Replay (M4) can feed stored visible ids back into the adapter; it should
  not assume a process-global torch seed was set. It can trust
  `generation_config.do_sample` on the Trace: False means tokens were argmax.

## Alternatives considered

- `torch.manual_seed` before each sample: shortest path, but process-global
  and easy to lie about in a test suite that also uses torch elsewhere.
- `torch.Generator` local to the recorder: also isolated, but would pull
  torch into the sampling module and skip CPU-only unit tests without the
  `hf` extra.
- Adapter-level `generate()`: hides the per-step pipeline M3 has to record.
