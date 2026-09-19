# ADR 0006: First-divergence compare on stored traces

Status: accepted  
Date: 2026-09-19

## Context

Milestone 5 has to answer why two recorded generations split. A string diff of
the final text cannot tell a prompt change from a truncated window, a sampler
draw, or tokens that only changed because the prefix had already diverged.

The recorder (ADR 0004) already walks, in order: full history, model-visible
context, raw logits, temperature logits, probabilities, sample, append. The
v1 Event schema (ADR 0001) stores `full_history`, `model_visible_context`,
top-k scores, and the sampled token. Replay (ADR 0005) is a one-trace check.
Compare is two stored traces and must not call a live model for the core
classification.

Root-cause narrative polish is Milestone 6. This ADR only names the first
divergence and tags later fallout as downstream.

## Decision

1. **Compare is deterministic on stored traces.**
   `llmfr.compare.compare_traces` loads two v1 Traces. It does not run an
   adapter. It does not invent logits when `logits.mode=none`.

2. **Config report is separate from first divergence.**
   The report always lists run-level fields that matter and differ: seed,
   temperature, other `generation_config` sampler fields, model id/revision
   /tokenizer/dtype, prompt and `prompt_token_ids`, logits capture policy,
   and environment fields that can change numerics. A config diff without a
   token or context split is not a first divergence.

3. **Walk events in step order.**
   At each step, compare `full_history`, `model_visible_context`, captured
   top-k logits, stored probabilities, and sampled tokens. The first step
   with a causal difference is the first divergence.

4. **Classify on the recorder pipeline, earliest stage wins.**
   One of: `prompt/history`, `tokenizer`, `model-visible context`,
   `model/version`, `raw-logit`, `decoding config`,
   `probability distribution`, `sampling`, `unknown/runtime`.
   Same prompt text with different `prompt_token_ids` is tokenizer, not
   prompt/history. Same visible context with a different model id or
   revision is model/version, not raw-logit. Same captured top-k logits
   with a different temperature or `do_sample` is decoding config. Same
   logits and decoding config with a different sampled token is sampling
   (Case B, including a seed change). Hosted traces with `logits.mode=none`
   never get a raw-logit class; they fall through to sampling when seeds
   differ, otherwise `unknown/runtime`.

5. **Later diffs are downstream (Case E).**
   After the first divergence, later full_history, visible-context, logit,
   and token diffs are tagged `downstream`. They are not a second root
   cause. A truncated window that already differs at the first split is
   `model-visible context` even when logits also differ (Case F).

6. **CLI is `llmfr compare TRACE_A TRACE_B`.**
   Arguments may be JSON/JSONL paths or TraceStore ids (`--store`). Default
   output is a short text report. `--json` prints the structured
   `CompareResult`. Exit 0 only when there is no first divergence and no
   config diff of interest.

7. **Out of scope.**
   M6 root-cause narrative beyond this classification, polished CLI (M7),
   and live-model classification tests.

## Consequences

- Tests can build Case B/E/F with the fake adapter or handwritten traces.
- M6 can consume `first_divergence` and `downstream` without changing the
  v1 schema.
- A token match with a different seed is config-only; compare will not
  invent a sampling split that the traces did not record.

## Alternatives considered

- Teacher-force every later prefix and reclassify logit diffs: that is the
  lie Case E exists to prevent.
- Softmax over stored top-k as the model distribution: forbidden. Top-k is
  a snapshot, not a vocab.
- Require a live replay before compare: slower, and not needed to classify
  stored events.
