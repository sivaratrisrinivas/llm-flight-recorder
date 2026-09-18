# ADR 0001: v1 Trace and Event schema

Status: accepted  
Date: 2026-09-18

## Context

Milestone 1 has to freeze a record of one LLM generation before any adapter or
recorder exists. Later compare work (M5) needs to tell a first token split from
context truncation and from tokens that only changed because the prefix already
diverged. If M1 omits those fields, M5 will have to guess.

The on-disk format also has to stay small and honest. Full-vocabulary logit
tensors are huge and unreadable. Hosted HTTP APIs often expose no real logits
at all. Inventing scores would make replay and compare look precise when they
are not.

## Decision

Freeze `SCHEMA_VERSION = "1.0.0"` in `llmfr.core.version`. A Trace is one
object with these required blocks:

1. `schema_version`  
   Explicit on every file. Unknown versions are rejected. See the migration
   stub in `llmfr.core.migrate`: identity for `1.0.0`, error otherwise. No
   upgrade path in v1.

2. `run_metadata`  
   Stable `trace_id` (UUID, assigned once, round-tripped unchanged),
   `created_at`, prompt text and optional prompt token ids, output text,
   optional source/tags, and `logits` capture policy (`topk` with `k`, or
   `none` with `unavailable_reason`).

3. `environment`  
   Optional facts that can change numeric results: Python version, platform,
   device, accelerator, library versions. All fields optional so a handwritten
   test fixture is still valid. Empty means "not recorded", not "default GPU".

4. `model_config`  
   Which weights: provider, name, optional revision, tokenizer, dtype,
   architecture. Serialized JSON key is `model_config`. The Python attribute
   is `Trace.model` because Pydantic already uses `model_config` for class
   options. Construction accepts `model=` or `model_config=`.

5. `generation_config`  
   Sampler request: temperature, top_p, top_k, max_new_tokens, do_sample,
   seed, repetition_penalty, stop. Distinct from `model_config` so a compare
   can see "same weights, different sampler" without parsing a blob.

6. `events`  
   One object per decode step, `step` contiguous from 0. Each event carries:

   - Sampled token fields: `sampled_token_id`, `sampled_token`, and optional
     `sampled_logit` / `sampled_prob` / `sampled_logprob` / `sampled_rank`.
   - `top_k`: ranked candidates (`rank`, `token_id`, `token`, plus at least one
     of `logit`, `prob`, `logprob`). Length capped at `MAX_TOP_K` (50). This is
     the simple format. Full vocab is invalid.
   - `full_history`: complete prefix used to predict this token (prompt plus
     earlier sampled tokens).
   - `model_visible_context`: tokens actually fed to the model. If
     `truncated` is false, token ids must equal `full_history`. If true, they
     must be a strict suffix (left-truncation). A later compare can then blame
     a window clip instead of the sampler.

`extra="forbid"` on every model. Extensions go in `run_metadata.tags`, not
undeclared keys.

If a backend cannot provide real logits or logprobs, `run_metadata.logits.mode`
is `none` and events must not carry `top_k`. Do not fill zeros, uniforms, or
softmax-over-top-k and call them the model distribution.

## Consequences

- A `1.0.0` file is readable without guessing field names.
- Recorders (M3) and adapters (M2) must populate `full_history` and
  `model_visible_context` even when they are equal. Equality is data, not a
  missing field.
- Schema bumps need a real migrator and a new ADR. Until then, mixed-version
  stores fail closed.
- Compare (M5) can walk events in order, compare sampled tokens, then inspect
  whether visible context already differed.

## Alternatives considered

- A single `prompt` plus `output_ids` with no per-step context: cheaper, but
  cannot explain truncation vs sampling.
- Storing full logits tensors (`.pt`, numpy, pickle): not human-readable, easy
  to smuggle a 32k-wide row into "simple JSON".
- Silent schema coercion: hides incompatibility until compare results lie.
- Synthetic logprobs for hosted APIs: makes M5 look like it has evidence it
  does not have.
