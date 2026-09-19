# ADR 0009: Honest OpenAI Chat Completions adapter

Status: accepted  
Date: 2026-09-19

## Context

v1 already records local Hugging Face causal LMs with a full last-position
logit vector (ADR 0003, ADR 0004). That path is the CI default
(`sshleifer/tiny-gpt2` on CPU). Callers also run hosted OpenAI Chat
Completions. Those APIs can return `logprobs` / `top_logprobs` (a short
ranked list, at most 20). They do not return a vocabulary of raw logits, do
not expose the chat-template prefix as token ids, and do not pin a Hub
commit.

Earlier write-ups listed "hosted-API logits" as a non-goal. That remains
true for full-vocab logits. It must not be read as "llmfr will never talk to
OpenAI." Capturing the scores the API actually returns, and refusing to
invent the rest, is in scope.

## Decision

1. **Optional extra, not the default.** `OpenAIChatAdapter` lives behind
   `llmfr[openai]` (`openai` + `tiktoken`). CLI default remains Hugging Face.
   Select OpenAI with `--provider openai` and/or an API model name
   (`gpt-4o-mini`, `openai:gpt-4o-mini`). `org/name` Hugging Face ids and
   bare `gpt2` stay on the HF backend.

2. **Capability flags stay conservative.**
   - `supports_logits=False` — `top_logprobs` is not a vocab vector.
   - `supports_logprobs=True` — the adapter requests real logprobs.
   - `supports_replay=False` and `supports_seed=False` — LocalRNG does not
     drive the API; OpenAI `seed` is forwarded as a request field only.
     Replay of `provider=openai` traces is `not_replayable` (ADR 0005).

3. **Record what the API returned.** `complete_prompt` calls
   `chat.completions.create` once with `logprobs=True` and
   `top_logprobs=k` (capped at 20). Event `top_k` rows store `logprob` /
   `prob` from that list. `logit` stays `None`. If the response omits
   per-token logprob content, or the model rejects logprobs, recording
   fails closed. Do not re-tokenize the completion text into fake steps,
   and do not store an opaque blob. Do not fill zeros, uniforms, or
   softmax over the shortlist and call them the model distribution.

4. **The recorder does not sample locally on this path.** Hosted tokens are
   the API's sampled tokens. Temperature / greedy map to the API request.
   There is no local left-window (`--max-visible-tokens` is rejected).
   Stored `full_history` and `model_visible_context` are tiktoken ids of the
   user prompt plus sampled token strings. Chat-template tokens the hosted
   model prepends are not in the response, so they are not stored.

5. **Secrets stay in the environment.** The client reads `OPENAI_API_KEY`
   only. There is no `--api-key` flag and no hardcoded key. Tests inject a
   fake `chat.completions.create` so CI does not need a live key.

6. **Compare and inspect read the stored trace.** They do not call OpenAI.
   Logprob-only top-k is not classified as `raw-logit`. Replay never claims
   bit-identical OpenAI reproduction.

## Consequences

- HF demos, CI, and the tiny-gpt2 default are unchanged.
- An OpenAI JSON/JSONL file is a valid v1 Trace only when the API returned
  per-token logprob content. Recording fails closed otherwise. Present
  scores are the API's logprobs / `top_logprobs`, which is weaker evidence
  than pinned HF CPU logits.
- Changing the default backend to OpenAI would need a new ADR.

## Alternatives considered

- Treat `top_logprobs` as full logits so LocalRNG replay "works": the lie
  this adapter exists to prevent.
- Re-tokenize the completion with tiktoken, or store the message as an
  opaque blob, when logprobs are missing: both write a trace that looks
  like a token sequence without the API's per-token scores.
- Keep hosted APIs entirely out of v1: honest, but then llmfr only applies
  to local toy checkpoints.
- An `--api-key` CLI flag: easy to paste into shell history and help text.
