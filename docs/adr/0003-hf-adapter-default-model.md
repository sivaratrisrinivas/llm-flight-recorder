# ADR 0003: Hugging Face adapter default models

Status: accepted  
Date: 2026-09-18

## Context

Milestone 2 adds a local Hugging Face causal LM adapter so later recorder work
can read real next-token logits. CI and this development box have no GPU.
Downloading a multi-billion-parameter checkpoint would make adapter tests
impractical. A toy checkpoint that never runs a real `forward` would also be
wrong: M2 must return token ids and logits from the model, not a fixture of
made-up scores.

The portfolio demo still needs a model that produces recognizable English,
because tiny-gpt2 (2 layers, 2 heads, embedding size 2) is only a smoke test.

## Decision

1. **CI and library default:** `sshleifer/tiny-gpt2`  
   `HuggingFaceCausalLMAdapter()` loads this id on `device="cpu"` with
   `float32`. The checkpoint is a GPT-2 LM head (`vocab_size=50257`,
   `n_positions=1024`) small enough to download and run in CI. Tests call
   `next_token_logits` on that model. They may mark the download as slow.
   They must not substitute a canned logit vector.

2. **Portfolio demo:** `distilbert/distilgpt2`  
   Constant: `PORTFOLIO_DEMO_MODEL_ID`. DistilGPT2 is a real English causal
   LM that still runs short prompts on CPU. Demo scripts should pass
   `model_id=PORTFOLIO_DEMO_MODEL_ID` explicitly. Tests do not download it.

3. **Install:** torch and transformers live in the optional extra `hf`. A
   core `pip install llmfr` (M1 schema and storage) stays free of those
   wheels. Adapter tests skip when the extra is missing; CI installs
   `.[dev,hf]` so the tiny model path actually runs.

4. **Capabilities this adapter claims**

   - `supports_logits=True` because `AutoModelForCausalLM` returns a vocab
     vector at the last position.
   - `supports_logprobs=True` because logprobs are `log_softmax` of those
     logits, not invented scores.
   - `supports_attention=False` and `supports_hidden_states=False` because
     this adapter does not return those tensors, even if a given HF model
     could compute them.
   - `supports_seed=True` because `torch.manual_seed` is applied when a seed
     is passed. Eval-mode logits on CPU are deterministic with or without it.
   - `supports_replay=True` because the same visible token ids and weights
     reproduce the same logits. The replay CLI is still M4.

Hosted APIs that do not expose real logits stay out of this adapter. Do not
fill zeros or a uniform distribution to look like a local model.

## Consequences

- Default constructor is CI-safe. Demos opt into DistilGPT2.
- M3 can feed `StepLogits.logits` into top-k `Event` rows without changing
  the v1 schema.
- Changing the default model id needs a new ADR (or an update to this one)
  so CI time and demo quality stay an explicit choice.

## Alternatives considered

- Default to `gpt2` (124M): better text, slower CI, still CPU-possible.
  Rejected as the implicit default; testers can pass `model_id="gpt2"`.
- Vendor a numpy dump of tiny-gpt2 logits in the repo: avoids the Hub, but
  is exactly the fake-logits path this project forbids.
- Require CUDA: fails this environment and GitHub-hosted runners.
