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

2. **Portfolio demo:** ungated `Qwen/Qwen2.5-0.5B-Instruct`  
   Constants: `PORTFOLIO_DEMO_MODEL_ID` and `PORTFOLIO_DEMO_MODEL_REVISION`
   (`7ae557604adf67be50417f59c2c2f167def9a775`). This 0.5B instruct
   checkpoint produces readable English on a short reasoning prompt on CPU.
   Demo scripts pass `--model` and `--revision` explicitly. Tests assert
   against checked-in `examples/demo` traces and do not download these
   weights. `meta-llama/Llama-3.2-1B-Instruct` was preferred for a 1B-class
   demo; that Hub repo is gated and the capture environment had no
   `HF_TOKEN` (HTTP 401 / `GatedRepoError`), so Llama was not recorded.

3. **Install:** torch and transformers live in the optional extra `hf`. A
   core `pip install llmfr` (M1 schema and storage) stays free of those
   wheels. Adapter tests skip when the extra is missing. CI (and README)
   install a CPU torch wheel from `https://download.pytorch.org/whl/cpu`
   first, then `pip install --upgrade-strategy only-if-needed -e ".[dev,hf]"`
   so the default PyPI CUDA torch is not the only documented path.

4. **Resolved Hub revision:** after `from_pretrained`, the adapter stores the
   40-character Hub commit SHA in `ModelConfig.revision` (from
   `config._commit_hash`, then `huggingface_hub.model_info(...).sha`). It
   raises if that pin cannot be determined. `supports_replay=True` depends
   on this pin; `main` or `None` is not stored.

5. **Capabilities this adapter claims**

   - `supports_logits=True` because `AutoModelForCausalLM` returns a vocab
     vector at the last position.
   - `supports_logprobs=True` because logprobs are `log_softmax` of those
     logits, not invented scores.
   - `supports_attention=False` and `supports_hidden_states=False` because
     this adapter does not return those tensors, even if a given HF model
     could compute them.
   - `supports_seed` was False in M2 because eval logits do not depend on a
     seed and the adapter did not sample. Milestone 3 sets this True; the
     recorder consumes `GenerationConfig.seed` with an isolated
     `random.Random`. The adapter still does not call `torch.manual_seed`.
     See ADR 0004.
   - `supports_replay=True` because the same visible token ids and the
     pinned weights reproduce the same logits. The replay CLI is still M4.

Hosted APIs that do not expose real logits stay out of this adapter. Do not
fill zeros or a uniform distribution to look like a local model.

## Consequences

- Default constructor is CI-safe. Demos opt into Qwen2.5-0.5B-Instruct at a
  pinned Hub SHA. CI must not fetch those weights.
- A recorded M2/M3 trace can name the exact Hub commit, so a later replay
  is not chasing `main`.
- M3 can feed `StepLogits.logits` into top-k `Event` rows without changing
  the v1 schema. Sampling seeds belong in M3 `GenerationConfig`, not here.
- Changing the default model id needs a new ADR (or an update to this one)
  so CI time and demo quality stay an explicit choice.

## Alternatives considered

- Default to `gpt2` (124M): better text, slower CI, still CPU-possible.
  Rejected as the implicit default; testers can pass `model_id="gpt2"`.
- Portfolio `distilbert/distilgpt2`: English completion, not an instruct
  reasoning path. Replaced by `Qwen/Qwen2.5-0.5B-Instruct` for the
  checked-in Demo 1/2 captures.
- Portfolio `meta-llama/Llama-3.2-1B-Instruct`: preferred 1B-class instruct
  demo. Not used: the Hub repo is gated, and the capture environment had
  no `HF_TOKEN` (HTTP 401 / `GatedRepoError` on tokenizer and weights).
  Ungated Qwen 0.5B is the accepted portfolio capture.
- Vendor a numpy dump of tiny-gpt2 logits in the repo: avoids the Hub, but
  is exactly the fake-logits path this project forbids.
- Require CUDA: fails this environment and GitHub-hosted runners.
