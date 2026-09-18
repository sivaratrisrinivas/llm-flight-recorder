"""Live Hugging Face adapter tests. Need `pip install 'llmfr[hf]'` and Hub access."""

from __future__ import annotations

import math
import statistics

import pytest

from llmfr.adapters.base import ModelAdapter
from llmfr.adapters.huggingface import DEFAULT_HF_MODEL_ID, HuggingFaceCausalLMAdapter
from llmfr.core.schema import Event, LogitsCapture, ModelConfig, RunMetadata, Trace
from llmfr.core.version import DEFAULT_TOP_K, SCHEMA_VERSION

pytest.importorskip("torch")
pytest.importorskip("transformers")

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def adapter() -> HuggingFaceCausalLMAdapter:
    return HuggingFaceCausalLMAdapter(
        DEFAULT_HF_MODEL_ID,
        device="cpu",
        seed=0,
        max_visible_tokens=8,
    )


def test_hf_adapter_satisfies_protocol(adapter: HuggingFaceCausalLMAdapter) -> None:
    assert isinstance(adapter, ModelAdapter)
    caps = adapter.capabilities
    assert caps.supports_logits is True
    assert caps.supports_logprobs is True
    assert caps.supports_attention is False
    assert caps.supports_hidden_states is False
    assert caps.supports_seed is True
    assert caps.supports_replay is True
    assert adapter.model_id == DEFAULT_HF_MODEL_ID
    assert adapter.max_visible_tokens == 8
    assert isinstance(adapter.model_config, ModelConfig)
    assert adapter.model_config.provider == "huggingface"
    assert adapter.model_config.name == DEFAULT_HF_MODEL_ID
    assert adapter.environment().device == "cpu"


def test_hf_adapter_returns_real_logits_and_token_ids(
    adapter: HuggingFaceCausalLMAdapter,
) -> None:
    prompt = "Hello"
    token_ids = adapter.encode(prompt)
    assert token_ids, "tokenizer returned no ids"
    assert all(isinstance(token_id, int) for token_id in token_ids)

    step = adapter.next_token_logits(token_ids)
    assert step.truncated is False
    assert list(step.token_ids) == token_ids
    assert list(step.requested_token_ids) == token_ids
    assert len(step.logits) == adapter.vocab_size
    assert adapter.vocab_size > 1
    assert step.logprobs is not None
    assert len(step.logprobs) == adapter.vocab_size

    # Real model scores, not a constant or a uniform fake.
    assert max(step.logits) != min(step.logits)
    assert statistics.pstdev(step.logits) > 0
    assert all(math.isfinite(value) for value in step.logits[:32])

    greedy_id = step.greedy_token_id
    assert 0 <= greedy_id < adapter.vocab_size
    greedy_token = adapter.decode_token(greedy_id)
    assert isinstance(greedy_token, str)

    logprobs = step.logprobs
    total = sum(math.exp(value) for value in logprobs)
    assert total == pytest.approx(1.0, rel=1e-3, abs=1e-3)

    again = adapter.next_token_logits(token_ids)
    assert again.logits == step.logits
    assert again.token_ids == step.token_ids


def test_hf_adapter_left_truncates_visible_context(
    adapter: HuggingFaceCausalLMAdapter,
) -> None:
    overlong = list(range(adapter.max_visible_tokens + 3))
    step = adapter.next_token_logits(overlong)
    assert step.truncated is True
    assert list(step.requested_token_ids) == overlong
    assert list(step.token_ids) == overlong[-adapter.max_visible_tokens :]
    assert len(step.logits) == adapter.vocab_size

    history, visible = step.as_contexts()
    event = Event(
        step=0,
        sampled_token_id=step.greedy_token_id,
        sampled_token=adapter.decode_token(step.greedy_token_id),
        sampled_logit=step.logits[step.greedy_token_id],
        sampled_logprob=None if step.logprobs is None else step.logprobs[step.greedy_token_id],
        sampled_rank=1,
        top_k=step.top_k_candidates(DEFAULT_TOP_K, decode=adapter.decode_token),
        full_history=history,
        model_visible_context=visible,
    )
    trace = Trace(
        schema_version=SCHEMA_VERSION,
        run_metadata=RunMetadata(
            prompt="overlong-ids",
            prompt_token_ids=overlong,
            logits=LogitsCapture(mode="topk", k=DEFAULT_TOP_K),
        ),
        model=adapter.model_config,
        environment=adapter.environment(),
        events=[event],
    )
    assert trace.events[0].model_visible_context.truncated is True
    assert len(trace.events[0].top_k) == DEFAULT_TOP_K
    assert trace.events[0].top_k[0].logit is not None


def test_hf_adapter_rejects_empty_context(adapter: HuggingFaceCausalLMAdapter) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        adapter.next_token_logits([])
