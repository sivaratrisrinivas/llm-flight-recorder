from __future__ import annotations

from dataclasses import fields

import pytest

from llmfr.adapters.base import AdapterCapabilities, StepLogits
from llmfr.adapters.huggingface import (
    HuggingFaceCausalLMAdapter,
    HuggingFaceExtraMissingError,
    _as_commit_hash,
    _resolved_hub_revision,
)
from llmfr.core.schema import Event, TopKCandidate
from llmfr.core.version import MAX_TOP_K


def test_capability_flag_names() -> None:
    names = {item.name for item in fields(AdapterCapabilities)}
    assert names == {
        "supports_logits",
        "supports_logprobs",
        "supports_attention",
        "supports_hidden_states",
        "supports_seed",
        "supports_replay",
    }


def test_huggingface_extra_missing_error_points_at_cpu_torch_install() -> None:
    err = str(HuggingFaceExtraMissingError())
    assert "https://download.pytorch.org/whl/cpu" in err
    assert "--upgrade-strategy only-if-needed" in err
    assert ".[dev,hf]" in err
    assert "llmfr[hf]" in err
    assert "Install with: pip install 'llmfr[hf]'" not in err


def test_constructor_requires_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    def _missing() -> tuple[object, object]:
        raise HuggingFaceExtraMissingError()

    monkeypatch.setattr("llmfr.adapters.huggingface._import_backend", _missing)
    with pytest.raises(HuggingFaceExtraMissingError, match=r"download\.pytorch\.org/whl/cpu"):
        HuggingFaceCausalLMAdapter()


def test_step_logits_maps_to_m1_contexts_and_topk() -> None:
    step = StepLogits(
        token_ids=(7, 8, 9),
        requested_token_ids=(1, 2, 3, 7, 8, 9),
        logits=(0.1, 4.0, 1.5, 3.0),
        logprobs=(-4.0, -0.2, -2.0, -0.5),
        truncated=True,
    )
    history, visible = step.as_contexts()
    assert history.token_ids == [1, 2, 3, 7, 8, 9]
    assert history.truncated is False
    assert visible.token_ids == [7, 8, 9]
    assert visible.truncated is True
    assert step.greedy_token_id == 1

    top = step.top_k_candidates(2, decode=lambda token_id: f"t{token_id}")
    assert [candidate.token_id for candidate in top] == [1, 3]
    assert top[0].rank == 1
    assert top[0].logit == 4.0
    assert top[0].logprob == -0.2
    assert isinstance(top[0], TopKCandidate)

    event = Event(
        step=0,
        sampled_token_id=step.greedy_token_id,
        sampled_token="t1",
        sampled_logit=top[0].logit,
        sampled_logprob=top[0].logprob,
        sampled_rank=1,
        top_k=top,
        full_history=history,
        model_visible_context=visible,
    )
    assert event.model_visible_context.truncated is True
    assert event.full_history.token_ids[-3:] == event.model_visible_context.token_ids


def test_resolved_hub_revision_requires_commit_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llmfr.adapters.huggingface._hub_commit_sha", lambda *_args, **_kwargs: None
    )

    class _Config:
        _commit_hash = "main"

    class _Model:
        config = _Config()

    with pytest.raises(RuntimeError, match="commit hash"):
        _resolved_hub_revision(model=_Model(), model_id="local-only", requested=None)

    pinned = "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"
    _Config._commit_hash = pinned
    assert _resolved_hub_revision(model=_Model(), model_id="x", requested=None) == pinned
    assert _as_commit_hash("main") is None
    assert _as_commit_hash(pinned.upper()) == pinned


def test_top_k_candidates_rejects_oversized_k() -> None:
    logits = tuple(float(i) for i in range(MAX_TOP_K + 5))
    step = StepLogits(
        token_ids=(1,),
        requested_token_ids=(1,),
        logits=logits,
        truncated=False,
    )
    with pytest.raises(ValueError, match="oversized logit payload"):
        step.top_k_candidates(MAX_TOP_K + 5, decode=str)


def test_top_k_candidates_stores_at_most_max_top_k() -> None:
    logits = tuple(float(i) for i in range(MAX_TOP_K + 5))
    step = StepLogits(
        token_ids=(1,),
        requested_token_ids=(1,),
        logits=logits,
        truncated=False,
    )
    top = step.top_k_candidates(MAX_TOP_K, decode=str)
    assert len(top) == MAX_TOP_K
    assert top[0].token_id == MAX_TOP_K + 4


def test_top_k_candidates_rejects_empty_logits() -> None:
    step = StepLogits(
        token_ids=(1,),
        requested_token_ids=(1,),
        logits=(),
        truncated=False,
    )
    with pytest.raises(ValueError, match="no logits"):
        step.top_k_candidates(5, decode=str)


def test_top_k_candidates_from_captured_logprobs_without_logits() -> None:
    captured = (
        TopKCandidate(rank=1, token_id=9, token="a", logit=None, logprob=-0.1, prob=0.9),
        TopKCandidate(rank=2, token_id=8, token="b", logit=None, logprob=-2.0, prob=0.1),
    )
    step = StepLogits(
        token_ids=(1,),
        requested_token_ids=(1,),
        logits=(),
        truncated=False,
        top_logprobs=captured,
        backend_sampled_token_id=9,
        backend_sampled_logprob=-0.1,
    )
    top = step.top_k_candidates(2, decode=str)
    assert [candidate.token_id for candidate in top] == [9, 8]
    assert all(candidate.logit is None for candidate in top)
    assert top[0].logprob == -0.1
    with pytest.raises(ValueError, match="no logits"):
        _ = step.greedy_token_id
