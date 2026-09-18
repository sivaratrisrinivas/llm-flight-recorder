from __future__ import annotations

from dataclasses import fields

import pytest

from llmfr.adapters.base import AdapterCapabilities, StepLogits
from llmfr.adapters.huggingface import HuggingFaceCausalLMAdapter, HuggingFaceExtraMissingError
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


def test_huggingface_extra_missing_error_points_at_extra() -> None:
    err = HuggingFaceExtraMissingError()
    assert "llmfr[hf]" in str(err)


def test_constructor_requires_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    def _missing() -> tuple[object, object]:
        raise HuggingFaceExtraMissingError()

    monkeypatch.setattr("llmfr.adapters.huggingface._import_backend", _missing)
    with pytest.raises(HuggingFaceExtraMissingError, match=r"llmfr\[hf\]"):
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


def test_top_k_candidates_respects_schema_cap() -> None:
    logits = tuple(float(i) for i in range(MAX_TOP_K + 5))
    step = StepLogits(
        token_ids=(1,),
        requested_token_ids=(1,),
        logits=logits,
        truncated=False,
    )
    top = step.top_k_candidates(MAX_TOP_K + 5, decode=str)
    assert len(top) == MAX_TOP_K
    assert top[0].token_id == MAX_TOP_K + 4
