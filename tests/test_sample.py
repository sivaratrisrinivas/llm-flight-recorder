"""Sampling pipeline unit tests. No model, no torch."""

from __future__ import annotations

import math

import pytest

from llmfr.core.schema import GenerationConfig
from llmfr.record.sample import (
    SAMPLING_STAGES,
    LocalRNG,
    categorical,
    choose_token,
    effective_generation_config,
    is_greedy,
    mask_top_k,
    scale_logits,
    softmax,
)


class _FixedRNG:
    def __init__(self, draw: float) -> None:
        self.draw = draw

    def random(self) -> float:
        return self.draw


def test_sampling_stages_are_ordered() -> None:
    assert SAMPLING_STAGES == (
        "raw_logits",
        "temperature_logits",
        "probabilities",
        "sample",
    )


def test_temperature_scales_before_softmax() -> None:
    raw = (0.0, 4.0, 2.0)
    decision = choose_token(raw, temperature=2.0, greedy=True)
    assert decision.stages == SAMPLING_STAGES
    assert decision.temperature_logits == (0.0, 2.0, 1.0)
    assert decision.probabilities == pytest.approx(softmax(decision.temperature_logits))
    assert decision.probabilities != pytest.approx(softmax(raw))
    assert decision.sampled_token_id == 1


def test_sampling_uses_temperature_distribution_not_raw() -> None:
    raw = (0.0, 1.0, 0.0)
    draw = 0.15
    cool = choose_token(raw, temperature=0.1, greedy=False, rng=_FixedRNG(draw))
    warm = choose_token(raw, temperature=1.0, greedy=False, rng=_FixedRNG(draw))
    assert cool.sampled_token_id == 1
    assert warm.sampled_token_id == 0
    assert cool.probabilities != pytest.approx(warm.probabilities)


def test_softmax_of_scaled_logits_matches_decision() -> None:
    raw = (1.0, 3.0, 0.0)
    scaled = scale_logits(raw, 2.0)
    decision = choose_token(raw, temperature=2.0, greedy=True)
    assert decision.temperature_logits == scaled
    assert decision.probabilities == pytest.approx(softmax(scaled))


def test_local_rng_is_seed_deterministic_and_isolated() -> None:
    import random

    random.seed(0)
    rng_a = LocalRNG(7)
    first = [rng_a.random() for _ in range(8)]
    random.seed(99)
    rng_b = LocalRNG(7)
    second = [rng_b.random() for _ in range(8)]
    rng_c = LocalRNG(8)
    other = [rng_c.random() for _ in range(8)]
    assert first == second
    assert first != other
    # Process-global random was mutated; a new LocalRNG(7) still matches.
    rng_again = LocalRNG(7)
    assert [rng_again.random() for _ in range(8)] == first


def test_seed_changes_sampled_tokens_on_a_flat_distribution() -> None:
    raw = tuple(0.0 for _ in range(32))

    def _draw(seed: int) -> list[int]:
        rng = LocalRNG(seed)
        return [
            choose_token(raw, temperature=1.0, greedy=False, rng=rng).sampled_token_id
            for _ in range(16)
        ]

    ids_a = _draw(1)
    ids_again = _draw(1)
    ids_b = _draw(2)
    assert ids_a == ids_again
    assert ids_a != ids_b


def test_greedy_does_not_require_rng() -> None:
    decision = choose_token((0.1, 9.0, 0.2), temperature=1.0, greedy=True)
    assert decision.greedy is True
    assert decision.sampled_token_id == 1


def test_sampling_without_rng_raises() -> None:
    with pytest.raises(ValueError, match="RNG"):
        choose_token((0.0, 1.0), temperature=1.0, greedy=False, rng=None)


def test_is_greedy_from_generation_config() -> None:
    assert is_greedy(GenerationConfig(do_sample=False, temperature=1.0)) is True
    assert is_greedy(GenerationConfig(do_sample=True, temperature=0.0)) is True
    assert is_greedy(GenerationConfig(do_sample=True, temperature=1.0)) is False
    assert is_greedy(GenerationConfig()) is True


def test_effective_generation_config_records_greedy_do_sample() -> None:
    requested = GenerationConfig(
        max_new_tokens=4,
        do_sample=True,
        temperature=0.0,
        seed=9,
    )
    stored = effective_generation_config(requested)
    assert stored.do_sample is False
    assert stored.temperature == 0.0
    assert stored.seed == 9
    assert stored.max_new_tokens == 4

    already = GenerationConfig(do_sample=False, temperature=1.0)
    assert effective_generation_config(already) is already

    sampling = GenerationConfig(do_sample=True, temperature=1.0)
    assert effective_generation_config(sampling) is sampling
    assert sampling.do_sample is True

    from_default = effective_generation_config(GenerationConfig(max_new_tokens=1))
    assert from_default.do_sample is False
    assert from_default.temperature is None

    negative = effective_generation_config(
        GenerationConfig(do_sample=True, temperature=-1.0, max_new_tokens=1)
    )
    assert negative.do_sample is False
    assert negative.temperature == -1.0


def test_mask_top_k_then_softmax() -> None:
    masked = mask_top_k((1.0, 3.0, 2.0, 0.0), k=2)
    assert masked[0] == float("-inf")
    assert masked[3] == float("-inf")
    probs = softmax(masked)
    assert probs[0] == 0.0
    assert probs[3] == 0.0
    assert probs[1] + probs[2] == pytest.approx(1.0)


def test_categorical_edges() -> None:
    probs = (0.25, 0.25, 0.5)
    assert categorical(probs, _FixedRNG(0.0)) == 0
    assert categorical(probs, _FixedRNG(0.24)) == 0
    assert categorical(probs, _FixedRNG(0.25)) == 1
    assert categorical(probs, _FixedRNG(0.99)) == 2


def test_softmax_rejects_empty_and_all_masked() -> None:
    with pytest.raises(ValueError, match="no logits"):
        softmax(())
    with pytest.raises(ValueError, match="all-masked"):
        softmax((float("-inf"), float("-inf")))


def test_scale_logits_rejects_non_positive_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        scale_logits((1.0, 2.0), 0.0)


def test_softmax_sums_to_one() -> None:
    probs = softmax((1.0, 2.0, 3.0))
    assert sum(probs) == pytest.approx(1.0)
    assert all(math.isfinite(value) for value in probs)
