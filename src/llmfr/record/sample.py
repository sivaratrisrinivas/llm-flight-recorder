"""Sampling math for the M3 recorder.

Order: raw logits -> temperature-adjusted logits -> probabilities -> sample.
Uses an isolated ``random.Random`` so a seed does not call ``torch.manual_seed``
or reseed the process-global ``random`` module.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from llmfr.core.schema import GenerationConfig

# Named so tests can assert the inner pipeline without a model.
SAMPLING_STAGES = (
    "raw_logits",
    "temperature_logits",
    "probabilities",
    "sample",
)


class UniformSource(Protocol):
    def random(self) -> float: ...


class LocalRNG:
    """Isolated uniform source for categorical draws.

    A seed here only affects this object. It does not reset global ``random``
    or ``torch`` state, so other code in the process cannot steal or scramble
    the sampler sequence. Reproducibility is for this RNG plus the logits it
    is given, not for a whole process that also draws from torch.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def random(self) -> float:
        return self._rng.random()


@dataclass(frozen=True)
class SampleDecision:
    """One pass through the sampling stages. ``stages`` is always SAMPLING_STAGES."""

    raw_logits: tuple[float, ...]
    temperature_logits: tuple[float, ...]
    probabilities: tuple[float, ...]
    sampled_token_id: int
    greedy: bool
    stages: tuple[str, ...] = SAMPLING_STAGES


def is_greedy(generation: GenerationConfig) -> bool:
    """Greedy when sampling is off or temperature is missing/non-positive."""
    if generation.do_sample is False:
        return True
    temperature = generation.temperature
    if temperature is not None and temperature <= 0:
        return True
    if generation.do_sample is True:
        return False
    return temperature is None


def softmax(logits: Sequence[float]) -> tuple[float, ...]:
    if not logits:
        raise ValueError("no logits")
    finite = [value for value in logits if math.isfinite(value)]
    if not finite:
        raise ValueError("softmax is undefined for an all-masked logit vector")
    peak = max(finite)
    shifted = [math.exp(value - peak) if math.isfinite(value) else 0.0 for value in logits]
    total = sum(shifted)
    if total == 0.0:
        raise ValueError("softmax is undefined for an all-masked logit vector")
    return tuple(value / total for value in shifted)


def scale_logits(raw: Sequence[float], temperature: float) -> tuple[float, ...]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0 to scale logits; use greedy instead")
    if temperature == 1.0:
        return tuple(float(value) for value in raw)
    return tuple(float(value) / temperature for value in raw)


def mask_top_k(logits: Sequence[float], k: int) -> tuple[float, ...]:
    if k < 1:
        raise ValueError("sampler top_k must be >= 1")
    if k >= len(logits):
        return tuple(float(value) for value in logits)
    keep = set(
        sorted(range(len(logits)), key=logits.__getitem__, reverse=True)[:k],
    )
    return tuple(
        float(value) if index in keep else float("-inf") for index, value in enumerate(logits)
    )


def categorical(probs: Sequence[float], rng: UniformSource) -> int:
    if not probs:
        raise ValueError("no probabilities")
    draw = rng.random()
    cumulative = 0.0
    last_positive = 0
    for index, prob in enumerate(probs):
        if prob > 0.0:
            last_positive = index
            cumulative += prob
            if draw < cumulative:
                return index
    return last_positive


def choose_token(
    raw_logits: Sequence[float],
    *,
    temperature: float | None,
    greedy: bool,
    rng: UniformSource | None = None,
    sampler_top_k: int | None = None,
) -> SampleDecision:
    """Walk raw logits -> temperature logits -> probabilities -> chosen token."""
    if not raw_logits:
        raise ValueError("no logits")
    use_greedy = greedy or (temperature is not None and temperature <= 0)
    greedy_id = max(range(len(raw_logits)), key=raw_logits.__getitem__)

    if temperature is None or temperature <= 0:
        scaled: tuple[float, ...] = tuple(float(value) for value in raw_logits)
    else:
        scaled = scale_logits(raw_logits, temperature)

    if sampler_top_k is not None and sampler_top_k > 0:
        scaled = mask_top_k(scaled, sampler_top_k)

    probabilities = softmax(scaled)
    if use_greedy:
        sampled = greedy_id
    else:
        if rng is None:
            raise ValueError("sampling requires an RNG")
        sampled = categorical(probabilities, rng)

    return SampleDecision(
        raw_logits=tuple(float(value) for value in raw_logits),
        temperature_logits=scaled,
        probabilities=probabilities,
        sampled_token_id=sampled,
        greedy=use_greedy,
    )


def rank_of(token_id: int, logits: Sequence[float]) -> int:
    if token_id < 0 or token_id >= len(logits):
        raise ValueError(f"token_id {token_id} is outside the logit vector")
    target = logits[token_id]
    return 1 + sum(1 for value in logits if value > target)
