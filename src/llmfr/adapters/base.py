"""Model adapter protocol and capability flags.

Adapters turn a model-visible token prefix into whatever the backend can
honestly expose. Hugging Face causal LMs return raw next-token logits.
Hosted HTTP APIs often cannot: OpenAI Chat Completions may return
`top_logprobs` (a short ranked list of logprobs, not a vocabulary vector).
Capability flags record that difference so the recorder (M3) can store
`logits.mode=topk` or `logits.mode=none` without inventing scores.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from llmfr.core.schema import TokenContext, TopKCandidate
from llmfr.core.version import MAX_TOP_K


@dataclass(frozen=True)
class AdapterCapabilities:
    """What this adapter can actually return. Do not set a flag the stack lacks."""

    supports_logits: bool
    supports_logprobs: bool
    supports_attention: bool
    supports_hidden_states: bool
    supports_seed: bool
    supports_replay: bool


@dataclass(frozen=True)
class HostedTokenStep:
    """One API-sampled token plus whatever scores the backend actually returned."""

    token_id: int
    token: str
    logprob: float | None
    top_k: tuple[TopKCandidate, ...]
    rank: int | None = None


@dataclass(frozen=True)
class HostedCompletion:
    """A hosted generation recorded without a local full-vocab logit vector.

    `logprobs_available` is True only when the API returned real top_logprobs
    (or per-token logprobs). It is never inferred from sampled text.
    """

    prompt_token_ids: tuple[int, ...]
    output_text: str
    steps: tuple[HostedTokenStep, ...]
    logprobs_available: bool
    unavailable_reason: str | None = None
    capture_k: int | None = None


@dataclass(frozen=True)
class StepLogits:
    """One forward pass over model-visible context.

    `token_ids` are the ids actually fed to the model (after any left clip).
    `requested_token_ids` is the prefix the caller passed, mapping to
    `Event.full_history`. `logits` is the last-position raw vocabulary vector
    from the model, not a top-k slice and not a stored Trace blob. Hosted APIs
    that only return `top_logprobs` leave `logits` empty and fill
    `top_logprobs` instead. Do not pad a sparse list into a vocab vector.
    """

    token_ids: tuple[int, ...]
    logits: tuple[float, ...]
    truncated: bool
    requested_token_ids: tuple[int, ...]
    logprobs: tuple[float, ...] | None = None
    top_logprobs: tuple[TopKCandidate, ...] | None = None
    backend_sampled_token_id: int | None = None
    backend_sampled_token: str | None = None
    backend_sampled_logprob: float | None = None

    def as_contexts(self) -> tuple[TokenContext, TokenContext]:
        """Map this step onto M1 `full_history` and `model_visible_context`."""
        history = TokenContext(token_ids=list(self.requested_token_ids), truncated=False)
        visible = TokenContext(token_ids=list(self.token_ids), truncated=self.truncated)
        return history, visible

    @property
    def greedy_token_id(self) -> int:
        if not self.logits:
            raise ValueError("no logits")
        return max(range(len(self.logits)), key=self.logits.__getitem__)

    def top_k_candidates(
        self,
        k: int,
        *,
        decode: Callable[[int], str],
    ) -> list[TopKCandidate]:
        """Ranked human-readable top-k from real logits or captured top_logprobs."""
        if k < 1:
            raise ValueError("k must be >= 1")
        if k > MAX_TOP_K:
            raise ValueError(
                f"k must be <= {MAX_TOP_K}; refusing to store an oversized logit payload"
            )
        if self.logits:
            return self._topk_from_logits(k, decode=decode)
        if self.top_logprobs is not None:
            return list(self.top_logprobs[: min(k, len(self.top_logprobs))])
        raise ValueError("no logits")

    def _topk_from_logits(
        self,
        k: int,
        *,
        decode: Callable[[int], str],
    ) -> list[TopKCandidate]:
        take = min(k, len(self.logits))
        top_ids = sorted(
            range(len(self.logits)),
            key=self.logits.__getitem__,
            reverse=True,
        )[:take]
        candidates: list[TopKCandidate] = []
        for rank, token_id in enumerate(top_ids, start=1):
            logit = float(self.logits[token_id])
            logprob = None if self.logprobs is None else float(self.logprobs[token_id])
            prob = None
            if logprob is not None:
                prob = min(1.0, max(0.0, math.exp(logprob)))
            candidates.append(
                TopKCandidate(
                    rank=rank,
                    token_id=token_id,
                    token=decode(token_id),
                    logit=logit,
                    prob=prob,
                    logprob=logprob,
                )
            )
        return candidates


@runtime_checkable
class ModelAdapter(Protocol):
    @property
    def capabilities(self) -> AdapterCapabilities: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...

    def next_token_logits(self, model_visible_context: Sequence[int]) -> StepLogits: ...
