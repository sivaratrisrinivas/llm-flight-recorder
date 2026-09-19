"""Model adapter protocol and capability flags.

Adapters turn a model-visible token prefix into whatever the backend can
honestly expose. Hugging Face causal LMs return raw next-token logits.
Hosted HTTP APIs often cannot. Capability flags record that difference so the
recorder (M3) can store `logits.mode=topk` or `logits.mode=none` without
inventing scores.
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
class StepLogits:
    """One forward pass over model-visible context.

    `token_ids` are the ids actually fed to the model (after any left clip).
    `requested_token_ids` is the prefix the caller passed, mapping to
    `Event.full_history`. `logits` is the last-position raw vocabulary vector
    from the model, not a top-k slice and not a stored Trace blob.
    """

    token_ids: tuple[int, ...]
    logits: tuple[float, ...]
    truncated: bool
    requested_token_ids: tuple[int, ...]
    logprobs: tuple[float, ...] | None = None

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
        """Ranked human-readable top-k from this step's real logits."""
        if k < 1:
            raise ValueError("k must be >= 1")
        if not self.logits:
            raise ValueError("no logits")
        take = min(k, MAX_TOP_K, len(self.logits))
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
