"""Deterministic adapter stand-in for recorder tests. No torch."""

from __future__ import annotations

import math
from collections.abc import Sequence

from llmfr.adapters.base import AdapterCapabilities, StepLogits
from llmfr.core.schema import Environment, ModelConfig
from llmfr.record.sample import softmax


def _log_softmax(logits: tuple[float, ...]) -> tuple[float, ...]:
    probs = softmax(logits)
    return tuple(math.log(prob) if prob > 0.0 else float("-inf") for prob in probs)


class FakeCausalLMAdapter:
    """Peaked or scripted logits over a tiny vocab. Clips like the HF adapter."""

    def __init__(
        self,
        *,
        logits: tuple[float, ...] = (0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        logits_for_prefix: dict[tuple[int, ...], tuple[float, ...]] | None = None,
        max_visible_tokens: int = 1024,
        prompt_ids: list[int] | None = None,
        supports_seed: bool = True,
        supports_logits: bool = True,
        supports_replay: bool = True,
        name: str = "fake-lm",
        revision: str | None = None,
    ) -> None:
        self._logits = logits
        self._logits_for_prefix = logits_for_prefix or {}
        self._max_visible_tokens = max_visible_tokens
        self._prompt_ids = prompt_ids
        self._supports_seed = supports_seed
        self._supports_logits = supports_logits
        self._supports_replay = supports_replay
        self._name = name
        self._revision = revision
        self.prefixes: list[tuple[int, ...]] = []

    @property
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            supports_logits=self._supports_logits,
            supports_logprobs=True,
            supports_attention=False,
            supports_hidden_states=False,
            supports_seed=self._supports_seed,
            supports_replay=self._supports_replay,
        )

    @property
    def model_config(self) -> ModelConfig:
        return ModelConfig(provider="test", name=self._name, revision=self._revision)

    @property
    def vocab_size(self) -> int:
        return len(self._logits)

    @property
    def max_visible_tokens(self) -> int:
        return self._max_visible_tokens

    def environment(self) -> Environment:
        return Environment(device="cpu", accelerator="cpu")

    def encode(self, text: str) -> list[int]:
        if self._prompt_ids is not None:
            return list(self._prompt_ids)
        if not text:
            return []
        return [index % self.vocab_size for index, _ch in enumerate(text)]

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(self.decode_token(token_id) for token_id in token_ids)

    def decode_token(self, token_id: int) -> str:
        return f"t{int(token_id)}"

    def next_token_logits(self, model_visible_context: Sequence[int]) -> StepLogits:
        requested = tuple(int(token_id) for token_id in model_visible_context)
        if not requested:
            raise ValueError("model_visible_context must be non-empty")
        self.prefixes.append(requested)
        visible, truncated = self._clip(requested)
        raw = self._logits_for_prefix.get(requested, self._logits)
        if len(raw) != self.vocab_size:
            raise ValueError("scripted logits must match vocab size")
        return StepLogits(
            token_ids=visible,
            requested_token_ids=requested,
            logits=raw,
            logprobs=_log_softmax(raw),
            truncated=truncated,
        )

    def _clip(self, token_ids: tuple[int, ...]) -> tuple[tuple[int, ...], bool]:
        limit = self._max_visible_tokens
        if len(token_ids) <= limit:
            return token_ids, False
        return token_ids[-limit:], True
