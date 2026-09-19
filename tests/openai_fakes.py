"""Mock OpenAI Chat Completions objects. No network and no live API key."""

from __future__ import annotations

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any
from uuid import UUID

from llmfr.core.schema import (
    Environment,
    Event,
    GenerationConfig,
    LogitsCapture,
    ModelConfig,
    RunMetadata,
    TokenContext,
    TopKCandidate,
    Trace,
)
from llmfr.core.version import DEFAULT_TOP_K, SCHEMA_VERSION
from tests.factories import FIXED_TIME, TRACE_ID, TRACE_ID_B, _event_id

LogprobToken = tuple[str, float, list[tuple[str, float]]]


class FakeEncoding:
    """Deterministic tokenizer stand-in so tests do not import tiktoken."""

    name = "test-encoding"

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}
        self._tokens: dict[int, str] = {}
        self._next = 256

    def encode(self, text: str, **_kwargs: object) -> list[int]:
        if not text:
            return []
        return [self.encode_single_token(ch) for ch in text]

    def decode(self, ids: Sequence[int], **_kwargs: object) -> str:
        parts: list[str] = []
        for token_id in ids:
            key = int(token_id)
            if key in self._tokens:
                parts.append(self._tokens[key])
            elif 0 <= key < 128:
                parts.append(chr(key))
            else:
                parts.append(f"<{key}>")
        return "".join(parts)

    def encode_single_token(self, token: str) -> int:
        if token in self._ids:
            return self._ids[token]
        if len(token) == 1 and ord(token) < 128:
            token_id = ord(token)
        else:
            token_id = self._next
            self._next += 1
        self._ids[token] = token_id
        self._tokens[token_id] = token
        return token_id


class FakeOpenAIClient:
    """`chat.completions.create` stand-in. Records kwargs for assertions."""

    def __init__(
        self,
        responses: list[Any] | None = None,
        *,
        errors: list[Exception] | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.errors = list(errors or [])
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        if not self.responses:
            raise RuntimeError("no fake OpenAI responses left")
        return self.responses.pop(0)


def make_chat_response(
    tokens: Sequence[LogprobToken],
    *,
    output_text: str | None = None,
    include_logprobs: bool = True,
) -> SimpleNamespace:
    content = None
    if include_logprobs:
        content = [
            SimpleNamespace(
                token=token,
                logprob=logprob,
                top_logprobs=[SimpleNamespace(token=alt, logprob=alt_lp) for alt, alt_lp in top],
            )
            for token, logprob, top in tokens
        ]
    text = output_text if output_text is not None else "".join(token for token, _lp, _top in tokens)
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                logprobs=None if content is None else SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ]
    )


def hello_logprob_tokens() -> tuple[LogprobToken, LogprobToken]:
    return (
        (
            "He",
            -0.1,
            [("He", -0.1), ("Hi", -1.2), ("ho", -2.0)],
        ),
        (
            "llo",
            -0.2,
            [("llo", -0.2), ("y", -1.5), ("lp", -2.5)],
        ),
    )


def make_openai_trace(
    *,
    k: int = DEFAULT_TOP_K,
    logits_mode: str = "topk",
    event_tokens: tuple[tuple[int, str, float], ...] = ((101, "He", -0.1), (102, "llo", -0.2)),
    trace_id: UUID | None = None,
    prompt: str = "Hi",
    prompt_token_ids: list[int] | None = None,
    generation: GenerationConfig | None = None,
    output_text: str | None = None,
) -> Trace:
    """OpenAI-shaped v1 trace: logprob-only top-k, never a vocab logit vector."""
    none_mode = logits_mode == "none"
    ids = [72, 105] if prompt_token_ids is None else list(prompt_token_ids)
    logits = (
        LogitsCapture(
            mode="none",
            unavailable_reason=(
                "OpenAI response did not include top_logprobs; refusing to invent them"
            ),
        )
        if none_mode
        else LogitsCapture(mode="topk", k=k)
    )
    prior: list[tuple[int, str]] = []
    events: list[Event] = []
    for i, (token_id, token, logprob) in enumerate(event_tokens):
        token_ids = ids + [item[0] for item in prior]
        text = prompt + "".join(item[1] for item in prior)
        ctx = TokenContext(token_ids=token_ids, text=text, truncated=False)
        top_k: list[TopKCandidate] = []
        if not none_mode:
            for rank in range(1, k + 1):
                cand_id = token_id if rank == 1 else 200 + rank
                cand_token = token if rank == 1 else f"alt{rank}"
                cand_lp = logprob if rank == 1 else logprob - rank
                top_k.append(
                    TopKCandidate(
                        rank=rank,
                        token_id=cand_id,
                        token=cand_token,
                        logit=None,
                        logprob=cand_lp,
                        prob=min(1.0, max(0.0, math.exp(cand_lp))),
                    )
                )
        events.append(
            Event(
                event_id=_event_id(i),
                step=i,
                sampled_token_id=token_id,
                sampled_token=token,
                sampled_logit=None,
                sampled_prob=None if none_mode else min(1.0, max(0.0, math.exp(logprob))),
                sampled_logprob=None if none_mode else logprob,
                sampled_rank=None if none_mode else 1,
                top_k=top_k,
                full_history=ctx,
                model_visible_context=ctx,
            )
        )
        prior.append((token_id, token))
    sampled = "".join(token for _token_id, token, _lp in event_tokens)
    return Trace(
        schema_version=SCHEMA_VERSION,
        run_metadata=RunMetadata(
            trace_id=TRACE_ID if trace_id is None else trace_id,
            created_at=FIXED_TIME,
            prompt=prompt,
            prompt_token_ids=ids,
            output_text=sampled if output_text is None else output_text,
            source="test",
            tags={"provider": "openai"},
            logits=logits,
        ),
        environment=Environment(device="api", library_versions={"openai": "1.0"}),
        model=ModelConfig(provider="openai", name="gpt-4o-mini", tokenizer="test-encoding"),
        generation_config=generation
        if generation is not None
        else GenerationConfig(temperature=0.0, max_new_tokens=8, do_sample=False),
        events=events,
    )


def alt_openai_trace() -> Trace:
    return make_openai_trace(
        trace_id=TRACE_ID_B,
        event_tokens=((109, "No", -0.3), (102, "llo", -0.2)),
    )
