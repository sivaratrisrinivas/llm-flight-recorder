"""Token-by-token generation recorder (Milestone 3).

Each step, in order:
  full_history -> model_visible_context -> raw logits -> temperature logits
  -> probabilities -> sample -> chosen token -> append -> next step

Windowing is the adapter's left-clip. Event contexts must satisfy the M1
schema: untruncated visible == full_history; truncated visible is a strict
suffix.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from llmfr.adapters.base import AdapterCapabilities, HostedCompletion, StepLogits
from llmfr.core.schema import (
    Environment,
    Event,
    GenerationConfig,
    LogitsCapture,
    ModelConfig,
    RunMetadata,
    TokenContext,
    Trace,
)
from llmfr.core.version import DEFAULT_TOP_K, MAX_TOP_K, SCHEMA_VERSION
from llmfr.privacy import Redactor
from llmfr.record.sample import (
    LocalRNG,
    choose_token,
    effective_generation_config,
    is_greedy,
    rank_of,
)
from llmfr.storage.store import FormatName, TraceStore

RECORDER_PIPELINE = (
    "full_history",
    "model_visible_context",
    "raw_logits",
    "temperature_logits",
    "probabilities",
    "sample",
    "append",
)


@runtime_checkable
class RecordableAdapter(Protocol):
    """ModelAdapter plus the metadata the recorder writes onto a Trace."""

    @property
    def capabilities(self) -> AdapterCapabilities: ...

    @property
    def model_config(self) -> ModelConfig: ...

    def environment(self) -> Environment: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...

    def decode_token(self, token_id: int) -> str: ...

    def next_token_logits(self, model_visible_context: Sequence[int]) -> StepLogits: ...


def record_generation(
    adapter: RecordableAdapter,
    prompt: str,
    *,
    generation: GenerationConfig,
    store: TraceStore | None = None,
    capture_k: int = DEFAULT_TOP_K,
    fmt: FormatName = "jsonl",
    source: str | None = "llmfr.record",
    tags: dict[str, str] | None = None,
    persist: bool = True,
    redact: Redactor | None = None,
) -> Trace:
    """Run the decode loop, build a Trace, and optionally persist it.

    ``persist=True`` (default) writes to ``store`` when one is given.
    ``persist=False`` skips the disk write even if ``store`` is passed.
    ``redact`` runs after the Trace is built and before any write.
    """
    _reject_unsupported(generation)
    if capture_k < 1 or capture_k > MAX_TOP_K:
        raise ValueError(f"capture_k must be in 1..{MAX_TOP_K}")
    max_new = generation.max_new_tokens
    if max_new is None or max_new < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if adapter.capabilities.supports_logits:
        if generation.seed is not None and not adapter.capabilities.supports_seed:
            raise ValueError("adapter does not support seed; refusing to pretend that it does")
        trace = _record_local(
            adapter,
            prompt,
            generation=generation,
            capture_k=capture_k,
            source=source,
            tags=tags,
        )
    else:
        complete = getattr(adapter, "complete_prompt", None)
        if not callable(complete):
            raise ValueError("recorder requires real next-token logits; refusing to invent them")
        trace = _record_hosted(
            adapter,
            prompt,
            generation=generation,
            capture_k=capture_k,
            source=source,
            tags=tags,
        )
    if redact is not None:
        trace = redact(trace)
    if persist and store is not None:
        store.put(trace, fmt=fmt)
    return trace


def _record_local(
    adapter: RecordableAdapter,
    prompt: str,
    *,
    generation: GenerationConfig,
    capture_k: int,
    source: str | None,
    tags: dict[str, str] | None,
) -> Trace:
    prompt_ids = adapter.encode(prompt)
    if not prompt_ids:
        raise ValueError("prompt encoded to no tokens")

    greedy = is_greedy(generation)
    rng = LocalRNG(generation.seed)
    history = list(prompt_ids)
    sampled_ids: list[int] = []
    events: list[Event] = []
    max_new = generation.max_new_tokens
    assert max_new is not None

    for step_index in range(max_new):
        step_logits = adapter.next_token_logits(history)
        history_ctx, visible_ctx = _contexts_with_text(adapter, step_logits)
        decision = choose_token(
            step_logits.logits,
            temperature=generation.temperature,
            greedy=greedy,
            rng=None if greedy else rng,
            sampler_top_k=generation.top_k,
        )
        token_id = decision.sampled_token_id
        token = adapter.decode_token(token_id)
        sampled_logit = float(step_logits.logits[token_id])
        sampled_logprob, sampled_prob = _model_scores(step_logits, token_id)
        events.append(
            Event(
                step=step_index,
                sampled_token_id=token_id,
                sampled_token=token,
                sampled_logit=sampled_logit,
                sampled_prob=sampled_prob,
                sampled_logprob=sampled_logprob,
                sampled_rank=rank_of(token_id, step_logits.logits),
                top_k=step_logits.top_k_candidates(capture_k, decode=adapter.decode_token),
                full_history=history_ctx,
                model_visible_context=visible_ctx,
            )
        )
        sampled_ids.append(token_id)
        history.append(token_id)
        if _hit_stop(adapter.decode(sampled_ids), generation.stop):
            break

    return Trace(
        schema_version=SCHEMA_VERSION,
        run_metadata=RunMetadata(
            prompt=prompt,
            prompt_token_ids=list(prompt_ids),
            output_text=adapter.decode(sampled_ids),
            source=source,
            tags={} if tags is None else dict(tags),
            logits=LogitsCapture(mode="topk", k=capture_k),
        ),
        environment=adapter.environment(),
        model=adapter.model_config,
        generation_config=effective_generation_config(generation),
        events=events,
    )


def _record_hosted(
    adapter: RecordableAdapter,
    prompt: str,
    *,
    generation: GenerationConfig,
    capture_k: int,
    source: str | None,
    tags: dict[str, str] | None,
) -> Trace:
    """Record API-sampled tokens. Never run LocalRNG over invented logits."""
    complete = getattr(adapter, "complete_prompt", None)
    if not callable(complete):
        raise ValueError("recorder requires real next-token logits; refusing to invent them")
    completion: HostedCompletion = complete(prompt, generation=generation, capture_k=capture_k)
    prompt_ids = list(completion.prompt_token_ids)
    if not prompt_ids:
        raise ValueError("prompt encoded to no tokens")
    if not completion.steps:
        raise RuntimeError("hosted adapter returned no tokens; refusing to invent them")

    if not _hosted_has_real_scores(completion):
        reason = completion.unavailable_reason or (
            "hosted API did not expose real logprobs; refusing to invent them"
        )
        raise RuntimeError(reason)
    stored_k = completion.capture_k if completion.capture_k is not None else capture_k
    logits = LogitsCapture(mode="topk", k=stored_k)

    history_ids = list(prompt_ids)
    history_text = prompt
    events: list[Event] = []
    sampled_tokens: list[str] = []
    for step_index, step in enumerate(completion.steps):
        history_ctx = TokenContext(
            token_ids=list(history_ids),
            text=history_text,
            truncated=False,
        )
        sampled_logprob = None if step.logprob is None else float(step.logprob)
        sampled_prob = None
        if sampled_logprob is not None:
            sampled_prob = min(1.0, max(0.0, math.exp(sampled_logprob)))
        events.append(
            Event(
                step=step_index,
                sampled_token_id=step.token_id,
                sampled_token=step.token,
                sampled_logit=None,
                sampled_prob=sampled_prob,
                sampled_logprob=sampled_logprob,
                sampled_rank=step.rank,
                top_k=list(step.top_k),
                full_history=history_ctx,
                model_visible_context=history_ctx,
            )
        )
        sampled_tokens.append(step.token)
        history_ids.append(step.token_id)
        history_text = history_text + step.token

    output_text = completion.output_text if completion.output_text else "".join(sampled_tokens)
    return Trace(
        schema_version=SCHEMA_VERSION,
        run_metadata=RunMetadata(
            prompt=prompt,
            prompt_token_ids=prompt_ids,
            output_text=output_text,
            source=source,
            tags={} if tags is None else dict(tags),
            logits=logits,
        ),
        environment=adapter.environment(),
        model=adapter.model_config,
        generation_config=effective_generation_config(generation),
        events=events,
    )


def _hosted_has_real_scores(completion: HostedCompletion) -> bool:
    """True when the API returned any real logprob or top-k row.

    Per-token logprob with empty ``top_logprobs`` still counts. A completion
    with no scores does not; the hosted path fails closed instead of storing
    reconstructed token steps.
    """
    return any(step.logprob is not None or step.top_k for step in completion.steps)


def _reject_unsupported(generation: GenerationConfig) -> None:
    if generation.top_p is not None:
        raise ValueError("top_p sampling is not implemented in Milestone 3")
    if generation.repetition_penalty is not None:
        raise ValueError("repetition_penalty is not implemented in Milestone 3")


def _contexts_with_text(
    adapter: RecordableAdapter, step: StepLogits
) -> tuple[TokenContext, TokenContext]:
    history, visible = step.as_contexts()
    history = history.model_copy(update={"text": adapter.decode(history.token_ids)})
    visible = visible.model_copy(update={"text": adapter.decode(visible.token_ids)})
    return history, visible


def _model_scores(step: StepLogits, token_id: int) -> tuple[float | None, float | None]:
    """Probabilities from the model distribution (raw logits), not the sampler."""
    if step.logprobs is not None:
        logprob = float(step.logprobs[token_id])
        return logprob, min(1.0, max(0.0, math.exp(logprob)))
    return None, None


def _hit_stop(output_text: str, stop: list[str] | None) -> bool:
    if not stop:
        return False
    return any(token and token in output_text for token in stop)
