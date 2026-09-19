from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid5

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

FIXED_TIME = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)
TRACE_ID = UUID("11111111-1111-4111-8111-111111111111")
TRACE_ID_B = UUID("11111111-1111-4111-8111-111111111112")
EVENT_IDS = (
    UUID("22222222-2222-4222-8222-222222222221"),
    UUID("22222222-2222-4222-8222-222222222222"),
)
PROMPT = "Say hello"
PROMPT_IDS = [1, 2, 3]
_EVENT_NS = UUID("33333333-3333-4333-8333-333333333333")


def make_candidates(k: int = DEFAULT_TOP_K, sampled_id: int = 101) -> list[TopKCandidate]:
    candidates: list[TopKCandidate] = []
    for rank in range(1, k + 1):
        token_id = sampled_id if rank == 1 else 200 + rank
        candidates.append(
            TopKCandidate(
                rank=rank,
                token_id=token_id,
                token="Hello" if rank == 1 else f"alt{rank}",
                logit=12.0 - rank,
                prob=0.9 if rank == 1 else None,
                logprob=-0.1 * rank if rank == 1 else None,
            )
        )
    return candidates


def _contexts(
    prior: list[tuple[int, str]],
    *,
    visible_limit: int | None,
    prompt: str,
    prompt_token_ids: list[int],
) -> tuple[TokenContext, TokenContext]:
    token_ids = list(prompt_token_ids) + [token_id for token_id, _ in prior]
    text = prompt + "".join(token for _, token in prior)
    full = TokenContext(token_ids=token_ids, text=text, truncated=False)
    if visible_limit is not None and visible_limit < len(token_ids):
        visible = TokenContext(token_ids=token_ids[-visible_limit:], text=None, truncated=True)
        return full, visible
    return full, TokenContext(token_ids=list(token_ids), text=text, truncated=False)


def _event_id(index: int) -> UUID:
    if index < len(EVENT_IDS):
        return EVENT_IDS[index]
    return uuid5(_EVENT_NS, str(index))


def make_trace(
    *,
    k: int = DEFAULT_TOP_K,
    logits_mode: str = "topk",
    event_tokens: tuple[tuple[int, str], ...] = ((101, "Hello"), (102, " world")),
    visible_limit: int | None = None,
    trace_id: UUID | None = None,
    prompt: str = PROMPT,
    prompt_token_ids: list[int] | None = None,
    generation: GenerationConfig | None = None,
    model: ModelConfig | None = None,
    output_text: str | None = None,
) -> Trace:
    prior: list[tuple[int, str]] = []
    events: list[Event] = []
    none_mode = logits_mode == "none"
    ids = list(PROMPT_IDS if prompt_token_ids is None else prompt_token_ids)
    logits = (
        LogitsCapture(
            mode="none",
            unavailable_reason="hosted API did not expose real logprobs; refusing to invent them",
        )
        if none_mode
        else LogitsCapture(mode="topk", k=k)
    )
    for i, (token_id, token) in enumerate(event_tokens):
        full, visible = _contexts(
            prior,
            visible_limit=visible_limit,
            prompt=prompt,
            prompt_token_ids=ids,
        )
        top_k = [] if none_mode else make_candidates(k=k, sampled_id=token_id)
        sampled = top_k[0] if top_k else None
        events.append(
            Event(
                event_id=_event_id(i),
                step=i,
                sampled_token_id=token_id,
                sampled_token=token,
                sampled_logit=None if sampled is None else sampled.logit,
                sampled_prob=None if sampled is None else sampled.prob,
                sampled_logprob=None if sampled is None else sampled.logprob,
                sampled_rank=None if sampled is None else 1,
                top_k=top_k,
                full_history=full,
                model_visible_context=visible,
            )
        )
        prior.append((token_id, token))
    return Trace(
        schema_version=SCHEMA_VERSION,
        run_metadata=RunMetadata(
            trace_id=TRACE_ID if trace_id is None else trace_id,
            created_at=FIXED_TIME,
            prompt=prompt,
            prompt_token_ids=ids,
            output_text="Hello world" if output_text is None else output_text,
            source="test",
            tags={"case": "G"},
            logits=logits,
        ),
        environment=Environment(
            python_version="3.12.3",
            platform="linux",
            device="cpu",
            library_versions={"pydantic": "2.13.5"},
        ),
        model=model
        if model is not None
        else ModelConfig(
            provider="test",
            name="tiny-model",
            revision="abc",
            tokenizer="tiny-tok",
            dtype="float32",
        ),
        generation_config=generation
        if generation is not None
        else GenerationConfig(
            temperature=0.0,
            max_new_tokens=8,
            do_sample=False,
            seed=1,
        ),
        events=events,
    )
