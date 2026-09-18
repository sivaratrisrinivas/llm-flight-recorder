from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from llmfr.schema import Event, LogitsCapture, ModelRef, SamplingConfig, TopKCandidate, Trace
from llmfr.version import DEFAULT_TOP_K, SCHEMA_VERSION

FIXED_TIME = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)
TRACE_ID = UUID("11111111-1111-4111-8111-111111111111")
EVENT_IDS = (
    UUID("22222222-2222-4222-8222-222222222221"),
    UUID("22222222-2222-4222-8222-222222222222"),
)


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
                logprob=-0.1 * rank if rank == 1 else None,
            )
        )
    return candidates


def make_trace(
    *,
    k: int = DEFAULT_TOP_K,
    logits_mode: str = "topk",
    event_tokens: tuple[tuple[int, str], ...] = ((101, "Hello"), (102, " world")),
) -> Trace:
    if logits_mode == "none":
        logits = LogitsCapture(
            mode="none",
            unavailable_reason="hosted API did not expose real logprobs; refusing to invent them",
        )
        events = [
            Event(event_id=EVENT_IDS[i], step=i, token_id=token_id, token=token, top_k=[])
            for i, (token_id, token) in enumerate(event_tokens)
        ]
    else:
        logits = LogitsCapture(mode="topk", k=k)
        events = []
        for i, (token_id, token) in enumerate(event_tokens):
            events.append(
                Event(
                    event_id=EVENT_IDS[i],
                    step=i,
                    token_id=token_id,
                    token=token,
                    top_k=make_candidates(k=k, sampled_id=token_id),
                )
            )
    return Trace(
        schema_version=SCHEMA_VERSION,
        trace_id=TRACE_ID,
        created_at=FIXED_TIME,
        model=ModelRef(provider="test", name="tiny-model", revision="abc", tokenizer="tiny-tok"),
        sampling=SamplingConfig(temperature=0.0, max_new_tokens=8, do_sample=False, seed=1),
        prompt="Say hello",
        prompt_token_ids=[1, 2, 3],
        output_text="Hello world",
        logits=logits,
        events=events,
        metadata={"case": "G"},
    )
