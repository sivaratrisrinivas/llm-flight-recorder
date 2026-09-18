"""Human-readable top-k logits. Never dumps a full vocabulary."""

from __future__ import annotations

from llmfr.schema import Event, Trace


def format_event_topk(event: Event) -> str:
    sampled = f"step {event.step}  sampled {event.token!r} (id={event.token_id})"
    if not event.top_k:
        return sampled + "\n  (no top-k logits)"
    lines = [sampled]
    for candidate in event.top_k:
        marker = "*" if candidate.token_id == event.token_id else " "
        logprob = f"  logprob={candidate.logprob: .4f}" if candidate.logprob is not None else ""
        lines.append(
            f" {marker}{candidate.rank:>2}. {candidate.token!r:<16} "
            f"id={candidate.token_id:<8} logit={candidate.logit: .4f}{logprob}"
        )
    return "\n".join(lines)


def format_trace_topk(trace: Trace) -> str:
    header = [
        f"trace {trace.trace_id}",
        f"model {trace.model.provider}:{trace.model.name}",
        f"schema {trace.schema_version}",
    ]
    if trace.logits.mode == "none":
        header.append(f"logits unavailable: {trace.logits.unavailable_reason}")
        return "\n".join(header) + "\n"
    header.append(f"logits top-k (k={trace.logits.k})")
    body = [format_event_topk(event) for event in trace.events]
    return "\n".join(header + [""] + body) + "\n"
