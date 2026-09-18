"""Human-readable top-k logits. Never dumps a full vocabulary."""

from __future__ import annotations

from llmfr.core.schema import Event, Trace


def format_event_topk(event: Event) -> str:
    sampled = f"step {event.step}  sampled {event.sampled_token!r} (id={event.sampled_token_id})"
    hist = len(event.full_history.token_ids)
    visible = len(event.model_visible_context.token_ids)
    context = f"  context full_history={hist} tokens, model_visible={visible} tokens"
    if event.model_visible_context.truncated:
        context += " (truncated)"
    if not event.top_k:
        return sampled + "\n" + context + "\n  (no top-k logits)"
    lines = [sampled, context]
    for candidate in event.top_k:
        marker = "*" if candidate.token_id == event.sampled_token_id else " "
        score = (
            _fmt_score("logit", candidate.logit)
            + _fmt_score("prob", candidate.prob)
            + _fmt_score("logprob", candidate.logprob)
        )
        lines.append(
            f" {marker}{candidate.rank:>2}. {candidate.token!r:<16} "
            f"id={candidate.token_id:<8}{score}"
        )
    return "\n".join(lines)


def format_trace_topk(trace: Trace) -> str:
    meta = trace.run_metadata
    header = [
        f"trace {meta.trace_id}",
        f"model {trace.model.provider}:{trace.model.name}",
        f"schema {trace.schema_version}",
    ]
    if meta.logits.mode == "none":
        header.append(f"logits unavailable: {meta.logits.unavailable_reason}")
        return "\n".join(header) + "\n"
    header.append(f"logits top-k (k={meta.logits.k})")
    body = [format_event_topk(event) for event in trace.events]
    return "\n".join(header + [""] + body) + "\n"


def _fmt_score(name: str, value: float | None) -> str:
    if value is None:
        return ""
    return f"  {name}={value: .4f}"
