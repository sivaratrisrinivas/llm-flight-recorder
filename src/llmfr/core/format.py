"""Human-readable top-k logits and inspect views. Never dumps a full vocabulary."""

from __future__ import annotations

from typing import Any

from llmfr.core.schema import Event, TokenContext, Trace


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


def format_inspect_overview(trace: Trace) -> str:
    """Trace-level inspect view. Use format_inspect_step for one decode step."""

    meta = trace.run_metadata
    gen = trace.generation_config
    model = trace.model
    model_line = f"model {model.provider}:{model.name}"
    if model.revision is not None:
        model_line += f" revision={model.revision}"
    lines = [
        f"trace {meta.trace_id}",
        f"schema {trace.schema_version}",
        model_line,
    ]
    if model.tokenizer is not None:
        lines.append(f"tokenizer {model.tokenizer}")
    lines.append(
        "generation"
        f" temperature={gen.temperature}"
        f" do_sample={gen.do_sample}"
        f" seed={gen.seed}"
        f" max_new_tokens={gen.max_new_tokens}"
    )
    logits = meta.logits
    if logits.mode == "none":
        lines.append(f"logits unavailable: {logits.unavailable_reason}")
    else:
        lines.append(f"logits top-k (k={logits.k})")
    lines.append(f"prompt {meta.prompt!r}")
    lines.append(f"output {meta.output_text!r}")
    lines.append(f"events {len(trace.events)}")
    if not trace.events:
        lines.append("sampled tokens: (none)")
    else:
        lines.append("sampled tokens:")
        for event in trace.events:
            lines.append(f"  {event.step}: {event.sampled_token!r} (id={event.sampled_token_id})")
        last = len(trace.events) - 1
        lines.append("")
        lines.append(f"inspect a step: llmfr inspect {meta.trace_id} --step N  (N is 0..{last})")
    return "\n".join(lines) + "\n"


def format_inspect_step(trace: Trace, step: int) -> str:
    """One recorded step: history vs visible context, sampled token, and top-k."""

    event = trace.events[step]
    meta = trace.run_metadata
    lines = [
        f"trace {meta.trace_id}  step {step} of {len(trace.events)}",
        "",
        "sampled token",
        f"  {event.sampled_token!r} (id={event.sampled_token_id})",
        f"  {_sampled_scores(event)}",
        "",
        "full_history",
        *_format_context(event.full_history),
        "",
        "model_visible_context",
        *_format_context(event.model_visible_context),
        f"  {_visible_relation(event)}",
        "",
        "top-k",
    ]
    if meta.logits.mode == "none":
        lines.append(f"  logits unavailable: {meta.logits.unavailable_reason}")
        lines.append("  (no invented logits)")
    elif not event.top_k:
        lines.append("  (no top-k logits)")
    else:
        lines.extend(_format_topk_rows(event))
    return "\n".join(lines) + "\n"


def inspect_overview_payload(trace: Trace) -> dict[str, Any]:
    meta = trace.run_metadata
    return {
        "trace_id": str(meta.trace_id),
        "schema_version": trace.schema_version,
        "model": trace.model.model_dump(mode="json"),
        "generation_config": trace.generation_config.model_dump(mode="json"),
        "logits": meta.logits.model_dump(mode="json"),
        "prompt": meta.prompt,
        "output_text": meta.output_text,
        "event_count": len(trace.events),
        "sampled_tokens": [
            {
                "step": event.step,
                "token": event.sampled_token,
                "token_id": event.sampled_token_id,
            }
            for event in trace.events
        ],
    }


def inspect_step_payload(trace: Trace, step: int) -> dict[str, Any]:
    event = trace.events[step]
    meta = trace.run_metadata
    return {
        "trace_id": str(meta.trace_id),
        "step": step,
        "event_count": len(trace.events),
        "sampled_token": event.sampled_token,
        "sampled_token_id": event.sampled_token_id,
        "sampled_logit": event.sampled_logit,
        "sampled_prob": event.sampled_prob,
        "sampled_logprob": event.sampled_logprob,
        "sampled_rank": event.sampled_rank,
        "full_history": _context_payload(event.full_history),
        "model_visible_context": _context_payload(event.model_visible_context),
        "visible_equals_history": (
            event.model_visible_context.token_ids == event.full_history.token_ids
            and not event.model_visible_context.truncated
        ),
        "top_k": [candidate.model_dump(mode="json") for candidate in event.top_k],
        "logits": meta.logits.model_dump(mode="json"),
    }


def _format_context(context: TokenContext) -> list[str]:
    lines = [
        f"  tokens={len(context.token_ids)}  truncated={str(context.truncated).lower()}",
        f"  ids={context.token_ids}",
    ]
    if context.text is None:
        lines.append("  text=None")
    else:
        lines.append(f"  text={context.text!r}")
    return lines


def _context_payload(context: TokenContext) -> dict[str, Any]:
    return {
        "token_ids": list(context.token_ids),
        "text": context.text,
        "truncated": context.truncated,
        "token_count": len(context.token_ids),
    }


def _visible_relation(event: Event) -> str:
    if event.model_visible_context.truncated:
        return "left-truncated suffix of full_history (model did not see the full prefix)"
    return "equals full_history (model saw the full prefix)"


def _sampled_scores(event: Event) -> str:
    parts: list[str] = []
    if event.sampled_rank is not None:
        parts.append(f"rank={event.sampled_rank}")
    for name, value in (
        ("logit", event.sampled_logit),
        ("prob", event.sampled_prob),
        ("logprob", event.sampled_logprob),
    ):
        if value is not None:
            parts.append(f"{name}={value: .4f}")
    if not parts:
        return "scores not captured"
    return "  ".join(parts)


def _format_topk_rows(event: Event) -> list[str]:
    lines: list[str] = []
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
    return lines


def _fmt_score(name: str, value: float | None) -> str:
    if value is None:
        return ""
    return f"  {name}={value: .4f}"
