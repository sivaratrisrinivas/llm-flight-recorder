"""Optional redaction and persist control. Local-first write remains the default.

Callers can pass a ``Redactor`` into ``record_generation`` to rewrite sensitive
string fields before a store write, and/or set ``persist=False`` to skip disk.
This module does not invent logits and does not send traces anywhere.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from llmfr.core.schema import Event, RunMetadata, Trace

Redactor = Callable[[Trace], Trace]

REDACTED = "[REDACTED]"

DEFAULT_REDACT_FIELDS: tuple[str, ...] = (
    "prompt",
    "output_text",
    "full_history.text",
    "model_visible_context.text",
)

_ALLOWED_REDACT_FIELDS = frozenset(
    {
        *DEFAULT_REDACT_FIELDS,
        "sampled_token",
        "tags",
    }
)


def redact_trace(
    trace: Trace,
    *,
    fields: Sequence[str] | None = None,
    replacement: str = REDACTED,
) -> Trace:
    """Return a copy with selected string fields replaced.

    Token ids, logits, and scores are left intact so replay and compare can
    still run on the redacted file. Unknown field names are a usage error.
    Empty strings and ``None`` text stay as recorded (nothing to hide).
    """
    selected = tuple(DEFAULT_REDACT_FIELDS if fields is None else fields)
    unknown = [name for name in selected if name not in _ALLOWED_REDACT_FIELDS]
    if unknown:
        raise ValueError(f"unknown redact field(s): {', '.join(unknown)}")
    chosen = set(selected)
    meta = _redact_metadata(trace.run_metadata, chosen, replacement)
    events = [_redact_event(event, chosen, replacement) for event in trace.events]
    return trace.model_copy(update={"run_metadata": meta, "events": events})


def _redact_metadata(meta: RunMetadata, chosen: set[str], replacement: str) -> RunMetadata:
    updates: dict[str, object] = {}
    if "prompt" in chosen:
        updates["prompt"] = _replace_text(meta.prompt, replacement)
    if "output_text" in chosen:
        updates["output_text"] = _replace_text(meta.output_text, replacement)
    if "tags" in chosen:
        updates["tags"] = {key: replacement for key in meta.tags}
    if not updates:
        return meta
    return meta.model_copy(update=updates)


def _redact_event(event: Event, chosen: set[str], replacement: str) -> Event:
    updates: dict[str, object] = {}
    if "sampled_token" in chosen:
        updates["sampled_token"] = _replace_text(event.sampled_token, replacement)
    history = event.full_history
    if "full_history.text" in chosen and history.text is not None:
        updates["full_history"] = history.model_copy(
            update={"text": _replace_text(history.text, replacement)}
        )
    visible = event.model_visible_context
    if "model_visible_context.text" in chosen and visible.text is not None:
        updates["model_visible_context"] = visible.model_copy(
            update={"text": _replace_text(visible.text, replacement)}
        )
    if not updates:
        return event
    return event.model_copy(update=updates)


def _replace_text(value: str, replacement: str) -> str:
    if value == "":
        return value
    return replacement
