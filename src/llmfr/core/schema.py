"""Versioned Trace and Event schema (SCHEMA_VERSION 1.0.0)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llmfr.core.migrate import migrate_payload
from llmfr.core.version import MAX_TOP_K, SCHEMA_VERSION


def utcnow() -> datetime:
    return datetime.now(UTC)


class FrozenModel(BaseModel):
    # populate_by_name lets us construct Trace(model=...) while serializing
    # the field as model_config. Pydantic reserves the name model_config.
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class TopKCandidate(FrozenModel):
    """One alternative at a decode step. Human-readable, not a vocab blob.

    At least one of logit, prob, or logprob must be a real captured value.
    """

    rank: int = Field(ge=1, le=MAX_TOP_K)
    token_id: int
    token: str
    logit: float | None = None
    prob: float | None = Field(default=None, ge=0.0, le=1.0)
    logprob: float | None = None

    @model_validator(mode="after")
    def _has_a_real_score(self) -> Self:
        if self.logit is None and self.prob is None and self.logprob is None:
            raise ValueError("top-k candidate needs a real logit, prob, or logprob")
        return self


class TokenContext(FrozenModel):
    """A token prefix: either the full history or the window the model saw."""

    token_ids: list[int]
    text: str | None = None
    truncated: bool = False


class Event(FrozenModel):
    """One generation step.

    full_history is the complete prefix used to predict this token (prompt plus
    earlier sampled tokens). model_visible_context is what was actually fed to
    the model, which may be a left-truncated suffix of full_history.
    """

    event_id: UUID = Field(default_factory=uuid4)
    step: int = Field(ge=0)
    sampled_token_id: int
    sampled_token: str
    sampled_logit: float | None = None
    sampled_prob: float | None = Field(default=None, ge=0.0, le=1.0)
    sampled_logprob: float | None = None
    sampled_rank: int | None = Field(default=None, ge=1)
    top_k: list[TopKCandidate] = Field(default_factory=list, max_length=MAX_TOP_K)
    full_history: TokenContext
    model_visible_context: TokenContext

    @model_validator(mode="after")
    def _ranks_and_context(self) -> Self:
        ranks = [candidate.rank for candidate in self.top_k]
        expected = list(range(1, len(self.top_k) + 1))
        if ranks != expected:
            raise ValueError("top_k ranks must be 1..n in order")
        visible = self.model_visible_context
        history = self.full_history
        if visible.truncated:
            if not _is_suffix(visible.token_ids, history.token_ids):
                raise ValueError("truncated visible context must be a suffix of full_history")
            if len(visible.token_ids) >= len(history.token_ids):
                raise ValueError("truncated visible context must be shorter than full_history")
        elif visible.token_ids != history.token_ids:
            raise ValueError("untruncated visible context must equal full_history")
        return self


class Environment(FrozenModel):
    python_version: str | None = None
    platform: str | None = None
    device: str | None = None
    accelerator: str | None = None
    library_versions: dict[str, str] = Field(default_factory=dict)


class ModelConfig(FrozenModel):
    provider: str = Field(min_length=1)
    name: str = Field(min_length=1)
    revision: str | None = None
    tokenizer: str | None = None
    dtype: str | None = None
    architecture: str | None = None


class GenerationConfig(FrozenModel):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = Field(default=None, ge=0)
    max_new_tokens: int | None = Field(default=None, ge=0)
    do_sample: bool | None = None
    seed: int | None = None
    repetition_penalty: float | None = None
    stop: list[str] | None = None


class LogitsCapture(FrozenModel):
    """How logits were captured for this trace.

    mode=topk: at most k candidates per step, from a real model distribution.
    mode=none: backend did not expose real logits; do not invent them.
    """

    mode: Literal["topk", "none"]
    k: int | None = Field(default=None, ge=1, le=MAX_TOP_K)
    unavailable_reason: str | None = None

    @model_validator(mode="after")
    def _mode_fields(self) -> Self:
        if self.mode == "topk" and self.k is None:
            raise ValueError("logits.mode=topk requires k")
        if self.mode == "none" and not self.unavailable_reason:
            raise ValueError("logits.mode=none requires unavailable_reason")
        return self


class RunMetadata(FrozenModel):
    """Identity and capture notes for one recorded generation. trace_id is stable."""

    trace_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=utcnow)
    prompt: str
    prompt_token_ids: list[int] | None = None
    output_text: str = ""
    source: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    logits: LogitsCapture


class Trace(FrozenModel):
    schema_version: str = SCHEMA_VERSION
    run_metadata: RunMetadata
    environment: Environment = Field(default_factory=Environment)
    model: ModelConfig = Field(
        validation_alias="model_config",
        serialization_alias="model_config",
    )
    generation_config: GenerationConfig = Field(default_factory=GenerationConfig)
    events: list[Event] = Field(default_factory=list)

    @model_validator(mode="after")
    def _schema_and_steps(self) -> Self:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {SCHEMA_VERSION}, got {self.schema_version!r}"
            )
        steps = [event.step for event in self.events]
        if steps != list(range(len(self.events))):
            raise ValueError("events must be ordered with contiguous step ids starting at 0")
        logits = self.run_metadata.logits
        if logits.mode == "none":
            if any(event.top_k for event in self.events):
                raise ValueError("logits.mode=none forbids top_k on events")
        elif logits.k is not None:
            for event in self.events:
                if len(event.top_k) > logits.k:
                    raise ValueError("event top_k longer than logits.k")
        return self


def dumps_json(trace: Trace) -> str:
    return json.dumps(_dump(trace), indent=2, ensure_ascii=False) + "\n"


def loads_json(text: str) -> Trace:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TypeError("trace JSON must be an object")
    return Trace.model_validate(migrate_payload(payload))


def dumps_jsonl(trace: Trace) -> str:
    header = _dump(trace, exclude={"events"})
    header["record"] = "header"
    lines = [_compact_json(header)]
    for event in trace.events:
        payload = _dump(event)
        payload["record"] = "event"
        lines.append(_compact_json(payload))
    return "\n".join(lines) + "\n"


def loads_jsonl(text: str) -> Trace:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("empty JSONL trace")
    header = json.loads(lines[0])
    if not isinstance(header, dict) or header.get("record") != "header":
        raise ValueError("first JSONL line must be a header record")
    header.pop("record")
    events: list[dict[str, Any]] = []
    for line in lines[1:]:
        record = json.loads(line)
        if not isinstance(record, dict) or record.get("record") != "event":
            raise ValueError("JSONL body lines must be event records")
        record.pop("record")
        events.append(record)
    header["events"] = events
    return Trace.model_validate(migrate_payload(header))


def load_path(path: str | Path) -> Trace:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if file_path.suffix == ".jsonl":
        return loads_jsonl(text)
    return loads_json(text)


def _dump(model: BaseModel, exclude: set[str] | None = None) -> dict[str, Any]:
    return model.model_dump(mode="json", by_alias=True, exclude=exclude)


def _compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _is_suffix(needle: list[int], haystack: list[int]) -> bool:
    if len(needle) > len(haystack):
        return False
    return haystack[len(haystack) - len(needle) :] == needle
