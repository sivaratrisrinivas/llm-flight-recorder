"""Versioned Trace and Event schema (SCHEMA_VERSION 1.0.0)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llmfr.migrate import migrate_payload
from llmfr.version import MAX_TOP_K, SCHEMA_VERSION


def utcnow() -> datetime:
    return datetime.now(UTC)


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TopKCandidate(FrozenModel):
    """One alternative at a decode step. Human-readable, not a vocab blob."""

    rank: int = Field(ge=1, le=MAX_TOP_K)
    token_id: int
    token: str
    logit: float
    logprob: float | None = None


class Event(FrozenModel):
    """One generation step (a sampled token plus optional top-k logits)."""

    event_id: UUID = Field(default_factory=uuid4)
    step: int = Field(ge=0)
    token_id: int
    token: str
    top_k: list[TopKCandidate] = Field(default_factory=list, max_length=MAX_TOP_K)

    @model_validator(mode="after")
    def _ranks_are_contiguous(self) -> Self:
        ranks = [candidate.rank for candidate in self.top_k]
        expected = list(range(1, len(self.top_k) + 1))
        if ranks != expected:
            raise ValueError("top_k ranks must be 1..n in order")
        return self


class ModelRef(FrozenModel):
    provider: str = Field(min_length=1)
    name: str = Field(min_length=1)
    revision: str | None = None
    tokenizer: str | None = None


class SamplingConfig(FrozenModel):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = Field(default=None, ge=0)
    max_new_tokens: int | None = Field(default=None, ge=0)
    do_sample: bool | None = None
    seed: int | None = None


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


class Trace(FrozenModel):
    schema_version: str = SCHEMA_VERSION
    trace_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=utcnow)
    model: ModelRef
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    prompt: str
    prompt_token_ids: list[int] | None = None
    output_text: str = ""
    logits: LogitsCapture
    events: list[Event] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _schema_and_steps(self) -> Self:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {SCHEMA_VERSION}, got {self.schema_version!r}"
            )
        steps = [event.step for event in self.events]
        if steps != list(range(len(self.events))):
            raise ValueError("events must be ordered with contiguous step ids starting at 0")
        if self.logits.mode == "none":
            if any(event.top_k for event in self.events):
                raise ValueError("logits.mode=none forbids top_k on events")
        elif self.logits.k is not None:
            for event in self.events:
                if len(event.top_k) > self.logits.k:
                    raise ValueError("event top_k longer than logits.k")
        return self


def dumps_json(trace: Trace) -> str:
    return json.dumps(trace.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"


def loads_json(text: str) -> Trace:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TypeError("trace JSON must be an object")
    return Trace.model_validate(migrate_payload(payload))


def dumps_jsonl(trace: Trace) -> str:
    header = trace.model_dump(mode="json", exclude={"events"})
    header["record"] = "header"
    lines = [_compact_json(header)]
    for event in trace.events:
        payload = event.model_dump(mode="json")
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


def _compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
