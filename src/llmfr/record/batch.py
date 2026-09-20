"""Batch prompt lists for ``llmfr record --prompts``.

A stranger can record many prompts in one CLI call. The file is the input;
CLI flags (provider, model, store, redact, persist) apply to every item.
JSONL objects may override generation fields per line. Provider selection
stays on the CLI (Hugging Face default; OpenAI only with ``--provider openai``
or an ``openai:`` model prefix).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from llmfr.core.schema import GenerationConfig, Trace
from llmfr.core.version import DEFAULT_TOP_K
from llmfr.privacy import Redactor
from llmfr.record.recorder import RecordableAdapter, record_generation
from llmfr.storage.store import FormatName, TraceStore

_JSONL_SUFFIX = ".jsonl"
_JSON_SUFFIX = ".json"


class PromptRecord(BaseModel):
    """One JSON object in a prompt list. Unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    id: str | None = None
    seed: int | None = None
    temperature: float | None = None
    max_new_tokens: int | None = Field(default=None, ge=1)
    greedy: bool | None = None
    gold: int | None = None
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("prompt")
    @classmethod
    def _prompt_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("prompt must be a non-empty string")
        return stripped

    @field_validator("id")
    @classmethod
    def _id_non_empty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("id must be a non-empty string")
        return stripped

    @field_validator("gold", mode="before")
    @classmethod
    def _gold_integer(cls, value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("gold must be an integer")
        return value


@dataclass(frozen=True)
class PromptJob:
    """One prompt to record, with optional per-item generation overrides."""

    prompt: str
    path: Path
    line: int
    id: str | None = None
    seed: int | None = None
    temperature: float | None = None
    max_new_tokens: int | None = None
    greedy: bool | None = None
    gold: int | None = None
    tags: dict[str, str] = field(default_factory=dict)


class BatchPromptError(ValueError):
    """A prompt-list or per-item record failure, with file location."""

    def __init__(self, message: str, *, path: Path | None = None, line: int | None = None) -> None:
        super().__init__(message)
        self.path = path
        self.line = line


def load_prompt_file(path: str | Path) -> list[PromptJob]:
    """Load prompts from a text, JSONL, or JSON file.

    ``.jsonl``: one JSON string or object per line (blank lines skipped).
    ``.json``: a JSON array of strings/objects, or one object/string.
    Anything else: UTF-8 text, one prompt per line (blank lines skipped).

    Surrounding whitespace on each prompt is stripped. An empty list is an
    error. Unknown JSON object keys are an error (fail closed, including
    ``provider``, ``model``, and ``api_key``).
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"prompt file not found: {file_path}")
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise BatchPromptError(
            f"{file_path}: not valid UTF-8 ({exc})",
            path=file_path,
        ) from exc
    suffix = file_path.suffix.lower()
    if suffix == _JSONL_SUFFIX:
        jobs = _load_jsonl(file_path, text)
    elif suffix == _JSON_SUFFIX:
        jobs = _load_json(file_path, text)
    else:
        jobs = _load_text(file_path, text)
    if not jobs:
        raise BatchPromptError(f"{file_path}: prompt list is empty", path=file_path)
    return jobs


def record_prompt_batch(
    adapter: RecordableAdapter,
    jobs: Sequence[PromptJob],
    *,
    generation: GenerationConfig,
    store: TraceStore | None = None,
    capture_k: int = DEFAULT_TOP_K,
    fmt: FormatName = "jsonl",
    source: str | None = "llmfr.record",
    persist: bool = True,
    redact: Redactor | None = None,
    on_recorded: Callable[[Trace], None] | None = None,
) -> list[Trace]:
    """Record each job with the same adapter. Fail closed on the first error.

    ``on_recorded`` runs after each successful item, before the next one,
    so a CLI can stream ``trace_id`` lines as they complete. Earlier items
    already written stay in ``store``. Does not invent logits. Per-item
    ``seed`` / ``temperature`` / ``max_new_tokens`` / ``greedy`` override
    ``generation``; backend selection is not a per-item field.
    """
    if not jobs:
        raise BatchPromptError("prompt list is empty")
    traces: list[Trace] = []
    for index, job in enumerate(jobs):
        tags = dict(job.tags)
        tags["batch_index"] = str(index)
        if job.id is not None:
            tags["id"] = job.id
        if job.gold is not None:
            tags["gold"] = str(job.gold)
        try:
            recorded = record_generation(
                adapter,
                job.prompt,
                generation=_job_generation(generation, job),
                store=store,
                capture_k=capture_k,
                fmt=fmt,
                source=source,
                tags=tags,
                persist=persist,
                redact=redact,
            )
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            raise BatchPromptError(
                f"{job.path}:{job.line}: {exc}",
                path=job.path,
                line=job.line,
            ) from exc
        traces.append(recorded)
        if on_recorded is not None:
            on_recorded(recorded)
    return traces


def _job_generation(defaults: GenerationConfig, job: PromptJob) -> GenerationConfig:
    do_sample: bool | None
    if job.greedy is True:
        do_sample = False
    elif job.greedy is False:
        do_sample = True
    else:
        do_sample = defaults.do_sample
    return GenerationConfig(
        temperature=defaults.temperature if job.temperature is None else job.temperature,
        top_p=defaults.top_p,
        top_k=defaults.top_k,
        max_new_tokens=(
            defaults.max_new_tokens if job.max_new_tokens is None else job.max_new_tokens
        ),
        do_sample=do_sample,
        seed=defaults.seed if job.seed is None else job.seed,
        repetition_penalty=defaults.repetition_penalty,
        stop=defaults.stop,
    )


def _load_text(path: Path, text: str) -> list[PromptJob]:
    jobs: list[PromptJob] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        jobs.append(PromptJob(prompt=stripped, path=path, line=line_number))
    return jobs


def _load_jsonl(path: Path, text: str) -> list[PromptJob]:
    jobs: list[PromptJob] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BatchPromptError(
                f"{path}:{line_number}: corrupt JSON ({exc})",
                path=path,
                line=line_number,
            ) from exc
        jobs.append(_job_from_payload(payload, path=path, line=line_number))
    return jobs


def _load_json(path: Path, text: str) -> list[PromptJob]:
    if not text.strip():
        raise BatchPromptError(f"{path}: prompt list is empty", path=path)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BatchPromptError(f"{path}: corrupt JSON ({exc})", path=path) from exc
    if isinstance(payload, list):
        return [
            _job_from_payload(item, path=path, line=index)
            for index, item in enumerate(payload, start=1)
        ]
    return [_job_from_payload(payload, path=path, line=1)]


def _job_from_payload(payload: Any, *, path: Path, line: int) -> PromptJob:
    if isinstance(payload, str):
        record = _validate_record({"prompt": payload}, path=path, line=line)
    elif isinstance(payload, dict):
        record = _validate_record(payload, path=path, line=line)
    else:
        raise BatchPromptError(
            f"{path}:{line}: each item must be a JSON object or string",
            path=path,
            line=line,
        )
    return PromptJob(
        prompt=record.prompt,
        path=path,
        line=line,
        id=record.id,
        seed=record.seed,
        temperature=record.temperature,
        max_new_tokens=record.max_new_tokens,
        greedy=record.greedy,
        gold=record.gold,
        tags=dict(record.tags),
    )


def _validate_record(payload: dict[str, Any], *, path: Path, line: int) -> PromptRecord:
    try:
        return PromptRecord.model_validate(payload)
    except ValidationError as exc:
        raise BatchPromptError(
            f"{path}:{line}: {_validation_message(exc)}",
            path=path,
            line=line,
        ) from exc


def _validation_message(exc: ValidationError) -> str:
    error = exc.errors()[0]
    loc = ".".join(str(part) for part in error.get("loc", ()) if part != "__root__")
    message = str(error.get("msg", exc))
    if loc:
        return f"{loc}: {message}"
    return message
