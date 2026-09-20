"""Run a sampling vs decoding-config graded study on a prompt list."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TextIO

from llmfr.compare import compare_traces
from llmfr.core.schema import GenerationConfig, Trace
from llmfr.core.version import DEFAULT_TOP_K
from llmfr.privacy import Redactor
from llmfr.record.batch import BatchPromptError, PromptJob
from llmfr.record.recorder import RecordableAdapter, record_generation
from llmfr.storage.store import FormatName, TraceStore
from llmfr.study.grade import (
    GRADING_RULE,
    PairKind,
    extract_last_whole_number,
    grade_output,
    pair_outcome,
    summarize_pairs,
)

DEFAULT_SAMPLING_SEEDS = (1, 2)
DEFAULT_SAMPLING_TEMPERATURE = 1.0
DEFAULT_DECODING_SEED = 1
DEFAULT_DECODING_TEMPERATURES = (0.7, 1.2)
DEFAULT_STUDY_MAX_NEW_TOKENS = 64
STUDY_KINDS: tuple[PairKind, ...] = ("sampling", "decoding_config")

_PER_ITEM_SPLIT_FIELDS = ("seed", "temperature", "greedy", "max_new_tokens")


@dataclass(frozen=True)
class StudyReport:
    """One graded study: pair rows plus a class table a stranger can read."""

    n_items: int
    n_pairs: int
    model: str
    revision: str | None
    sampling_seeds: tuple[int, int]
    sampling_temperature: float
    decoding_seed: int
    decoding_temperatures: tuple[float, float]
    max_new_tokens: int
    capture_k: int
    grading_rule: str
    pairs: tuple[dict[str, Any], ...]
    summary: dict[str, Any]

    def payload(self) -> dict[str, Any]:
        return {
            "n_items": self.n_items,
            "n_pairs": self.n_pairs,
            "model": self.model,
            "revision": self.revision,
            "sampling_seeds": list(self.sampling_seeds),
            "sampling_temperature": self.sampling_temperature,
            "decoding_seed": self.decoding_seed,
            "decoding_temperatures": list(self.decoding_temperatures),
            "max_new_tokens": self.max_new_tokens,
            "capture_k": self.capture_k,
            "grading_rule": self.grading_rule,
            "pairs": list(self.pairs),
            "summary": self.summary,
        }


def parse_int_pair(raw: str, *, name: str) -> tuple[int, int]:
    parts = _split_pair(raw, name=name)
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(
            f"{name} needs exactly two comma-separated integers (got {raw!r})"
        ) from exc


def parse_float_pair(raw: str, *, name: str) -> tuple[float, float]:
    parts = _split_pair(raw, name=name)
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(f"{name} needs exactly two comma-separated numbers (got {raw!r})") from exc


def _split_pair(raw: str, *, name: str) -> tuple[str, str]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2 or any(not part for part in parts):
        raise ValueError(f"{name} needs exactly two comma-separated values (got {raw!r})")
    return parts[0], parts[1]


def require_study_jobs(jobs: Sequence[PromptJob]) -> None:
    """Fail closed unless every item has integer gold and no split overrides."""
    if not jobs:
        raise BatchPromptError("prompt list is empty")
    for job in jobs:
        if job.gold is None:
            raise BatchPromptError(
                f"{job.path}:{job.line}: study requires integer gold",
                path=job.path,
                line=job.line,
            )
        for field in _PER_ITEM_SPLIT_FIELDS:
            if getattr(job, field) is not None:
                raise BatchPromptError(
                    f"{job.path}:{job.line}: study puts seed/temperature/greedy/"
                    "max_new_tokens on the CLI splits, not per prompt line",
                    path=job.path,
                    line=job.line,
                )


def run_study(
    adapter: RecordableAdapter,
    jobs: Sequence[PromptJob],
    *,
    sampling_seeds: tuple[int, int] = DEFAULT_SAMPLING_SEEDS,
    sampling_temperature: float = DEFAULT_SAMPLING_TEMPERATURE,
    decoding_seed: int = DEFAULT_DECODING_SEED,
    decoding_temperatures: tuple[float, float] = DEFAULT_DECODING_TEMPERATURES,
    max_new_tokens: int = DEFAULT_STUDY_MAX_NEW_TOKENS,
    capture_k: int = DEFAULT_TOP_K,
    store: TraceStore | None = None,
    persist: bool = True,
    redact: Redactor | None = None,
    fmt: FormatName = "jsonl",
    source: str | None = "llmfr.study",
    progress: TextIO | None = None,
) -> StudyReport:
    """Record sampling and decoding-config pairs, compare, and grade.

    Grades the unredacted completion, then applies ``redact`` before persist.
    Does not invent logits. Compare does not call a model.
    """
    require_study_jobs(jobs)
    progress_stream = sys.stderr if progress is None else progress
    pairs: list[dict[str, Any]] = []
    total = len(jobs) * len(STUDY_KINDS)
    done = 0
    for job in jobs:
        assert job.gold is not None
        item_id = job.id if job.id is not None else f"line_{job.line}"
        for kind in STUDY_KINDS:
            done += 1
            progress_stream.write(f"recording {done}/{total} {item_id} {kind}\n")
            progress_stream.flush()
            gen_a, gen_b = _pair_kind_configs(
                kind,
                max_new_tokens=max_new_tokens,
                sampling_seeds=sampling_seeds,
                sampling_temperature=sampling_temperature,
                decoding_seed=decoding_seed,
                decoding_temperatures=decoding_temperatures,
            )
            trace_a = _record_side(
                adapter,
                job,
                generation=gen_a,
                item_id=item_id,
                kind=kind,
                side="a",
                store=store,
                persist=persist,
                redact=redact,
                capture_k=capture_k,
                fmt=fmt,
                source=source,
            )
            trace_b = _record_side(
                adapter,
                job,
                generation=gen_b,
                item_id=item_id,
                kind=kind,
                side="b",
                store=store,
                persist=persist,
                redact=redact,
                capture_k=capture_k,
                fmt=fmt,
                source=source,
            )
            pairs.append(
                _grade_pair(
                    item_id=item_id,
                    gold=job.gold,
                    kind=kind,
                    trace_a=trace_a["graded"],
                    trace_b=trace_b["graded"],
                    stored_a=trace_a["stored"],
                    stored_b=trace_b["stored"],
                    include_outputs=redact is None,
                )
            )
    summary = summarize_pairs(pairs)
    model = adapter.model_config
    return StudyReport(
        n_items=len(jobs),
        n_pairs=len(pairs),
        model=model.name,
        revision=model.revision,
        sampling_seeds=sampling_seeds,
        sampling_temperature=sampling_temperature,
        decoding_seed=decoding_seed,
        decoding_temperatures=decoding_temperatures,
        max_new_tokens=max_new_tokens,
        capture_k=capture_k,
        grading_rule=GRADING_RULE,
        pairs=tuple(pairs),
        summary=summary,
    )


def _record_side(
    adapter: RecordableAdapter,
    job: PromptJob,
    *,
    generation: GenerationConfig,
    item_id: str,
    kind: PairKind,
    side: Literal["a", "b"],
    store: TraceStore | None,
    persist: bool,
    redact: Redactor | None,
    capture_k: int,
    fmt: FormatName,
    source: str | None,
) -> dict[str, Trace]:
    assert job.gold is not None
    tags = dict(job.tags)
    tags["study_item"] = item_id
    tags["study_kind"] = kind
    tags["side"] = side
    tags["gold"] = str(job.gold)
    if job.id is not None:
        tags["id"] = job.id
    try:
        graded = record_generation(
            adapter,
            job.prompt,
            generation=generation,
            store=None,
            capture_k=capture_k,
            fmt=fmt,
            source=source,
            tags=tags,
            persist=False,
            redact=None,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise BatchPromptError(
            f"{job.path}:{job.line}: {exc}",
            path=job.path,
            line=job.line,
        ) from exc
    stored = graded if redact is None else redact(graded)
    if persist and store is not None:
        store.put(stored, fmt=fmt)
    return {"graded": graded, "stored": stored}


def _pair_kind_configs(
    kind: PairKind,
    *,
    max_new_tokens: int,
    sampling_seeds: tuple[int, int],
    sampling_temperature: float,
    decoding_seed: int,
    decoding_temperatures: tuple[float, float],
) -> tuple[GenerationConfig, GenerationConfig]:
    if kind == "sampling":
        return (
            GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=sampling_temperature,
                seed=sampling_seeds[0],
            ),
            GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=sampling_temperature,
                seed=sampling_seeds[1],
            ),
        )
    return (
        GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=decoding_temperatures[0],
            seed=decoding_seed,
        ),
        GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=decoding_temperatures[1],
            seed=decoding_seed,
        ),
    )


def _grade_pair(
    *,
    item_id: str,
    gold: int,
    kind: PairKind,
    trace_a: Trace,
    trace_b: Trace,
    stored_a: Trace,
    stored_b: Trace,
    include_outputs: bool,
) -> dict[str, Any]:
    compared = compare_traces(trace_a, trace_b)
    first = compared.first_divergence
    grade_a = grade_output(trace_a.run_metadata.output_text, gold)
    grade_b = grade_output(trace_b.run_metadata.output_text, gold)
    return {
        "item_id": item_id,
        "gold": gold,
        "intended_kind": kind,
        "observed_class": None if first is None else first.classification,
        "first_step": None if first is None else first.step,
        "identical": compared.identical,
        "grade_a": grade_a,
        "grade_b": grade_b,
        "extracted_a": extract_last_whole_number(trace_a.run_metadata.output_text),
        "extracted_b": extract_last_whole_number(trace_b.run_metadata.output_text),
        "outcome": pair_outcome(grade_a, grade_b),
        "output_a": trace_a.run_metadata.output_text if include_outputs else None,
        "output_b": trace_b.run_metadata.output_text if include_outputs else None,
        "trace_a": str(stored_a.run_metadata.trace_id),
        "trace_b": str(stored_b.run_metadata.trace_id),
        "seed_a": trace_a.generation_config.seed,
        "seed_b": trace_b.generation_config.seed,
        "temperature_a": trace_a.generation_config.temperature,
        "temperature_b": trace_b.generation_config.temperature,
    }
