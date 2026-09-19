"""Minimal inspect, record, replay, and compare CLI."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from llmfr.adapters.huggingface import (
    DEFAULT_HF_MODEL_ID,
    HuggingFaceCausalLMAdapter,
    HuggingFaceExtraMissingError,
)
from llmfr.compare import CompareResult, compare_traces, format_compare_result
from llmfr.core.format import format_trace_topk
from llmfr.core.migrate import UnsupportedSchemaVersionError
from llmfr.core.schema import GenerationConfig, Trace, load_path
from llmfr.core.version import DEFAULT_TOP_K, SCHEMA_VERSION, __version__
from llmfr.record import record_generation
from llmfr.replay import BIT_IDENTICAL_CAVEAT, ReplayResult, replay_trace
from llmfr.storage import TraceStore
from llmfr.storage.store import FormatName

_LOAD_ERRORS = (
    OSError,
    TypeError,
    ValueError,
    ValidationError,
    UnsupportedSchemaVersionError,
    json.JSONDecodeError,
)

_RECORD_ERRORS = (
    OSError,
    TypeError,
    ValueError,
    ValidationError,
    RuntimeError,
    HuggingFaceExtraMissingError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmfr",
        description=(
            "LLM Flight Recorder. Record a short generation, replay a stored trace, "
            "compare two traces, or inspect stored traces."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="Print package and schema versions")

    validate = sub.add_parser("validate", help="Validate a JSON or JSONL trace file")
    validate.add_argument("path")

    topk = sub.add_parser("topk", help="Print human-readable top-k logits from a trace")
    topk.add_argument("path")

    record = sub.add_parser(
        "record",
        help="Record a short generation and print trace_id (Milestone 3, minimal)",
    )
    record.add_argument("prompt", help="Prompt text to encode and generate from")
    record.add_argument(
        "--store",
        default=".llmfr",
        help="TraceStore directory (SQLite index plus traces/)",
    )
    record.add_argument("--max-new-tokens", type=int, default=8)
    record.add_argument("--seed", type=int, default=None)
    record.add_argument("--temperature", type=float, default=1.0)
    record.add_argument(
        "--greedy",
        action="store_true",
        help="Argmax instead of sampling (ignores --temperature for the choice)",
    )
    record.add_argument("--model", default=None, help="Hugging Face model id (default: tiny-gpt2)")
    record.add_argument("--capture-k", type=int, default=DEFAULT_TOP_K)
    record.add_argument("--max-visible-tokens", type=int, default=None)
    record.add_argument(
        "--format",
        choices=("json", "jsonl"),
        default="jsonl",
        dest="fmt",
    )

    replay = sub.add_parser(
        "replay",
        help="Replay a stored trace and print a structured result (Milestone 4, minimal)",
    )
    replay.add_argument("trace_id", help="Stable trace_id from TraceStore")
    replay.add_argument(
        "--store",
        default=".llmfr",
        help="TraceStore directory (SQLite index plus traces/)",
    )

    compare = sub.add_parser(
        "compare",
        help="Compare two traces: first divergence vs downstream effects",
    )
    compare.add_argument("trace_a", help="Trace file path or TraceStore trace_id")
    compare.add_argument("trace_b", help="Trace file path or TraceStore trace_id")
    compare.add_argument(
        "--store",
        default=".llmfr",
        help="TraceStore directory when arguments are trace_ids",
    )
    compare.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print structured CompareResult JSON instead of the text report",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "version":
        sys.stdout.write(f"llmfr {__version__}\nschema {SCHEMA_VERSION}\n")
        return 0
    if args.command == "validate":
        trace = _load_trace(args.path)
        if isinstance(trace, int):
            return trace
        sys.stdout.write(
            f"ok {trace.run_metadata.trace_id} "
            f"schema={trace.schema_version} events={len(trace.events)}\n"
        )
        return 0
    if args.command == "topk":
        trace = _load_trace(args.path)
        if isinstance(trace, int):
            return trace
        sys.stdout.write(format_trace_topk(trace))
        return 0
    if args.command == "record":
        return _cmd_record(args)
    if args.command == "replay":
        return _cmd_replay(args)
    if args.command == "compare":
        return _cmd_compare(args)
    raise AssertionError(f"unknown command {args.command}")


def _cmd_record(args: argparse.Namespace) -> int:
    try:
        adapter = _build_hf_adapter(
            model_id=args.model,
            max_visible_tokens=args.max_visible_tokens,
        )
        generation = GenerationConfig(
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            do_sample=not args.greedy,
            seed=args.seed,
        )
        trace = record_generation(
            adapter,
            args.prompt,
            generation=generation,
            store=TraceStore(Path(args.store)),
            capture_k=args.capture_k,
            fmt=cast(FormatName, args.fmt),
            source="cli",
        )
    except _RECORD_ERRORS as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    sys.stdout.write(f"{trace.run_metadata.trace_id}\n")
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    try:
        trace = TraceStore(Path(args.store)).get(args.trace_id)
    except KeyError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    except _RECORD_ERRORS as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    try:
        result = _replay(trace)
    except _RECORD_ERRORS as exc:
        result = ReplayResult(
            trace_id=str(trace.run_metadata.trace_id),
            status="not_replayable",
            matched_steps=0,
            total_steps=len(trace.events),
            recorded_revision=trace.model.revision,
            reason=str(exc),
            notes=(BIT_IDENTICAL_CAVEAT, f"replay failed: {exc}"),
        )
    sys.stdout.write(result.model_dump_json(indent=2) + "\n")
    if result.status == "reproduced":
        return 0
    return 1


def _cmd_compare(args: argparse.Namespace) -> int:
    store_root = Path(args.store)
    try:
        trace_a = _load_trace_ref(args.trace_a, store_root)
        trace_b = _load_trace_ref(args.trace_b, store_root)
    except KeyError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    except _LOAD_ERRORS as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    result = _compare(trace_a, trace_b)
    if args.as_json:
        sys.stdout.write(result.model_dump_json(indent=2) + "\n")
    else:
        sys.stdout.write(format_compare_result(result))
    if result.identical:
        return 0
    return 1


def _looks_like_trace_path(ref: str) -> bool:
    path = Path(ref)
    if path.suffix.lower() in {".json", ".jsonl"}:
        return True
    if os.sep in ref:
        return True
    return os.altsep is not None and os.altsep in ref


def _load_trace_ref(ref: str, store_root: Path) -> Trace:
    path = Path(ref)
    if path.is_file():
        return load_path(path)
    if _looks_like_trace_path(ref):
        raise FileNotFoundError(f"trace file not found: {ref}")
    return TraceStore(store_root).get(ref)


def _replay(trace: Trace) -> ReplayResult:
    return replay_trace(trace)


def _compare(trace_a: Trace, trace_b: Trace) -> CompareResult:
    return compare_traces(trace_a, trace_b)


def _build_hf_adapter(
    *,
    model_id: str | None,
    max_visible_tokens: int | None,
) -> HuggingFaceCausalLMAdapter:
    return HuggingFaceCausalLMAdapter(
        DEFAULT_HF_MODEL_ID if model_id is None else model_id,
        device="cpu",
        max_visible_tokens=max_visible_tokens,
    )


def _load_trace(path: str) -> Trace | int:
    try:
        return load_path(path)
    except _LOAD_ERRORS as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run(argv))
