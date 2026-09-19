"""Minimal inspect CLI for Milestone 1. No record, replay, or compare commands."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from pydantic import ValidationError

from llmfr.core.format import format_trace_topk
from llmfr.core.migrate import UnsupportedSchemaVersionError
from llmfr.core.schema import Trace, load_path
from llmfr.core.version import SCHEMA_VERSION, __version__

_LOAD_ERRORS = (
    OSError,
    TypeError,
    ValueError,
    ValidationError,
    UnsupportedSchemaVersionError,
    json.JSONDecodeError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmfr",
        description=(
            "LLM Flight Recorder. Validate and print stored traces. "
            "Record, replay, and compare commands are not available yet."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="Print package and schema versions")

    validate = sub.add_parser("validate", help="Validate a JSON or JSONL trace file")
    validate.add_argument("path")

    topk = sub.add_parser("topk", help="Print human-readable top-k logits from a trace")
    topk.add_argument("path")

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
    raise AssertionError(f"unknown command {args.command}")


def _load_trace(path: str) -> Trace | int:
    try:
        return load_path(path)
    except _LOAD_ERRORS as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run(argv))
