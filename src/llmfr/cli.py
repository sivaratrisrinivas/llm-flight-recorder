"""Minimal CLI for schema/storage work in Milestone 1."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from llmfr.format import format_trace_topk
from llmfr.schema import load_path
from llmfr.version import SCHEMA_VERSION, __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmfr",
        description="LLM Flight Recorder (schema and storage in Milestone 1).",
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
        trace = load_path(args.path)
        sys.stdout.write(
            f"ok {trace.trace_id} schema={trace.schema_version} events={len(trace.events)}\n"
        )
        return 0
    if args.command == "topk":
        sys.stdout.write(format_trace_topk(load_path(args.path)))
        return 0
    raise AssertionError(f"unknown command {args.command}")


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run(argv))
