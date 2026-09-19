"""Production Typer CLI: record, replay, compare, and inspect stored traces."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal

import typer
from pydantic import ValidationError
from typer._click.exceptions import NoArgsIsHelpError
from typer.exceptions import Abort, TyperException
from typer.main import get_command
from typer.models import Context

from llmfr.adapters.huggingface import (
    DEFAULT_HF_MODEL_ID,
    HuggingFaceCausalLMAdapter,
    HuggingFaceExtraMissingError,
)
from llmfr.adapters.openai import (
    DEFAULT_OPENAI_MODEL,
    OPENAI_KEY_ENV,
    OpenAIAPIKeyMissingError,
    OpenAIChatAdapter,
    OpenAIExtraMissingError,
    OpenAILogprobsUnavailableError,
    openai_model_name,
    resolve_record_provider,
)
from llmfr.compare import CompareResult, compare_traces, format_compare_result
from llmfr.core.format import (
    format_inspect_overview,
    format_inspect_step,
    format_trace_topk,
    inspect_overview_payload,
    inspect_step_payload,
)
from llmfr.core.migrate import UnsupportedSchemaVersionError
from llmfr.core.schema import GenerationConfig, Trace, load_path
from llmfr.core.version import DEFAULT_TOP_K, SCHEMA_VERSION, __version__
from llmfr.privacy import redact_trace
from llmfr.record import record_generation
from llmfr.replay import BIT_IDENTICAL_CAVEAT, ReplayResult, replay_trace
from llmfr.storage import TraceStore

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2

EXIT_CODES_HELP = (
    "Exit codes: 0 success (compare: traces identical; replay: status reproduced); "
    "1 expected failure (missing or invalid input, compare diverged, "
    "replay not reproduced); 2 usage error (unknown command or invalid options)."
)

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
    OpenAIExtraMissingError,
    OpenAIAPIKeyMissingError,
    OpenAILogprobsUnavailableError,
)

app = typer.Typer(
    name="llmfr",
    help=(
        "LLM Flight Recorder. Record a generation, replay a stored trace, "
        "compare two traces (first divergence vs downstream effects), "
        "or inspect a stored trace at a step. Does not invent logits. " + EXIT_CODES_HELP
    ),
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def format_help() -> str:
    command = get_command(app)
    ctx = Context(command, info_name="llmfr")
    return command.get_help(ctx)


class _HelpFacade:
    """Argparse-shaped helper so existing tests can still call format_help()."""

    def format_help(self) -> str:
        return format_help()


def build_parser() -> _HelpFacade:
    return _HelpFacade()


def run(argv: Sequence[str] | None = None) -> int:
    args = None if argv is None else list(argv)
    try:
        result = app(args=args, standalone_mode=False)
    except Abort:
        sys.stderr.write("error: aborted\n")
        return EXIT_FAIL
    except NoArgsIsHelpError as exc:
        help_text = exc.format_message()
        if not help_text.endswith("\n"):
            help_text += "\n"
        sys.stdout.write(help_text)
        return EXIT_OK
    except TyperException as exc:
        sys.stderr.write(f"error: {exc.format_message()}\n")
        return int(exc.exit_code)
    if isinstance(result, int):
        return result
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run(argv))


@app.command()
def version() -> None:
    """Print package and schema versions."""
    sys.stdout.write(f"llmfr {__version__}\nschema {SCHEMA_VERSION}\n")
    raise typer.Exit(EXIT_OK)


@app.command()
def validate(
    path: Annotated[
        str,
        typer.Argument(metavar="PATH", help="JSON or JSONL trace file to validate."),
    ],
) -> None:
    """Validate a JSON or JSONL trace file. Exit 0 if the schema loads."""
    trace = _load_trace(path)
    if isinstance(trace, int):
        raise typer.Exit(trace)
    sys.stdout.write(
        f"ok {trace.run_metadata.trace_id} "
        f"schema={trace.schema_version} events={len(trace.events)}\n"
    )
    raise typer.Exit(EXIT_OK)


@app.command()
def topk(
    path: Annotated[str, typer.Argument(metavar="PATH", help="JSON or JSONL trace file.")],
) -> None:
    """Print human-readable top-k logits for every step of a trace file.

    Prefer `llmfr inspect TRACE --step N` to see history vs visible context
    together with that step's top-k. This command keeps the full-trace dump.
    """
    trace = _load_trace(path)
    if isinstance(trace, int):
        raise typer.Exit(trace)
    sys.stdout.write(format_trace_topk(trace))
    raise typer.Exit(EXIT_OK)


@app.command()
def record(
    prompt: Annotated[
        str,
        typer.Argument(metavar="PROMPT", help="Prompt text to encode and generate from."),
    ],
    store: Annotated[
        str,
        typer.Option("--store", help="TraceStore directory (SQLite index plus traces/)."),
    ] = ".llmfr",
    max_new_tokens: Annotated[
        int,
        typer.Option("--max-new-tokens", help="Maximum number of new tokens to record."),
    ] = 8,
    seed: Annotated[
        int | None,
        typer.Option(
            "--seed",
            help=(
                "Hugging Face: LocalRNG seed, isolated from torch. "
                "OpenAI: API request field only; not LocalRNG."
            ),
        ),
    ] = None,
    temperature: Annotated[
        float,
        typer.Option("--temperature", help="Sampling temperature. Ignored when --greedy."),
    ] = 1.0,
    greedy: Annotated[
        bool,
        typer.Option("--greedy", help="Argmax instead of sampling."),
    ] = False,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help=(
                "Hugging Face model id (default: sshleifer/tiny-gpt2). "
                "OpenAI requires --provider openai or an openai: prefix "
                "(openai:gpt-4o-mini)."
            ),
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            help=(
                "Backend: huggingface (default, local/CI) or openai. OpenAI is also "
                f"selected when --model has an openai: prefix. Reads {OPENAI_KEY_ENV} "
                "from the environment; there is no API-key flag."
            ),
        ),
    ] = None,
    revision: Annotated[
        str | None,
        typer.Option(
            "--revision",
            help="Hub commit SHA to pin. Default: resolve after download.",
        ),
    ] = None,
    capture_k: Annotated[
        int,
        typer.Option("--capture-k", help="Top-k candidates stored per step (not full vocab)."),
    ] = DEFAULT_TOP_K,
    max_visible_tokens: Annotated[
        int | None,
        typer.Option(
            "--max-visible-tokens",
            help="Left-window size for model-visible context. Default: model max.",
        ),
    ] = None,
    fmt: Annotated[
        Literal["json", "jsonl"],
        typer.Option("--format", help="On-disk trace format under traces/."),
    ] = "jsonl",
    no_persist: Annotated[
        bool,
        typer.Option(
            "--no-persist",
            help="Skip writing the trace to disk. Default is local persist.",
        ),
    ] = False,
    redact: Annotated[
        bool,
        typer.Option(
            "--redact",
            help=(
                "Redact prompt, output, sampled-token strings, top-k token strings, "
                "and context text before any store write."
            ),
        ),
    ] = False,
) -> None:
    """Record a short generation into TraceStore and print trace_id.

    Exit 0 prints the new trace_id on stdout. Exit 1 for expected failures
    (missing Hugging Face extra, missing OPENAI_API_KEY, invalid config, I/O).
    Does not invent logits. Does not accept an API key flag.
    """
    raise typer.Exit(
        _cmd_record(
            prompt=prompt,
            store=store,
            max_new_tokens=max_new_tokens,
            seed=seed,
            temperature=temperature,
            greedy=greedy,
            model=model,
            provider=provider,
            revision=revision,
            capture_k=capture_k,
            max_visible_tokens=max_visible_tokens,
            fmt=fmt,
            persist=not no_persist,
            redact=redact,
        )
    )


@app.command()
def replay(
    trace_id: Annotated[
        str,
        typer.Argument(metavar="TRACE_ID", help="Stable trace_id from TraceStore."),
    ],
    store: Annotated[
        str,
        typer.Option("--store", help="TraceStore directory (SQLite index plus traces/)."),
    ] = ".llmfr",
) -> None:
    """Replay a stored trace and print a structured ReplayResult JSON.

    Exit 0 only when status is reproduced. Any other replay status, unknown
    trace_id, or store error is exit 1. Usage errors are exit 2.
    """
    raise typer.Exit(_cmd_replay(trace_id=trace_id, store=store))


@app.command()
def compare(
    trace_a: Annotated[
        str,
        typer.Argument(metavar="TRACE_A", help="Trace file path or TraceStore trace_id."),
    ],
    trace_b: Annotated[
        str,
        typer.Argument(metavar="TRACE_B", help="Trace file path or TraceStore trace_id."),
    ],
    store: Annotated[
        str,
        typer.Option("--store", help="TraceStore directory when arguments are trace_ids."),
    ] = ".llmfr",
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Print structured CompareResult JSON instead of text."),
    ] = False,
) -> None:
    """Compare two traces: first divergence vs downstream effects.

    Exit 0 only when the traces are identical (no first divergence and no
    config diff of interest). Diverged compares and load errors are exit 1.
    """
    raise typer.Exit(_cmd_compare(trace_a=trace_a, trace_b=trace_b, store=store, as_json=as_json))


@app.command()
def inspect(
    trace: Annotated[
        str,
        typer.Argument(metavar="TRACE", help="Trace file path or TraceStore trace_id."),
    ],
    store: Annotated[
        str,
        typer.Option("--store", help="TraceStore directory when TRACE is a trace_id."),
    ] = ".llmfr",
    step: Annotated[
        int | None,
        typer.Option(
            "--step",
            help=(
                "Inspect one recorded step (0-based): history vs visible context, "
                "top-k, sampled token."
            ),
        ),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Print structured inspect JSON instead of text."),
    ] = False,
) -> None:
    """Inspect a stored trace, optionally at --step N.

    Without --step, prints run metadata and the sampled-token sequence.
    With --step N, prints full_history vs model_visible_context, the sampled
    token, and that step's top-k. Does not invent logits. Exit 1 for missing
    traces or an out-of-range step.
    """
    raise typer.Exit(_cmd_inspect(ref=trace, store=store, step=step, as_json=as_json))


def _cmd_record(
    *,
    prompt: str,
    store: str,
    max_new_tokens: int,
    seed: int | None,
    temperature: float,
    greedy: bool,
    model: str | None,
    provider: str | None,
    revision: str | None,
    capture_k: int,
    max_visible_tokens: int | None,
    fmt: Literal["json", "jsonl"],
    persist: bool,
    redact: bool,
) -> int:
    try:
        adapter = _build_record_adapter(
            model_id=model,
            provider=provider,
            revision=revision,
            max_visible_tokens=max_visible_tokens,
            capture_k=capture_k,
        )
        generation = GenerationConfig(
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            do_sample=not greedy,
            seed=seed,
        )
        recorded = record_generation(
            adapter,
            prompt,
            generation=generation,
            store=None if not persist else TraceStore(Path(store), create=True),
            capture_k=capture_k,
            fmt=fmt,
            source="cli",
            persist=persist,
            redact=redact_trace if redact else None,
        )
    except _RECORD_ERRORS as exc:
        return _fail(exc)
    sys.stdout.write(f"{recorded.run_metadata.trace_id}\n")
    return EXIT_OK


def _cmd_replay(*, trace_id: str, store: str) -> int:
    try:
        loaded = TraceStore(Path(store), create=False).get(trace_id)
    except KeyError as exc:
        return _fail(exc)
    except _RECORD_ERRORS as exc:
        return _fail(exc)
    try:
        result = _replay(loaded)
    except _RECORD_ERRORS as exc:
        result = ReplayResult(
            trace_id=str(loaded.run_metadata.trace_id),
            status="not_replayable",
            matched_steps=0,
            total_steps=len(loaded.events),
            recorded_revision=loaded.model.revision,
            reason=str(exc),
            notes=(BIT_IDENTICAL_CAVEAT, f"replay failed: {exc}"),
        )
    sys.stdout.write(result.model_dump_json(indent=2) + "\n")
    if result.status == "reproduced":
        return EXIT_OK
    return EXIT_FAIL


def _cmd_compare(*, trace_a: str, trace_b: str, store: str, as_json: bool) -> int:
    store_root = Path(store)
    try:
        loaded_a = _load_trace_ref(trace_a, store_root)
        loaded_b = _load_trace_ref(trace_b, store_root)
    except KeyError as exc:
        return _fail(exc)
    except _LOAD_ERRORS as exc:
        return _fail(exc)
    result = _compare(loaded_a, loaded_b)
    if as_json:
        sys.stdout.write(result.model_dump_json(indent=2) + "\n")
    else:
        sys.stdout.write(format_compare_result(result))
    if result.identical:
        return EXIT_OK
    return EXIT_FAIL


def _cmd_inspect(*, ref: str, store: str, step: int | None, as_json: bool) -> int:
    try:
        loaded = _load_trace_ref(ref, Path(store))
    except KeyError as exc:
        return _fail(exc)
    except _LOAD_ERRORS as exc:
        return _fail(exc)
    if step is not None:
        if step < 0 or step >= len(loaded.events):
            return _fail(_step_range_error(step, len(loaded.events)))
        if as_json:
            sys.stdout.write(json.dumps(inspect_step_payload(loaded, step), indent=2) + "\n")
        else:
            sys.stdout.write(format_inspect_step(loaded, step))
        return EXIT_OK
    if as_json:
        sys.stdout.write(json.dumps(inspect_overview_payload(loaded), indent=2) + "\n")
    else:
        sys.stdout.write(format_inspect_overview(loaded))
    return EXIT_OK


def _step_range_error(step: int, event_count: int) -> str:
    if event_count <= 0:
        return f"step {step} is out of range (trace has no events)"
    last = event_count - 1
    return f"step {step} is out of range (trace has {event_count} events, steps 0..{last})"


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
    return TraceStore(store_root, create=False).get(ref)


def _replay(trace: Trace) -> ReplayResult:
    return replay_trace(trace)


def _compare(trace_a: Trace, trace_b: Trace) -> CompareResult:
    return compare_traces(trace_a, trace_b)


def _build_record_adapter(
    *,
    model_id: str | None,
    provider: str | None,
    max_visible_tokens: int | None,
    capture_k: int,
    revision: str | None = None,
) -> HuggingFaceCausalLMAdapter | OpenAIChatAdapter:
    backend = resolve_record_provider(provider, model_id)
    if backend == "openai":
        if revision is not None:
            raise ValueError("--revision pins a Hugging Face Hub commit; omit it for OpenAI")
        return _build_openai_adapter(
            model_id=model_id,
            max_visible_tokens=max_visible_tokens,
            capture_k=capture_k,
        )
    return _build_hf_adapter(
        model_id=model_id,
        revision=revision,
        max_visible_tokens=max_visible_tokens,
    )


def _build_hf_adapter(
    *,
    model_id: str | None,
    max_visible_tokens: int | None,
    revision: str | None = None,
) -> HuggingFaceCausalLMAdapter:
    return HuggingFaceCausalLMAdapter(
        DEFAULT_HF_MODEL_ID if model_id is None else model_id,
        revision=revision,
        device="cpu",
        max_visible_tokens=max_visible_tokens,
    )


def _build_openai_adapter(
    *,
    model_id: str | None,
    max_visible_tokens: int | None,
    capture_k: int,
) -> OpenAIChatAdapter:
    if max_visible_tokens is not None:
        raise ValueError(
            "OpenAI adapter does not implement a local left-window; "
            "omit --max-visible-tokens (the API owns context length)"
        )
    name = DEFAULT_OPENAI_MODEL if model_id is None else openai_model_name(model_id)
    return OpenAIChatAdapter(name, top_logprobs=capture_k)


def _load_trace(path: str) -> Trace | int:
    try:
        return load_path(path)
    except _LOAD_ERRORS as exc:
        return _fail(exc)


def _fail(exc: BaseException | str) -> int:
    message = exc if isinstance(exc, str) else str(exc)
    sys.stderr.write(f"error: {message}\n")
    return EXIT_FAIL
