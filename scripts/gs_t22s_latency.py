"""GS-T22s: p50/p99 wall-clock latency for llmfr record, compare, and study.

Finding numbers come from a `--backend hf` run you actually perform
(subprocess `python -m llmfr`, portfolio Qwen CPU pin). `--backend fake`
is in-process CLI smoke with a fake adapter and must not overwrite
`docs/findings`. `--backend tiny-gpt2` is a real CLI smoke path and also
must not overwrite this finding. Recompute the table from checked-in JSON
with `--from-results`. Do not hand-edit timings into the finding.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

from llmfr.adapters.huggingface import (
    DEFAULT_HF_MODEL_ID,
    PORTFOLIO_DEMO_MODEL_ID,
    PORTFOLIO_DEMO_MODEL_REVISION,
)
from llmfr.cli import run as llmfr_run
from llmfr.core.version import __version__ as LLMFR_VERSION

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.fakes import FakeCausalLMAdapter  # noqa: E402

FINDING_DIR = ROOT / "docs" / "findings"
DEFAULT_RESULTS = FINDING_DIR / "gs-t22s-results.json"
DEFAULT_FINDING = FINDING_DIR / "gs-t22s-latency.md"
DEFAULT_PROMPTS = ROOT / "examples" / "findings" / "gs-t22s" / "prompts.jsonl"
DEMO1_A = ROOT / "examples" / "demo" / "demo1_a.jsonl"
DEMO1_B = ROOT / "examples" / "demo" / "demo1_b.jsonl"

FINDING_ID = "GS-T22s"
MODEL_ID = PORTFOLIO_DEMO_MODEL_ID
MODEL_REVISION = PORTFOLIO_DEMO_MODEL_REVISION
PROMPT = (
    "A farmer has 17 sheep. All but 9 run away. "
    "How many sheep are left? Think step by step, then give the final number."
)
CAPTURE_K = 5
RECORD_SEED = 1
DEFAULT_WARMUP = 2
DEFAULT_TRIALS = 11
DEFAULT_MAX_NEW_TOKENS = 16
FAKE_WARMUP = 1
FAKE_TRIALS = 3
FAKE_MAX_NEW_TOKENS = 4
PERCENTILE_METHOD = (
    "nearest-rank: rank = ceil(p/100 * n), value = sorted[rank-1]. "
    "With N=11, p99 is the maximum timed trial."
)
FAKE_SMOKE_COMMAND = (
    "python scripts/gs_t22s_latency.py --backend fake "
    "--out /tmp/llmfr-gs-t22s-fake "
    "--results /tmp/llmfr-gs-t22s-fake/results.json "
    "--finding /tmp/llmfr-gs-t22s-fake/finding.md"
)

_COMPARE_EXIT = frozenset({0, 1})
_OK_EXIT = frozenset({0})
_COMMAND_NAMES = ("record", "compare", "study")


def percentile(samples: Sequence[float], p: float) -> float:
    """Nearest-rank percentile. ``p`` is in [0, 100]."""
    if not samples:
        raise ValueError("percentile needs at least one sample")
    if p < 0 or p > 100:
        raise ValueError(f"percentile p must be in [0, 100], got {p}")
    ordered = sorted(float(value) for value in samples)
    if p == 0:
        return ordered[0]
    rank = math.ceil(p / 100.0 * len(ordered))
    return ordered[rank - 1]


def summarize_samples(samples: Sequence[float], *, warmup: int, n_trials: int) -> dict[str, Any]:
    values = [float(sample) for sample in samples]
    if len(values) != n_trials:
        raise ValueError(f"expected {n_trials} timed samples, got {len(values)}")
    return {
        "warmup": warmup,
        "n_trials": n_trials,
        "n_samples": len(values),
        "samples_s": values,
        "min_s": min(values),
        "max_s": max(values),
        "mean_s": sum(values) / len(values),
        "p50_s": percentile(values, 50),
        "p99_s": percentile(values, 99),
        "percentile_method": PERCENTILE_METHOD,
    }


def fmt_seconds(value: float) -> str:
    return f"{value:.3f}"


def require_portfolio_qwen(payload: Mapping[str, Any]) -> None:
    model = str(payload.get("model", ""))
    revision = str(payload.get("revision", ""))
    if "llama" in model.lower():
        raise ValueError("GS-T22s finding docs require the portfolio Qwen pin; no silent Llama")
    if model == DEFAULT_HF_MODEL_ID or "tiny-gpt2" in model.lower():
        raise ValueError("GS-T22s finding docs require the portfolio Qwen pin, not tiny-gpt2")
    if model != MODEL_ID or revision != MODEL_REVISION:
        raise ValueError(
            f"GS-T22s finding docs require {MODEL_ID} at {MODEL_REVISION}; "
            f"got model={model!r} revision={revision!r}"
        )
    if payload.get("backend") != "hf":
        raise ValueError("GS-T22s finding docs require backend=hf")


def _docs_bound(path: Path) -> bool:
    try:
        path.resolve().relative_to(FINDING_DIR.resolve())
    except ValueError:
        return False
    return True


def _repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _hardware() -> dict[str, Any]:
    cpu_model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
    }


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {"llmfr": LLMFR_VERSION}
    try:
        import torch
        import transformers
    except ImportError:
        return versions
    versions["torch"] = str(torch.__version__)
    versions["transformers"] = str(transformers.__version__)
    return versions


def _flat_adapter() -> FakeCausalLMAdapter:
    peaked_after = {
        (1, 0): (5.0, 0.0, 0.0, 0.0),
        (1, 1): (0.0, 5.0, 0.0, 0.0),
        (1, 2): (0.0, 0.0, 5.0, 0.0),
        (1, 3): (0.0, 0.0, 0.0, 5.0),
    }
    return FakeCausalLMAdapter(
        prompt_ids=[1],
        logits=(1.0, 1.0, 1.0, 1.0),
        logits_for_prefix={(1,): (1.0, 1.0, 1.0, 1.0), **peaked_after},
    )


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def _llmfr(*args: str) -> list[str]:
    return [sys.executable, "-m", "llmfr", *args]


def _time_subprocess(argv: Sequence[str], *, allowed: frozenset[int], name: str) -> float:
    start = time.perf_counter()
    proc = subprocess.run(
        _llmfr(*argv),
        check=False,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    elapsed = time.perf_counter() - start
    if proc.returncode not in allowed:
        detail = (proc.stderr or proc.stdout or "no output").strip()
        raise RuntimeError(f"{name} failed ({proc.returncode}): {detail}")
    return elapsed


def _time_cli_run(argv: Sequence[str], *, allowed: frozenset[int], name: str) -> float:
    buf = io.StringIO()
    err = io.StringIO()
    start = time.perf_counter()
    with redirect_stdout(buf), redirect_stderr(err):
        code = llmfr_run(list(argv))
    elapsed = time.perf_counter() - start
    if code not in allowed:
        detail = (err.getvalue() or buf.getvalue() or "no output").strip()
        raise RuntimeError(f"{name} failed ({code}): {detail}")
    return elapsed


def _record_argv(
    *,
    store: Path,
    model: str,
    revision: str | None,
    max_new_tokens: int,
) -> list[str]:
    argv = [
        "record",
        PROMPT,
        "--store",
        str(store),
        "--max-new-tokens",
        str(max_new_tokens),
        "--seed",
        str(RECORD_SEED),
        "--capture-k",
        str(CAPTURE_K),
        "--model",
        model,
    ]
    if revision is not None:
        argv.extend(["--revision", revision])
    return argv


def _compare_argv() -> list[str]:
    return ["compare", _repo_rel(DEMO1_A), _repo_rel(DEMO1_B)]


def _study_argv(
    *,
    store: Path,
    model: str,
    revision: str | None,
    max_new_tokens: int,
    prompts: Path,
) -> list[str]:
    argv = [
        "study",
        _repo_rel(prompts),
        "--store",
        str(store),
        "--max-new-tokens",
        str(max_new_tokens),
        "--capture-k",
        str(CAPTURE_K),
        "--model",
        model,
    ]
    if revision is not None:
        argv.extend(["--revision", revision])
    return argv


def _time_loop(
    *,
    name: str,
    warmup: int,
    n_trials: int,
    allowed: frozenset[int],
    make_argv: Callable[[Path], list[str]],
    timer: Callable[..., float],
    scratch: Path,
) -> list[float]:
    samples: list[float] = []
    total = warmup + n_trials
    for index in range(total):
        trial_store = scratch / name / f"trial-{index}"
        trial_store.mkdir(parents=True, exist_ok=True)
        argv = make_argv(trial_store)
        label = "warmup" if index < warmup else f"trial {index - warmup + 1}/{n_trials}"
        sys.stderr.write(f"timing {name} {label}\n")
        sys.stderr.flush()
        elapsed = timer(argv, allowed=allowed, name=f"{name} {label}")
        if index >= warmup:
            samples.append(elapsed)
    return samples


def _command_payload(
    *,
    name: str,
    argv: Sequence[str],
    samples: Sequence[float],
    warmup: int,
    n_trials: int,
    allowed: Sequence[int],
    workload: Mapping[str, Any],
    timing_mode: str,
) -> dict[str, Any]:
    stats = summarize_samples(samples, warmup=warmup, n_trials=n_trials)
    return {
        "command": name,
        "argv": list(argv),
        "displayed_command": "llmfr " + shlex.join(list(argv)),
        "timing_mode": timing_mode,
        "allowed_exit_codes": list(allowed),
        "workload": dict(workload),
        **stats,
    }


def run_benchmark(
    *,
    backend: str,
    warmup: int,
    n_trials: int,
    max_new_tokens: int,
    prompts: Path,
    scratch: Path,
) -> dict[str, Any]:
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if n_trials < 1:
        raise ValueError("trials must be >= 1")
    if max_new_tokens < 1:
        raise ValueError("max-new-tokens must be >= 1")
    if not DEMO1_A.is_file() or not DEMO1_B.is_file():
        raise FileNotFoundError("checked-in Demo 1 traces are required for compare timing")
    if not prompts.is_file():
        raise FileNotFoundError(f"study prompt file not found: {prompts}")

    if backend == "hf":
        model = MODEL_ID
        revision: str | None = MODEL_REVISION
        timing_mode = "subprocess"
        timer: Callable[..., float] = _time_subprocess
        adapter_patch = None
    elif backend == "tiny-gpt2":
        model = DEFAULT_HF_MODEL_ID
        revision = None
        timing_mode = "subprocess"
        timer = _time_subprocess
        adapter_patch = None
    elif backend == "fake":
        model = "fake-lm"
        revision = None
        timing_mode = "in-process-cli"
        timer = _time_cli_run
        adapter_patch = patch("llmfr.cli._build_record_adapter", lambda **_kwargs: _flat_adapter())
    else:
        raise ValueError(f"unknown backend {backend!r}")

    def timed(
        name: str, allowed: frozenset[int], make_argv: Callable[[Path], list[str]]
    ) -> list[float]:
        if adapter_patch is None:
            return _time_loop(
                name=name,
                warmup=warmup,
                n_trials=n_trials,
                allowed=allowed,
                make_argv=make_argv,
                timer=timer,
                scratch=scratch,
            )
        with adapter_patch:
            return _time_loop(
                name=name,
                warmup=warmup,
                n_trials=n_trials,
                allowed=allowed,
                make_argv=make_argv,
                timer=timer,
                scratch=scratch,
            )

    record_argv_for_docs = _record_argv(
        store=Path(".llmfr-gs-t22s"),
        model=model,
        revision=revision,
        max_new_tokens=max_new_tokens,
    )
    record_samples = timed(
        "record",
        _OK_EXIT,
        lambda store: _record_argv(
            store=store,
            model=model,
            revision=revision,
            max_new_tokens=max_new_tokens,
        ),
    )
    compare_samples = timed("compare", _COMPARE_EXIT, lambda _store: _compare_argv())
    study_samples = timed(
        "study",
        _OK_EXIT,
        lambda store: _study_argv(
            store=store,
            model=model,
            revision=revision,
            max_new_tokens=max_new_tokens,
            prompts=prompts,
        ),
    )

    recorded_at = datetime.now(UTC).isoformat()
    compare_argv = _compare_argv()
    study_argv_for_docs = _study_argv(
        store=Path(".llmfr-gs-t22s"),
        model=model,
        revision=revision,
        max_new_tokens=max_new_tokens,
        prompts=prompts,
    )
    payload: dict[str, Any] = {
        "finding_id": FINDING_ID,
        "recorded_at": recorded_at,
        "backend": backend,
        "timing_mode": timing_mode,
        "model": model,
        "revision": revision,
        "warmup": warmup,
        "n_trials": n_trials,
        "max_new_tokens": max_new_tokens,
        "capture_k": CAPTURE_K,
        "percentile_method": PERCENTILE_METHOD,
        "hardware": _hardware(),
        "library_versions": _library_versions() if backend != "fake" else {"llmfr": LLMFR_VERSION},
        "prompt": PROMPT,
        "study_prompts": _repo_rel(prompts),
        "compare_trace_a": _repo_rel(DEMO1_A),
        "compare_trace_b": _repo_rel(DEMO1_B),
        "commands": {
            "record": _command_payload(
                name="record",
                argv=record_argv_for_docs,
                samples=record_samples,
                warmup=warmup,
                n_trials=n_trials,
                allowed=sorted(_OK_EXIT),
                workload={
                    "prompt": PROMPT,
                    "max_new_tokens": max_new_tokens,
                    "seed": RECORD_SEED,
                    "capture_k": CAPTURE_K,
                    "persist": True,
                },
                timing_mode=timing_mode,
            ),
            "compare": _command_payload(
                name="compare",
                argv=compare_argv,
                samples=compare_samples,
                warmup=warmup,
                n_trials=n_trials,
                allowed=sorted(_COMPARE_EXIT),
                workload={
                    "trace_a": _repo_rel(DEMO1_A),
                    "trace_b": _repo_rel(DEMO1_B),
                    "calls_model": False,
                    "expected_exit": "0 identical or 1 diverged (Demo 1 is diverged)",
                },
                timing_mode=timing_mode,
            ),
            "study": _command_payload(
                name="study",
                argv=study_argv_for_docs,
                samples=study_samples,
                warmup=warmup,
                n_trials=n_trials,
                allowed=sorted(_OK_EXIT),
                workload={
                    "prompts": _repo_rel(prompts),
                    "n_items": 1,
                    "max_new_tokens": max_new_tokens,
                    "capture_k": CAPTURE_K,
                    "persist": True,
                    "recorded_sides_per_item": 4,
                },
                timing_mode=timing_mode,
            ),
        },
    }
    return payload


def recompute_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    updated = dict(payload)
    commands = payload.get("commands")
    if not isinstance(commands, dict):
        raise ValueError("results JSON missing commands")
    recomputed: dict[str, Any] = {}
    warmup = int(payload["warmup"])
    n_trials = int(payload["n_trials"])
    for name in _COMMAND_NAMES:
        row = commands.get(name)
        if not isinstance(row, dict):
            raise ValueError(f"results JSON missing commands.{name}")
        samples = row.get("samples_s")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"results JSON commands.{name}.samples_s is empty")
        stats = summarize_samples(samples, warmup=warmup, n_trials=n_trials)
        merged = {**row, **stats}
        argv = row.get("argv")
        if isinstance(argv, list) and all(isinstance(part, str) for part in argv):
            merged["displayed_command"] = "llmfr " + shlex.join(list(argv))
        recomputed[name] = merged
    updated["commands"] = recomputed
    updated["percentile_method"] = PERCENTILE_METHOD
    return updated


def readme_latency_line(results: Mapping[str, Any]) -> str:
    rec = results["commands"]["record"]
    cmp_ = results["commands"]["compare"]
    study = results["commands"]["study"]
    warmup = results["warmup"]
    n_trials = results["n_trials"]
    return (
        f"CLI wall-clock on this CPU Qwen capture (warmup {warmup}, N={n_trials}): "
        f"`llmfr record` p50/p99 {fmt_seconds(rec['p50_s'])}/{fmt_seconds(rec['p99_s'])} s, "
        f"`llmfr compare` p50/p99 {fmt_seconds(cmp_['p50_s'])}/{fmt_seconds(cmp_['p99_s'])} s, "
        f"`llmfr study` (1 item, 16 tokens) p50/p99 "
        f"{fmt_seconds(study['p50_s'])}/{fmt_seconds(study['p99_s'])} s. "
        "Method: [`docs/findings/gs-t22s-latency.md`](docs/findings/gs-t22s-latency.md)."
    )


def render_finding(results: Mapping[str, Any]) -> str:
    rec = results["commands"]["record"]
    cmp_ = results["commands"]["compare"]
    study = results["commands"]["study"]
    hardware = results["hardware"]
    cpu_model = hardware.get("cpu_model") or "unspecified"
    table = [
        "| command | p50 (s) | p99 (s) | min (s) | max (s) |",
        "|---|---:|---:|---:|---:|",
        (
            f"| `llmfr record` | {fmt_seconds(rec['p50_s'])} | {fmt_seconds(rec['p99_s'])} | "
            f"{fmt_seconds(rec['min_s'])} | {fmt_seconds(rec['max_s'])} |"
        ),
        (
            f"| `llmfr compare` | {fmt_seconds(cmp_['p50_s'])} | {fmt_seconds(cmp_['p99_s'])} | "
            f"{fmt_seconds(cmp_['min_s'])} | {fmt_seconds(cmp_['max_s'])} |"
        ),
        (
            f"| `llmfr study` | {fmt_seconds(study['p50_s'])} | {fmt_seconds(study['p99_s'])} | "
            f"{fmt_seconds(study['min_s'])} | {fmt_seconds(study['max_s'])} |"
        ),
    ]
    lines = [
        "# GS-T22s: CLI wall-clock latency (p50/p99)",
        "",
        "Status: measured",
        f"Date: {results['recorded_at']}",
        "",
        "## Question",
        "",
        "What is the wall-clock latency of `llmfr record`, `llmfr compare`, and",
        "`llmfr study` on the portfolio Qwen CPU path? Report measured p50 and p99.",
        "Do not invent timings. Fake-adapter and tiny-gpt2 smoke runs are not this table.",
        "",
        "## Setup",
        "",
        f"- **Warmup:** {results['warmup']} discarded invocations per command",
        "  (not included in p50/p99).",
        f"- **N trials:** {results['n_trials']} timed invocations per command.",
        f"- **Percentile:** {results['percentile_method']}",
        f"- **Clock:** `time.perf_counter` around `{results['timing_mode']}`",
        "  `python -m llmfr ...` (same interpreter as `llmfr`).",
        f"- **Model:** `{results['model']}` revision `{results['revision']}`.",
        "- **Library default / CI smoke:** `sshleifer/tiny-gpt2` and `--backend fake`",
        "  are not used for this finding.",
        f"- **record:** Demo 1 prompt, `--max-new-tokens {results['max_new_tokens']}`",
        f"  `--seed {RECORD_SEED}`, persist to a fresh temp store each trial.",
        "- **compare:** checked-in Demo 1 JSONL paths (no model). Exit 1 (diverged)",
        "  is expected and counted as a successful timed trial.",
        "- **study:** 1-item `examples/findings/gs-t22s/prompts.jsonl` (`sheep_trick`),",
        f"  `--max-new-tokens {results['max_new_tokens']}` (not the 8-item / 64-token",
        "  study default). Four recorded sides; the model loads once per invocation.",
        f"- **Hardware:** {hardware['platform']}; Python {hardware['python_version']};",
        f"  cpus={hardware['cpu_count']}; cpu_model={cpu_model}.",
        f"- **Library versions:** {results['library_versions']}",
        f"- **Backend:** `{results['backend']}` (`{results['timing_mode']}`).",
        "",
        "Re-run:",
        "",
        "```bash",
        "python scripts/gs_t22s_latency.py --backend hf --write-docs",
        "```",
        "",
        "CI smoke (fake adapter; must not overwrite this finding):",
        "",
        "```bash",
        FAKE_SMOKE_COMMAND,
        "```",
        "",
        "Raw JSON: `docs/findings/gs-t22s-results.json`.",
        "Study prompt: `examples/findings/gs-t22s/prompts.jsonl`.",
        "",
        "## Result",
        "",
        "Measured seconds. min/max are from the same timed trials as p50/p99.",
        "",
        *table,
        "",
        "## Commands timed",
        "",
        f"- record: `{rec['displayed_command']}`",
        f"- compare: `{cmp_['displayed_command']}`",
        f"- study: `{study['displayed_command']}`",
        "",
        "## Limits",
        "",
        "- N is small on purpose. This is not an SLA and not a CI budget.",
        "- Each record/study trial includes interpreter start, checkpoint load,",
        "  generation, and persist. Compare does not call a model.",
        "- 0.5B-class instruct model on CPU, no chat template (same raw-prompt",
        "  style as the portfolio demo).",
        "- The 1-item / 16-token study workload is not GS-T22q (N=30, 64 tokens).",
        "- tiny-gpt2 answers and `--backend fake` timings are smoke. They are",
        "  not the table above.",
        "- Llama-3.2-1B-Instruct was not used (gated; no HF_TOKEN).",
        "",
    ]
    return "\n".join(lines)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _load_results(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return recompute_payload(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("hf", "fake", "tiny-gpt2"),
        default=None,
        help="hf: live portfolio Qwen via CLI subprocess. fake: in-process adapter. "
        "tiny-gpt2: real CLI smoke.",
    )
    parser.add_argument(
        "--from-results",
        default=None,
        help="Recompute p50/p99 and the finding from a checked-in results JSON. No model.",
    )
    parser.add_argument("--out", default=None, help="Scratch directory for per-trial stores.")
    parser.add_argument("--results", default=None, help="Results JSON path.")
    parser.add_argument("--finding", default=None, help="Finding markdown path.")
    parser.add_argument(
        "--prompts", default=None, help="Study JSONL (default: 1-item sheep_trick)."
    )
    parser.add_argument(
        "--warmup", type=int, default=None, help="Discarded invocations per command."
    )
    parser.add_argument("--trials", type=int, default=None, help="Timed invocations per command.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Tokens for record and study (compare does not generate).",
    )
    parser.add_argument(
        "--write-docs",
        action="store_true",
        help="Write results JSON and finding markdown (hf Qwen pin only).",
    )
    args = parser.parse_args(None if argv is None else list(argv))

    results_path = DEFAULT_RESULTS if args.results is None else Path(args.results)
    finding_path = DEFAULT_FINDING if args.finding is None else Path(args.finding)

    if args.from_results:
        payload = _load_results(Path(args.from_results))
        text = render_finding(payload)
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.write(readme_latency_line(payload) + "\n")
        if args.write_docs:
            if args.backend == "fake" or payload.get("backend") != "hf":
                sys.stderr.write("error: refusing to write finding docs from a non-hf capture\n")
                return 2
            try:
                require_portfolio_qwen(payload)
            except ValueError as exc:
                sys.stderr.write(f"error: {exc}\n")
                return 2
            _write_json(results_path, payload)
            finding_path.write_text(text, encoding="utf-8")
        return 0

    if args.backend is None:
        sys.stderr.write("error: pass --backend hf|fake|tiny-gpt2 or --from-results PATH\n")
        return 2

    warmup = (
        args.warmup
        if args.warmup is not None
        else (FAKE_WARMUP if args.backend == "fake" else DEFAULT_WARMUP)
    )
    n_trials = (
        args.trials
        if args.trials is not None
        else (FAKE_TRIALS if args.backend == "fake" else DEFAULT_TRIALS)
    )
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else (FAKE_MAX_NEW_TOKENS if args.backend == "fake" else DEFAULT_MAX_NEW_TOKENS)
    )
    prompts = DEFAULT_PROMPTS if args.prompts is None else Path(args.prompts)

    if args.backend != "hf":
        if _docs_bound(results_path) or _docs_bound(finding_path):
            sys.stderr.write(
                "error: refusing to overwrite docs/findings with non-hf output; "
                "pass --results and --finding under a scratch directory\n"
            )
            return 2
        if args.write_docs:
            sys.stderr.write("error: refusing to write finding docs from a non-hf backend\n")
            return 2

    if args.backend == "hf":
        model_probe = {"model": MODEL_ID, "revision": MODEL_REVISION, "backend": "hf"}
        try:
            require_portfolio_qwen(model_probe)
        except ValueError as exc:
            sys.stderr.write(f"error: {exc}\n")
            return 2

    scratch_cm: Any
    if args.out is None:
        scratch_cm = tempfile.TemporaryDirectory(prefix="llmfr-gs-t22s-")
        scratch = Path(scratch_cm.name)
    else:
        scratch_cm = None
        scratch = Path(args.out)
        scratch.mkdir(parents=True, exist_ok=True)

    try:
        payload = run_benchmark(
            backend=args.backend,
            warmup=warmup,
            n_trials=n_trials,
            max_new_tokens=max_new_tokens,
            prompts=prompts,
            scratch=scratch,
        )
    finally:
        if scratch_cm is not None:
            scratch_cm.cleanup()

    text = render_finding(payload)
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.write(readme_latency_line(payload) + "\n")
    if args.write_docs:
        if args.backend != "hf":
            sys.stderr.write("error: refusing to write finding docs from a non-hf backend\n")
            return 2
        try:
            require_portfolio_qwen(payload)
        except ValueError as exc:
            sys.stderr.write(f"error: {exc}\n")
            return 2
        _write_json(results_path, payload)
        finding_path.write_text(text, encoding="utf-8")
    elif args.backend != "hf":
        _write_json(results_path, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
