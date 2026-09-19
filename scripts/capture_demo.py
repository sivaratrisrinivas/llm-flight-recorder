"""Record the portfolio demo traces and compare reports.

Default Hugging Face path uses `PORTFOLIO_DEMO_MODEL_ID` (Qwen2.5-0.5B-Instruct)
at `PORTFOLIO_DEMO_MODEL_REVISION` through the public `llmfr` CLI. That download
is not a CI step: pytest asserts against the checked-in JSONL. `--backend fake`
never overwrites `examples/demo` (docs/demo.md and tests lock those files to the
portfolio capture). Pass `--out DIR` for a scratch fake-adapter capture that
also writes inspect and an honest `SOURCE.txt`. Do not hand-edit logits into
the fixtures.

CI/smoke `llmfr record` without `--model` still loads `sshleifer/tiny-gpt2`.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from llmfr.adapters.huggingface import (
    DEFAULT_HF_MODEL_ID,
    PORTFOLIO_DEMO_MODEL_ID,
    PORTFOLIO_DEMO_MODEL_REVISION,
)
from llmfr.core.schema import GenerationConfig, dumps_jsonl, load_path
from llmfr.record import record_generation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.fakes import FakeCausalLMAdapter  # noqa: E402

DEMO_DIR = ROOT / "examples" / "demo"

PROMPT = (
    "A farmer has 17 sheep. All but 9 run away. "
    "How many sheep are left? Think step by step, then give the final number."
)
MAX_NEW_TOKENS = 16
DEMO1_SEEDS = (1, 2)
DEMO2_SEED = 1
DEMO2_TEMPERATURES = (0.7, 1.2)
SMOKE_PROMPT = "Hello"
SMOKE_MAX_NEW_TOKENS = 6

_COMPARE_EXIT = frozenset({0, 1})
_OK_EXIT = frozenset({0})


def _flat_adapter() -> FakeCausalLMAdapter:
    peaked_after = {
        (1, 0): (5.0, 0.0, 0.0, 0.0),
        (1, 1): (0.0, 5.0, 0.0, 0.0),
        (1, 2): (0.0, 0.0, 5.0, 0.0),
        (1, 3): (0.0, 0.0, 0.0, 5.0),
        (1, 2, 3): (0.0, 0.0, 0.0, 5.0),
        (1, 3, 0): (5.0, 0.0, 0.0, 0.0),
        (1, 0, 1): (0.0, 5.0, 0.0, 0.0),
        (1, 1, 2): (0.0, 0.0, 5.0, 0.0),
    }
    return FakeCausalLMAdapter(
        prompt_ids=[1],
        logits=(1.0, 1.0, 1.0, 1.0),
        logits_for_prefix={(1,): (1.0, 1.0, 1.0, 1.0), **peaked_after},
    )


def _docs_bound(path: Path) -> bool:
    return path.resolve() == DEMO_DIR.resolve()


def _checked_run(
    cmd: Sequence[str],
    *,
    name: str,
    allowed: frozenset[int],
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    detail = (proc.stderr or proc.stdout or "no output").strip()
    if proc.returncode not in allowed:
        raise RuntimeError(f"{name} failed ({proc.returncode}): {detail}")
    if not proc.stdout.strip():
        raise RuntimeError(f"{name} exited {proc.returncode} with empty stdout: {detail}")
    return proc


def _llmfr(*args: str) -> list[str]:
    return [sys.executable, "-m", "llmfr", *args]


def _copy_pair(out: Path, name: str, trace_a_src: Path, trace_b_src: Path) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    dest_a = out / f"{name}_a.jsonl"
    dest_b = out / f"{name}_b.jsonl"
    shutil.copyfile(trace_a_src, dest_a)
    shutil.copyfile(trace_b_src, dest_b)
    return dest_a, dest_b


def _store_path(store_root: Path, trace_id: str) -> Path:
    jsonl = store_root / "traces" / f"{trace_id}.jsonl"
    if jsonl.is_file():
        return jsonl
    json_path = store_root / "traces" / f"{trace_id}.json"
    if json_path.is_file():
        return json_path
    raise FileNotFoundError(f"recorded trace not found: {trace_id}")


def _write_source(
    out: Path,
    *,
    backend: str,
    model: str,
    revision: str | None,
    prompt: str,
    max_new_tokens: int,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    revision_line = "" if revision is None else f"revision={revision}\n"
    (out / "SOURCE.txt").write_text(
        f"backend={backend}\n"
        f"model={model}\n"
        f"{revision_line}"
        f"prompt={prompt!r}\n"
        f"max_new_tokens={max_new_tokens}\n"
        f"demo1_seeds={DEMO1_SEEDS[0]} {DEMO1_SEEDS[1]}\n"
        f"demo2_seed={DEMO2_SEED} temperatures={DEMO2_TEMPERATURES[0]} "
        f"{DEMO2_TEMPERATURES[1]}\n"
        f"ci_smoke_default={DEFAULT_HF_MODEL_ID}\n"
        f"ci_smoke_prompt={SMOKE_PROMPT!r}\n"
        f"ci_smoke_max_new_tokens={SMOKE_MAX_NEW_TOKENS}\n",
        encoding="utf-8",
    )


def _write_cli_reports(out: Path, dest_1a: Path, dest_1b: Path, dest_2a: Path, dest_2b: Path) -> str:
    compare1 = _checked_run(
        _llmfr("compare", str(dest_1a), str(dest_1b)),
        name="compare demo1",
        allowed=_COMPARE_EXIT,
    )
    compare2 = _checked_run(
        _llmfr("compare", str(dest_2a), str(dest_2b)),
        name="compare demo2",
        allowed=_COMPARE_EXIT,
    )
    inspect = _checked_run(
        _llmfr("inspect", str(dest_1a), "--step", "0"),
        name="inspect demo1 step 0",
        allowed=_OK_EXIT,
    )
    (out / "demo1_compare.txt").write_text(compare1.stdout, encoding="utf-8")
    (out / "demo2_compare.txt").write_text(compare2.stdout, encoding="utf-8")
    (out / "demo1_inspect_step0.txt").write_text(inspect.stdout, encoding="utf-8")
    return compare1.stdout + compare2.stdout


def capture_hf(out: Path = DEMO_DIR) -> str:
    with tempfile.TemporaryDirectory(prefix="llmfr-demo-") as raw:
        store = Path(raw)

        def record(seed: int, temperature: float) -> str:
            proc = _checked_run(
                _llmfr(
                    "record",
                    PROMPT,
                    "--store",
                    str(store),
                    "--max-new-tokens",
                    str(MAX_NEW_TOKENS),
                    "--seed",
                    str(seed),
                    "--temperature",
                    str(temperature),
                    "--model",
                    PORTFOLIO_DEMO_MODEL_ID,
                    "--revision",
                    PORTFOLIO_DEMO_MODEL_REVISION,
                ),
                name="record",
                allowed=_OK_EXIT,
            )
            return proc.stdout.strip()

        id_1a = record(DEMO1_SEEDS[0], 1.0)
        id_1b = record(DEMO1_SEEDS[1], 1.0)
        dest_1a, dest_1b = _copy_pair(out, "demo1", _store_path(store, id_1a), _store_path(store, id_1b))
        id_2a = record(DEMO2_SEED, DEMO2_TEMPERATURES[0])
        id_2b = record(DEMO2_SEED, DEMO2_TEMPERATURES[1])
        dest_2a, dest_2b = _copy_pair(out, "demo2", _store_path(store, id_2a), _store_path(store, id_2b))
        reports = _write_cli_reports(out, dest_1a, dest_1b, dest_2a, dest_2b)
        loaded = load_path(dest_1a)
        _write_source(
            out,
            backend="huggingface",
            model=loaded.model.name,
            revision=loaded.model.revision,
            prompt=PROMPT,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        return reports


def capture_fake(out: Path) -> str:
    if _docs_bound(out):
        raise RuntimeError(
            "refusing to overwrite docs-bound captures in examples/demo; "
            "those are portfolio Hugging Face CLI traces used by docs/demo.md"
        )
    demo1_a = record_generation(
        _flat_adapter(),
        PROMPT,
        generation=GenerationConfig(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=1.0,
            seed=DEMO1_SEEDS[0],
        ),
        capture_k=5,
        source="demo-fixture",
    )
    demo1_b = record_generation(
        _flat_adapter(),
        PROMPT,
        generation=GenerationConfig(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=1.0,
            seed=DEMO1_SEEDS[1],
        ),
        capture_k=5,
        source="demo-fixture",
    )
    demo2_a = record_generation(
        _flat_adapter(),
        PROMPT,
        generation=GenerationConfig(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=DEMO2_TEMPERATURES[0],
            seed=DEMO2_SEED,
        ),
        capture_k=5,
        source="demo-fixture",
    )
    demo2_b = record_generation(
        _flat_adapter(),
        PROMPT,
        generation=GenerationConfig(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=DEMO2_TEMPERATURES[1],
            seed=DEMO2_SEED,
        ),
        capture_k=5,
        source="demo-fixture",
    )
    out.mkdir(parents=True, exist_ok=True)
    dest_1a = out / "demo1_a.jsonl"
    dest_1b = out / "demo1_b.jsonl"
    dest_2a = out / "demo2_a.jsonl"
    dest_2b = out / "demo2_b.jsonl"
    dest_1a.write_text(dumps_jsonl(demo1_a), encoding="utf-8")
    dest_1b.write_text(dumps_jsonl(demo1_b), encoding="utf-8")
    dest_2a.write_text(dumps_jsonl(demo2_a), encoding="utf-8")
    dest_2b.write_text(dumps_jsonl(demo2_b), encoding="utf-8")
    reports = _write_cli_reports(out, dest_1a, dest_1b, dest_2a, dest_2b)
    _write_source(
        out,
        backend="fake",
        model="fake-lm",
        revision=None,
        prompt=PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    return reports


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("hf", "fake"),
        default="hf",
        help=(
            "hf: live portfolio model via the CLI (not CI). "
            "fake: in-repo adapter."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Directory for capture files. Fake backend cannot use examples/demo.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Does not override the fake-backend refusal to overwrite examples/demo.",
    )
    args = parser.parse_args(None if argv is None else list(argv))
    out = DEMO_DIR if args.out is None else Path(args.out)
    if args.backend == "fake":
        if _docs_bound(out):
            extra = " (--force does not override this)" if args.force else ""
            sys.stderr.write(
                "error: refusing to overwrite docs-bound captures in examples/demo; "
                "pass --out DIR for a scratch fake-adapter capture that includes inspect"
                f"{extra}\n"
            )
            return 2
        sys.stdout.write(capture_fake(out))
        return 0
    sys.stdout.write(capture_hf(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
