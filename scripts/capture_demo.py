"""Record the portfolio demo traces and compare reports.

Default path uses the live Hugging Face adapter (`sshleifer/tiny-gpt2` on
CPU) through the public `llmfr` CLI. Pass `--backend fake --force` only when
you cannot run tiny-gpt2; that overwrites the checked-in fixtures with the
in-repo fake adapter. Do not hand-edit logits into the fixtures.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from llmfr.compare import compare_traces, format_compare_result
from llmfr.core.schema import GenerationConfig, dumps_jsonl, load_path
from llmfr.record import record_generation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.fakes import FakeCausalLMAdapter  # noqa: E402

DEMO_DIR = ROOT / "examples" / "demo"
FIXTURES = DEMO_DIR
CAPTURES = DEMO_DIR

PROMPT = "Hello"
MAX_NEW_TOKENS = 6
DEMO1_SEEDS = (1, 2)
DEMO2_SEED = 1
DEMO2_TEMPERATURES = (0.7, 1.2)


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


def _write_pair(name: str, trace_a_src: Path, trace_b_src: Path) -> tuple[Path, Path]:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    CAPTURES.mkdir(parents=True, exist_ok=True)
    dest_a = FIXTURES / f"{name}_a.jsonl"
    dest_b = FIXTURES / f"{name}_b.jsonl"
    shutil.copyfile(trace_a_src, dest_a)
    shutil.copyfile(trace_b_src, dest_b)
    report = format_compare_result(compare_traces(load_path(dest_a), load_path(dest_b)))
    (CAPTURES / f"{name}_compare.txt").write_text(report, encoding="utf-8")
    return dest_a, dest_b


def _store_path(store_root: Path, trace_id: str) -> Path:
    jsonl = store_root / "traces" / f"{trace_id}.jsonl"
    if jsonl.is_file():
        return jsonl
    json_path = store_root / "traces" / f"{trace_id}.json"
    if json_path.is_file():
        return json_path
    raise FileNotFoundError(f"recorded trace not found: {trace_id}")


def _write_source(*, backend: str, model: str, revision: str | None) -> None:
    CAPTURES.mkdir(parents=True, exist_ok=True)
    revision_line = "" if revision is None else f"revision={revision}\n"
    (CAPTURES / "SOURCE.txt").write_text(
        f"backend={backend}\n"
        f"model={model}\n"
        f"{revision_line}"
        f"prompt={PROMPT!r}\n"
        f"max_new_tokens={MAX_NEW_TOKENS}\n"
        f"demo1_seeds={DEMO1_SEEDS[0]} {DEMO1_SEEDS[1]}\n"
        f"demo2_seed={DEMO2_SEED} temperatures={DEMO2_TEMPERATURES[0]} "
        f"{DEMO2_TEMPERATURES[1]}\n",
        encoding="utf-8",
    )


def capture_hf() -> str:
    with tempfile.TemporaryDirectory(prefix="llmfr-demo-") as raw:
        store = Path(raw)

        def record(seed: int, temperature: float) -> str:
            cmd = [
                sys.executable,
                "-m",
                "llmfr",
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
            ]
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"record failed ({proc.returncode}): {proc.stderr or proc.stdout}"
                )
            return proc.stdout.strip()

        id_1a = record(DEMO1_SEEDS[0], 1.0)
        id_1b = record(DEMO1_SEEDS[1], 1.0)
        dest_1a, dest_1b = _write_pair(
            "demo1", _store_path(store, id_1a), _store_path(store, id_1b)
        )
        compare1 = subprocess.run(
            [sys.executable, "-m", "llmfr", "compare", str(dest_1a), str(dest_1b)],
            check=False,
            capture_output=True,
            text=True,
        )
        id_2a = record(DEMO2_SEED, DEMO2_TEMPERATURES[0])
        id_2b = record(DEMO2_SEED, DEMO2_TEMPERATURES[1])
        dest_2a, dest_2b = _write_pair(
            "demo2", _store_path(store, id_2a), _store_path(store, id_2b)
        )
        compare2 = subprocess.run(
            [sys.executable, "-m", "llmfr", "compare", str(dest_2a), str(dest_2b)],
            check=False,
            capture_output=True,
            text=True,
        )
        inspect = subprocess.run(
            [sys.executable, "-m", "llmfr", "inspect", str(dest_1a), "--step", "0"],
            check=False,
            capture_output=True,
            text=True,
        )
        (CAPTURES / "demo1_inspect_step0.txt").write_text(inspect.stdout, encoding="utf-8")
        loaded = load_path(dest_1a)
        _write_source(
            backend="huggingface",
            model=loaded.model.name,
            revision=loaded.model.revision,
        )
        return compare1.stdout + compare2.stdout


def capture_fake() -> str:
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
    FIXTURES.mkdir(parents=True, exist_ok=True)
    CAPTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "demo1_a.jsonl").write_text(dumps_jsonl(demo1_a), encoding="utf-8")
    (FIXTURES / "demo1_b.jsonl").write_text(dumps_jsonl(demo1_b), encoding="utf-8")
    (FIXTURES / "demo2_a.jsonl").write_text(dumps_jsonl(demo2_a), encoding="utf-8")
    (FIXTURES / "demo2_b.jsonl").write_text(dumps_jsonl(demo2_b), encoding="utf-8")
    report1 = format_compare_result(compare_traces(demo1_a, demo1_b))
    report2 = format_compare_result(compare_traces(demo2_a, demo2_b))
    (CAPTURES / "demo1_compare.txt").write_text(report1, encoding="utf-8")
    (CAPTURES / "demo2_compare.txt").write_text(report2, encoding="utf-8")
    _write_source(backend="fake", model="fake-lm", revision=None)
    return report1 + report2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("hf", "fake"),
        default="hf",
        help="hf: live sshleifer/tiny-gpt2 via the CLI. fake: in-repo adapter.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow --backend fake to overwrite checked-in Hugging Face fixtures.",
    )
    args = parser.parse_args()
    if args.backend == "fake":
        if not args.force:
            sys.stderr.write(
                "error: --backend fake overwrites real tiny-gpt2 fixtures; pass --force\n"
            )
            return 2
        sys.stdout.write(capture_fake())
        return 0
    sys.stdout.write(capture_hf())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
