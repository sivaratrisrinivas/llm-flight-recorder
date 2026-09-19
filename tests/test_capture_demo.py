"""capture_demo.py guards: no fake overwrite of docs-bound files; fail on empty CLI captures."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from llmfr.adapters.huggingface import (
    DEFAULT_HF_MODEL_ID,
    PORTFOLIO_DEMO_MODEL_ID,
    PORTFOLIO_DEMO_MODEL_REVISION,
)

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "capture_demo.py"
DEMO_DIR = REPO / "examples" / "demo"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("llmfr_capture_demo", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capture_demo_hf_pins_portfolio_model_not_ci_default() -> None:
    script = _load_script()
    assert script.PORTFOLIO_DEMO_MODEL_ID == PORTFOLIO_DEMO_MODEL_ID
    assert script.DEFAULT_HF_MODEL_ID == DEFAULT_HF_MODEL_ID
    assert script.PORTFOLIO_DEMO_MODEL_ID != script.DEFAULT_HF_MODEL_ID
    assert script.PORTFOLIO_DEMO_MODEL_REVISION == PORTFOLIO_DEMO_MODEL_REVISION
    assert "Think step by step" in script.PROMPT
    assert script.SMOKE_PROMPT == "Hello"
    assert script.SMOKE_MAX_NEW_TOKENS == 6
    assert script.MAX_NEW_TOKENS == 16


def test_fake_backend_refuses_docs_bound_dir(capsys: pytest.CaptureFixture[str]) -> None:
    script = _load_script()
    source_before = (DEMO_DIR / "SOURCE.txt").read_text(encoding="utf-8")
    inspect_before = (DEMO_DIR / "demo1_inspect_step0.txt").read_text(encoding="utf-8")
    assert script.main(["--backend", "fake"]) == 2
    err = capsys.readouterr().err
    assert "refusing to overwrite docs-bound captures" in err
    assert script.main(["--backend", "fake", "--force"]) == 2
    err_force = capsys.readouterr().err
    assert "refusing to overwrite docs-bound captures" in err_force
    assert "--force does not override this" in err_force
    assert script.main(["--backend", "fake", "--out", str(DEMO_DIR)]) == 2
    assert (DEMO_DIR / "SOURCE.txt").read_text(encoding="utf-8") == source_before
    assert (DEMO_DIR / "demo1_inspect_step0.txt").read_text(encoding="utf-8") == inspect_before
    assert "backend=huggingface" in source_before
    assert PORTFOLIO_DEMO_MODEL_ID in source_before


def test_fake_backend_scratch_out_writes_inspect(tmp_path: Path) -> None:
    script = _load_script()
    out = tmp_path / "scratch"
    source_before = (DEMO_DIR / "SOURCE.txt").read_text(encoding="utf-8")
    assert script.main(["--backend", "fake", "--out", str(out)]) == 0
    inspect = (out / "demo1_inspect_step0.txt").read_text(encoding="utf-8")
    source = (out / "SOURCE.txt").read_text(encoding="utf-8")
    compare = (out / "demo1_compare.txt").read_text(encoding="utf-8")
    assert inspect.strip()
    assert "step 0 of" in inspect
    assert "backend=fake" in source
    assert "backend=huggingface" not in source
    assert "FIRST BEHAVIORAL DIVERGENCE" in compare
    assert (DEMO_DIR / "SOURCE.txt").read_text(encoding="utf-8") == source_before


def test_checked_run_fails_on_bad_status_or_empty_stdout() -> None:
    script = _load_script()
    with pytest.raises(RuntimeError, match="failed \\(2\\)"):
        script._checked_run(
            [sys.executable, "-c", "raise SystemExit(2)"],
            name="inspect demo1 step 0",
            allowed=script._OK_EXIT,
        )
    with pytest.raises(RuntimeError, match="empty stdout"):
        script._checked_run(
            [sys.executable, "-c", "raise SystemExit(0)"],
            name="inspect demo1 step 0",
            allowed=script._OK_EXIT,
        )
    with pytest.raises(RuntimeError, match="empty stdout"):
        script._checked_run(
            [
                sys.executable,
                "-c",
                "import sys; sys.stderr.write('error: boom\\n'); raise SystemExit(1)",
            ],
            name="compare demo1",
            allowed=script._COMPARE_EXIT,
        )
