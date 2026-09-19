# ADR 0008: Production CLI (Typer)

Status: accepted  
Date: 2026-09-19

## Context

Milestones 3 through 6 shipped working `record`, `replay`, and `compare`
commands on argparse. Help text still said "minimal", inspect was only
`validate` / `topk` on files, and exit codes were implied rather than
documented. Milestone 7 is the production CLI: the same TraceStore,
recorder, replay, and compare code, with inspect-at-step and a documented
happy path (record twice, then compare).

Privacy hardening (M8), a compare UI (M9), and portfolio polish (M10) stay
out of scope. Help must not invite pasting secrets.

## Decision

1. **Typer, same command names.** `llmfr record`, `llmfr replay`,
   `llmfr compare`, plus `llmfr inspect`. `validate` and `topk` remain so
   M1 file workflows keep working. `inspect --step N` is the step view:
   full_history vs model_visible_context, sampled token, and that step's
   top-k.

2. **Expected failures print `error:` and never a traceback.** Unknown
   ids, missing files, invalid traces, out-of-range `--step`, missing
   Hugging Face extra, and validation errors are exit 1.

3. **Exit codes are stable.** 0 success (compare identical, replay
   `reproduced`); 1 expected failure including diverged compare and
   non-reproduced replay; 2 usage error. Root `--help` and the README
   state this.

4. **Inspect and compare accept a file path or a TraceStore id.** Replay
   stays store-id-only (it rebuilds an adapter from the stored trace).
   Record still writes a store and prints `trace_id`.

5. **No invented logits.** Inspect reports `logits.mode=none` honestly.
   No API-key flags. OpenAI recording reads `OPENAI_API_KEY` from the
   environment when `--provider openai` or an explicit `openai:` model prefix
   is selected. Bare `gpt-*` names stay Hugging Face. Hugging Face stays the
   default for local/CI.

## Consequences

- Tests can drive `llmfr.cli.run(argv)` and still monkeypatch
  `_build_hf_adapter` / `_replay`.
- Definition of Done is CLI-shaped: record two traces, inspect a step,
  compare.
