# ADR 0011: Graded sampling vs decoding-config study CLI

Status: accepted  
Date: 2026-09-20

## Context

GS-T22n measured whether first-divergence class (`sampling` vs
`decoding config`) predicts correctness disagreement. That work lived in
`scripts/gs_t22n_divergence_accuracy.py`. A stranger should not have to
read that script to run the same kind of study. GS-T22o already loads a
prompt file for `llmfr record --prompts`. The study surface should reuse
that file format, `record_generation`, and `compare_traces`.

## Decision

1. **First-class command.** `llmfr study PROMPTS` records two splits for
   every item, grades completions, and prints a small table (or `--json`).
   Hugging Face is the default. OpenAI is opt-in (`--provider openai` or an
   `openai:` model prefix). There is no API-key flag.

2. **Prompt file.** Same UTF-8 formats as ADR 0010. JSONL/JSON objects
   must include integer `gold`. Per-line `seed`, `temperature`, `greedy`,
   and `max_new_tokens` are errors: the study owns those as CLI splits.
   A `.txt` file has no gold, so it is an error.

3. **Splits.** Sampling: `--seeds` (default `1,2`) at `--sampling-temperature`
   (default `1.0`). Decoding-config: `--temperatures` (default `0.7,1.2`)
   at `--decoding-seed` (default `1`). `--max-new-tokens` defaults to 64.
   Each side is a normal `record_generation`. Compare classifies stored
   events and does not call a model.

4. **Grading.** Last whole number in `output_text` vs `gold`. A token
   matching `-?\d+(?:\.\d+)?` is a whole number iff `float(token).is_integer()`.
   Grade `correct` / `wrong` / `no_answer`. Pair outcome is `agree_correct`,
   `agree_wrong`, `disagree`, or `ungraded` when either side is `no_answer`.
   No LLM judge. Prompt numerals in a truncated completion can become the
   last number; that is a documented limit, not a hidden rewrite.

   `disagree_rate` is disagree / gradeable diverged pairs. `wrong_answer_rate`
   is wrong traces / gradeable traces among those same pairs. Ungraded pairs
   (any `no_answer` side) are excluded from both rates.

   Compare `identical` is true only when there is no first event divergence
   and no config diff. A sampling or decoding-config split that changes seed
   or temperature but not sampled tokens is `no first divergence`, not
   identical. The table counts those separately. `llmfr study` does not
   emit a positive/null verdict; that label is GS-T22n-only and is not a
   p-value.

5. **Output.** Exit 0 prints the table (class counts, disagree rate,
   wrong-answer rate, per-pair grades). `--json` prints the same payload.
   `--redact` grades first, then redacts store writes; JSON omits
   `output_text`. `--no-persist` still grades in memory. A record failure
   (including omitted OpenAI logprobs) is fail closed, exit 1.

## Consequences

- Library callers can use `run_study`, `format_study_report`, and the
  grader in `llmfr.study`.
- The GS-T22n finding script imports the same grader so the rule cannot
  drift.
- Tests drive the CLI with fake adapters. CI does not download large
  models or call a live API.

## Alternatives considered

- Keep the experiment script as the only entry: cheaper, but a stranger
  has to read `scripts/`.
- LLM-as-judge: richer, but this tree does not invent a judge or scores.
- Continue after a failed item: more rows, but easier to miss a
  fail-closed OpenAI response.
