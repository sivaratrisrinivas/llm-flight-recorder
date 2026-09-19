# ADR 0010: Batch record from a prompt file / JSONL

Status: accepted  
Date: 2026-09-19

## Context

`llmfr record PROMPT` records one generation per CLI invocation. A stranger
who wants many traces should not have to wrap that in a shell loop, and
should not have to guess the input format. Batch recording must keep the
same honesty rules as a single `record`: Hugging Face is the default,
OpenAI is opt-in, omitted OpenAI logprobs fail closed, and secrets stay in
the environment.

## Decision

1. **Same command.** `llmfr record --prompts PATH` records every prompt in
   the file. `PROMPT` and `--prompts` are mutually exclusive (exit 2).
   CLI flags (`--store`, `--model`, `--provider`, `--revision`, `--seed`,
   `--temperature`, `--greedy`, `--max-new-tokens`, `--capture-k`,
   `--redact`, `--no-persist`, `--format`) apply to every item. The adapter
   is built once.

2. **Input formats** (UTF-8). Surrounding whitespace on each prompt is
   stripped. Blank lines are skipped. An empty list is an error (exit 1).

   **`.jsonl`** — one JSON value per line:

   | Field | Type | Required | Notes |
   | --- | --- | --- | --- |
   | `prompt` | string | yes | Non-empty after strip. |
   | `id` | string | no | Stored as `run_metadata.tags["id"]`. |
   | `seed` | int | no | Overrides `--seed` for this item. |
   | `temperature` | float | no | Overrides `--temperature`. |
   | `max_new_tokens` | int >= 1 | no | Overrides `--max-new-tokens`. |
   | `greedy` | bool | no | Overrides `--greedy`. |
   | `tags` | object of string → string | no | Merged into `run_metadata.tags`. |

   A JSON string per line (`"Hello"`) is the same as `{"prompt":"Hello"}`.
   Unknown keys are an error. `provider`, `model`, and `api_key` are not
   fields; backend selection stays on the CLI.

   **`.json`** — a JSON array of those strings/objects, or a single
   string/object.

   **Other suffixes** (including `.txt`) — one prompt per line.

3. **Output.** Exit 0 prints one `trace_id` per prompt, in file order, one
   per stdout line. Each trace is tagged `batch_index` (`"0"`, `"1"`, …).
   The first failing item stops the batch (fail closed). Items already
   written stay in the store. `--redact` and `--no-persist` apply to every
   item the same way they do for a single record.

4. **Honesty is unchanged.** Hugging Face remains the default. OpenAI is
   selected only with `--provider openai` or an `openai:` model prefix;
   `--provider` overrides the prefix. Bare `gpt-*` names stay Hugging Face.
   `OPENAI_API_KEY` is read from the environment only (no `--api-key`, and
   no key in the prompt file). If OpenAI omits per-token logprob content,
   that item fails closed; llmfr does not invent scores.

## Consequences

- Library callers can use `load_prompt_file` and `record_prompt_batch`.
- Tests parse files and drive the CLI with fake adapters; CI does not
  download large models or call a live API.
- A later change that auto-selects OpenAI from a JSONL `model` field would
  need a new ADR.

## Alternatives considered

- A separate `record-batch` command: clearer isolation, but a second verb
  for the same flags and honesty rules.
- Continue after a failed item: more traces, but easier to miss a fail-closed
  OpenAI response in a long file.
- Per-line `provider` / `model` / `api_key`: invites silent backend switches
  and keys in files.
