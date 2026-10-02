# llmfr evals

Two things in llmfr make judgments that can be wrong, so both are evaluated:

1. **The study answer grader** (`llmfr.study.grade`): pulls a final number out of
   model text and grades it against the gold answer.
2. **First-divergence classification** (`llmfr compare`): names why two traces
   split (sampling, decoding config, runtime logits, prompt, model).

Everything here runs on CPU with no model download: `make eval` or
`python evals/run_evals.py`. CI runs it on every push and fails if any number
drops below `evals/baseline.json`.

## How the failure modes were found

Error analysis on 120 real traces from the GS-T22q study
(`examples/findings/gs-t22q`, Qwen2.5-0.5B-Instruct, 64 new tokens). Reading every
trace and labeling the answer the model actually committed to gave:

| Label | Traces |
|---|---|
| no answer yet (cut off before committing) | 72 |
| correct | 47 |
| wrong | 1 |

The old grader (take the last whole number in the text) called 87 of these
"wrong". Its failure modes, in order of frequency:

- **Truncated at 64 tokens** before any answer, so the last number was a step
  label ("Step 3"), a number from the prompt, or a number cut in half ("= 1" for 12).
- **Text after `<|endoftext|>`**: the model finished, then started a new made-up
  problem whose numbers were graded.
- **Bleeding into a new problem** without an end token.

Root cause on the recording side: the study never stopped at the end token. Fixes:

- `extract_final_answer` and `FINAL_ANSWER_RULE`. Order: explicit answer markers
  ("the answer is", "Answer:", boxed), then the last completed equation or
  "equals" not followed by another operator. Text after an end token is dropped.
  Trailing numbers in truncated text are ignored. A bare-number reply counts only
  if the run was not truncated.
- `llmfr study` now passes end-of-text stop strings and defaults to 256 new
  tokens (was 64), and records whether each side hit the token limit.
- The old rule is kept as `grade_output_last_number` / `LAST_NUMBER_RULE` so the
  published GS-T22n and GS-T22q scripts still reproduce their numbers.

## Eval sets

| Set | Size | What it checks | Source |
|---|---|---|---|
| Real dev traces | 120 | grader vs hand label | GS-T22q traces; rule was written while reading these |
| Real held-out traces | 40 | grader vs hand label | 40 new prompts (`data/heldout_prompts.jsonl`, op x phrasing x magnitude), recorded with `record_heldout.py` and labeled after the rule was frozen |
| Synthetic answer cases | 520 | grader regression suite | `answer_cases.py`: preamble x how the answer is committed x ending (clean, end token, truncated, new problem) x distractor numbers x operation |
| Divergence pairs | 816 | compare names the injected cause | `divergence_eval.py`: real `record_generation` + `compare_traces` on a fake model with context-hashed logits, across cause x logit shape x prompt length x sampler x length x capture_k x seed |

TPR here means: of traces with a labeled answer, the share where the grader found
that exact number. TNR: of traces with no labeled answer, the share where the
grader returned no answer.

## Results (2026-10-02)

| Set | New rule | Old last-number rule |
|---|---|---|
| Real dev (n=120) | TPR 0.979, TNR 0.986, grade agreement 0.983 | TPR 0.604, TNR 0.000, agreement 0.250 |
| Real held-out (n=40) | TPR 0.818 (9/11), TNR 1.000 (29/29), agreement 0.950 | TPR 0.545, TNR 0.000, agreement 0.150 |
| Synthetic (n=520) | 520/520 | 86/520 |
| Divergence (n=816) | 812/816 | n/a |

The held-out number is the honest one. The dev number is optimistic because the
rule was tuned on those traces. Held-out misses (left in on purpose, not tuned
away): "The division gives you 20." (no marker or equation) and
"43 (existing cards) + 37 (found cards) = 80 cards" (words inside the equation).

The 4 divergence misses are all `runtime_logits` with `capture_k=1` and a peaked
distribution: the stored top-1 logit did not change, so compare honestly reports
a probability-distribution split. Capture k >= 2 if you need that distinction.

### Held-out traces at the new study default (2026-10-03)

The 40 held-out traces above were recorded at 64 tokens, before the study switched to 256 tokens with
stop strings. So the grader had not been checked on the traces the study now produces. The first 20
held-out prompts were re-recorded at 256 tokens with stop strings (`record_heldout.py --max-new-tokens
256 --stop-at-eos`; the other 20 were not, because of CPU time) and labeled from the raw text before
scoring (`data/heldout256_labels.jsonl`).

| Set | New rule | Old last-number rule |
|---|---|---|
| Held-out, 256 tokens + stop (n=20, all with an answer) | TPR 0.85 (17/20) | TPR 0.90 (18/20) |

In plain words: once runs finish cleanly, most answers are the last number anyway, and the new rule
is no better than the old one on this small set. Its gain is on truncated runs (the 64-token sets
above). New-rule misses: "43 (existing cards) + 37 (found cards) = 80 cards ... the final number of
cards Arjun has is 80" (words inside the equation, the same miss as before, and "final number ... is"
is not an answer marker); a rambling truncated run whose last completed equation is a side
calculation (13500 instead of 135); and a truncated run that never settles (label 20, both rules say
108; that label is debatable). They were not tuned away, because this is a test set. Add these
patterns to `answer_cases.py` and fix them on dev data first. There are no no-answer traces in this
set, so TNR is not measured at 256 tokens.

### What this means for GS-T22q

Re-grading the same 60 pairs:

| Split | Old grader | New grader | Hand labels |
|---|---|---|---|
| sampling | agree correct 4, agree wrong 16, disagree 10 | agree correct 5, disagree 2, ungraded 23 | agree correct 6, disagree 1, ungraded 23 |
| decoding config | agree correct 1, agree wrong 16, disagree 13 | agree correct 2, ungraded 28 | agree correct 2, ungraded 28 |

The published 0.70 / 0.75 wrong-answer rates and the 10/30 and 13/30 disagree
counts come from the old grader and mostly measure truncation. The "null" verdict
still holds, but the real reason is that most pairs never reached an answer at 64
tokens. A full re-run of the study at 256 tokens with stop strings is the fix; it was not run here
(CPU time). Only the 20 held-out grader traces above were re-recorded at 256 tokens.

## Caveats

- Labels were made by an AI agent, not a person. They need a human spot check;
  each row carries a `labeler` field.
- The held-out set is small (40 traces, 11 with an answer), so TPR has wide error
  bars. Grow it with the monitoring loop below.
- The divergence suite uses a fake model, so it tests compare logic, not any
  real backend's numerics.

## Ongoing monitoring

After each real `llmfr study` run:

1. Sample about 20 traces into a labeling file:
   `python evals/record_heldout.py <new prompts> out.jsonl` for new prompts, or
   copy rows from the study output.
2. Label `label_answer` by hand (null if no committed answer) and append to
   `evals/data/heldout_labels.jsonl`.
3. Run `python evals/run_evals.py`. If held-out TPR or TNR drops, read the misses,
   add each new failure pattern to `answer_cases.py` as synthetic cases, and only
   then change the rule. Re-label after any model or prompt change.
4. Raise `evals/baseline.json` only when numbers improve on data the rule was not
   tuned on.
