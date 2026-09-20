# GS-T22q: Scaled sampling vs decoding-config correctness disagreement

Status: measured (GS-T22n descriptive label: null)
Date: 2026-09-20T00:52:41.350361+00:00
Supersedes: N=8 GS-T22n (`docs/findings/gs-t22n.md`).

## Question

When two recorded generations split, does the M5 first-divergence class
(`sampling` vs `decoding config`) predict whether the completions disagree
on correctness, at a larger N than GS-T22n? Report the measured rates.
Do not invent a metric the run did not produce. `llmfr study` does not
emit a positive/null verdict; the GS-T22n descriptive label below uses
the same documented 2x rule and is not a p-value.

## Setup

- **N items:** 30 gradeable integer questions (60 pairs;
  2 pair kinds per item).
- **N completed:** 30 of 30 listed prompts.
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- **Library default / CI smoke:** `sshleifer/tiny-gpt2` is not used
  for this finding. Llama was not recorded (gated Hub repo; no silent
  substitution).
- **Sampling pairs:** seed 1 vs 2,
  temperature 1.0.
- **Decoding-config pairs:** seed 1,
  temperature 0.7 vs 1.2.
- **max_new_tokens:** 64;
  **capture_k:** 5.
- **Grading rule:** Extract the last whole number in output_text. A token matching -?\d+(?:\.\d+)? is a whole number iff float(token).is_integer(). Grade correct if that integer equals gold, wrong if it differs, no_answer if none is found. No LLM judge.
- **Descriptive label rule:** GS-T22n descriptive label only (not emitted by llmfr study; not a p-value): positive if both sampling and decoding-config disagree_rate are defined, each n_gradeable_diverged >= 4, and the higher rate is at least 2.0x the lower (or lower is 0 and higher > 0); else null.
- **Hardware:** Linux-6.12.94+-x86_64-with-glibc2.39; Python 3.12.3; cpus=4.
- **Library versions:** {'torch': '2.14.0+cpu', 'transformers': '5.17.0'}
- **Backend:** `hf` via `llmfr study` (live recorder + `compare_traces`
  on stored events; compare does not call a model).

Record (does not write this finding):

```bash
llmfr study examples/findings/gs-t22q/prompts.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775 --store .llmfr-gs-t22q --max-new-tokens 64 --capture-k 5 --json
```

Pack named traces and regenerate this file from that JSON + store:

```bash
python scripts/gs_t22q_pack_study.py --from-json STUDY.json --from-store .llmfr-gs-t22q --write-docs
```

Recompute the table from checked-in JSON with no model:

```bash
python scripts/gs_t22q_pack_study.py --from-results docs/findings/gs-t22q-results.json
```

Raw JSON: `docs/findings/gs-t22q-results.json`.
Prompts: `examples/findings/gs-t22q/prompts.jsonl`.

## Result

**Rates (raw).** Sampling disagree rate is
0.3333 (10/30);
decoding-config disagree rate is
0.4333 (13/30).
Wrong-answer rate among gradeable traces is
0.7 for sampling pairs and
0.75 for decoding-config pairs.
GS-T22n descriptive label: **null**.

## So-what

On N=30 `Qwen/Qwen2.5-0.5B-Instruct` items, sampling and decoding-config splits did not separate on this correctness-disagreement count (GS-T22n 2x descriptive label: null). First-divergence class is still a useful debug label; it did not predict wrong-answer disagreement here.

## Table

| class | n pairs | n gradeable diverged | agree correct | agree wrong | disagree | ungraded | disagree rate | wrong-answer rate | wrong/gradeable traces |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sampling | 30 | 30 | 4 | 16 | 10 | 0 | 0.3333 | 0.7 | 42/60 |
| decoding config | 30 | 30 | 1 | 16 | 13 | 0 | 0.4333 | 0.75 | 45/60 |

Pairs with no first event divergence: 0.
Other observed classes: none.

## Pair detail

- `sheep_trick` intended `sampling`: class `sampling` step=0 grades correct/wrong (extracted 9/3, gold 9) outcome `disagree`
- `sheep_trick` intended `decoding_config`: class `decoding config` step=8 grades correct/correct (extracted 9/9, gold 9) outcome `agree_correct`
- `seven_plus_five` intended `sampling`: class `sampling` step=0 grades wrong/wrong (extracted 7/1, gold 12) outcome `agree_wrong`
- `seven_plus_five` intended `decoding_config`: class `decoding config` step=12 grades correct/wrong (extracted 12/1, gold 12) outcome `disagree`
- `twelve_minus_four` intended `sampling`: class `sampling` step=0 grades wrong/wrong (extracted 12/12, gold 8) outcome `agree_wrong`
- `twelve_minus_four` intended `decoding_config`: class `decoding config` step=5 grades wrong/wrong (extracted 4/2, gold 8) outcome `agree_wrong`
- `three_times_six` intended `sampling`: class `sampling` step=0 grades correct/wrong (extracted 18/3, gold 18) outcome `disagree`
- `three_times_six` intended `decoding_config`: class `decoding config` step=2 grades wrong/wrong (extracted 3/6, gold 18) outcome `agree_wrong`
- `cows_sold` intended `sampling`: class `sampling` step=0 grades wrong/wrong (extracted 2/3, gold 7) outcome `agree_wrong`
- `cows_sold` intended `decoding_config`: class `decoding config` step=7 grades wrong/correct (extracted 3/7, gold 7) outcome `disagree`
- `hundred_div_four` intended `sampling`: class `sampling` step=0 grades wrong/wrong (extracted 100/4, gold 25) outcome `agree_wrong`
- `hundred_div_four` intended `decoding_config`: class `decoding config` step=2 grades wrong/wrong (extracted 4/3, gold 25) outcome `agree_wrong`
- `apples_left` intended `sampling`: class `sampling` step=0 grades correct/correct (extracted 6/6, gold 6) outcome `agree_correct`
- `apples_left` intended `decoding_config`: class `decoding config` step=1 grades correct/wrong (extracted 6/3, gold 6) outcome `disagree`
- `two_plus_two` intended `sampling`: class `sampling` step=0 grades wrong/correct (extracted 2/4, gold 4) outcome `disagree`
- `two_plus_two` intended `decoding_config`: class `decoding config` step=0 grades wrong/wrong (extracted 2/3, gold 4) outcome `agree_wrong`
- `nine_plus_eight` intended `sampling`: class `sampling` step=0 grades wrong/correct (extracted 9/17, gold 17) outcome `disagree`
- `nine_plus_eight` intended `decoding_config`: class `decoding config` step=12 grades correct/wrong (extracted 17/1, gold 17) outcome `disagree`
- `twenty_minus_six` intended `sampling`: class `sampling` step=0 grades correct/correct (extracted 14/14, gold 14) outcome `agree_correct`
- `twenty_minus_six` intended `decoding_config`: class `decoding config` step=17 grades wrong/wrong (extracted 20/2, gold 14) outcome `agree_wrong`
- `five_times_four` intended `sampling`: class `sampling` step=0 grades correct/correct (extracted 20/20, gold 20) outcome `agree_correct`
- `five_times_four` intended `decoding_config`: class `decoding config` step=1 grades correct/wrong (extracted 20/3, gold 20) outcome `disagree`
- `fifty_div_five` intended `sampling`: class `sampling` step=0 grades wrong/wrong (extracted 50/4, gold 10) outcome `agree_wrong`
- `fifty_div_five` intended `decoding_config`: class `decoding config` step=13 grades correct/wrong (extracted 10/5, gold 10) outcome `disagree`
- `eggs_left` intended `sampling`: class `sampling` step=0 grades correct/wrong (extracted 8/4, gold 8) outcome `disagree`
- `eggs_left` intended `decoding_config`: class `decoding config` step=8 grades correct/wrong (extracted 8/3, gold 8) outcome `disagree`
- `birds_remain` intended `sampling`: class `sampling` step=0 grades wrong/wrong (extracted 3/9, gold 5) outcome `agree_wrong`
- `birds_remain` intended `decoding_config`: class `decoding config` step=13 grades wrong/wrong (extracted 4/9, gold 5) outcome `agree_wrong`
- `three_plus_eleven` intended `sampling`: class `sampling` step=0 grades wrong/wrong (extracted 3/1, gold 14) outcome `agree_wrong`
- `three_plus_eleven` intended `decoding_config`: class `decoding config` step=8 grades wrong/wrong (extracted 25/3, gold 14) outcome `agree_wrong`
- `thirty_minus_ten` intended `sampling`: class `sampling` step=0 grades wrong/wrong (extracted 30/30, gold 20) outcome `agree_wrong`
- `thirty_minus_ten` intended `decoding_config`: class `decoding config` step=17 grades wrong/wrong (extracted 30/3, gold 20) outcome `agree_wrong`
- `two_times_nine` intended `sampling`: class `sampling` step=0 grades correct/correct (extracted 18/18, gold 18) outcome `agree_correct`
- `two_times_nine` intended `decoding_config`: class `decoding config` step=0 grades wrong/correct (extracted 100/18, gold 18) outcome `disagree`
- `eighty_div_ten` intended `sampling`: class `sampling` step=0 grades wrong/wrong (extracted 10/3, gold 8) outcome `agree_wrong`
- `eighty_div_ten` intended `decoding_config`: class `decoding config` step=14 grades wrong/wrong (extracted 1/80, gold 8) outcome `agree_wrong`
- `books_left` intended `sampling`: class `sampling` step=0 grades wrong/correct (extracted 8/7, gold 7) outcome `disagree`
- `books_left` intended `decoding_config`: class `decoding config` step=0 grades wrong/correct (extracted 1/7, gold 7) outcome `disagree`
- `cookies_remain` intended `sampling`: class `sampling` step=0 grades wrong/wrong (extracted 1/3, gold 9) outcome `agree_wrong`
- `cookies_remain` intended `decoding_config`: class `decoding config` step=0 grades correct/wrong (extracted 9/14, gold 9) outcome `disagree`
- `six_plus_seven` intended `sampling`: class `sampling` step=0 grades wrong/wrong (extracted 1/7, gold 13) outcome `agree_wrong`
- `six_plus_seven` intended `decoding_config`: class `decoding config` step=12 grades correct/wrong (extracted 13/7, gold 13) outcome `disagree`
- `eighteen_minus_nine` intended `sampling`: class `sampling` step=0 grades correct/wrong (extracted 9/3, gold 9) outcome `disagree`
- `eighteen_minus_nine` intended `decoding_config`: class `decoding config` step=17 grades wrong/wrong (extracted 1/2, gold 9) outcome `agree_wrong`
- `four_times_seven` intended `sampling`: class `sampling` step=0 grades correct/wrong (extracted 28/4, gold 28) outcome `disagree`
- `four_times_seven` intended `decoding_config`: class `decoding config` step=3 grades wrong/wrong (extracted 4/7, gold 28) outcome `agree_wrong`
- `sixty_div_three` intended `sampling`: class `sampling` step=0 grades wrong/wrong (extracted 3/6, gold 20) outcome `agree_wrong`
- `sixty_div_three` intended `decoding_config`: class `decoding config` step=13 grades wrong/wrong (extracted 3/3, gold 20) outcome `agree_wrong`
- `pencils_left` intended `sampling`: class `sampling` step=0 grades correct/wrong (extracted 11/3, gold 11) outcome `disagree`
- `pencils_left` intended `decoding_config`: class `decoding config` step=0 grades wrong/correct (extracted 3/11, gold 11) outcome `disagree`
- `chairs_remain` intended `sampling`: class `sampling` step=0 grades wrong/wrong (extracted 6/6, gold 4) outcome `agree_wrong`
- `chairs_remain` intended `decoding_config`: class `decoding config` step=13 grades wrong/wrong (extracted 6/6, gold 4) outcome `agree_wrong`
- `one_plus_nineteen` intended `sampling`: class `sampling` step=0 grades wrong/wrong (extracted 1/19, gold 20) outcome `agree_wrong`
- `one_plus_nineteen` intended `decoding_config`: class `decoding config` step=13 grades wrong/wrong (extracted 1/1, gold 20) outcome `agree_wrong`
- `forty_minus_fifteen` intended `sampling`: class `sampling` step=0 grades wrong/wrong (extracted 4/40, gold 25) outcome `agree_wrong`
- `forty_minus_fifteen` intended `decoding_config`: class `decoding config` step=5 grades wrong/correct (extracted 15/25, gold 25) outcome `disagree`
- `eight_times_three` intended `sampling`: class `sampling` step=0 grades correct/wrong (extracted 24/3, gold 24) outcome `disagree`
- `eight_times_three` intended `decoding_config`: class `decoding config` step=1 grades wrong/wrong (extracted 2/6, gold 24) outcome `agree_wrong`
- `ninety_div_nine` intended `sampling`: class `sampling` step=0 grades wrong/wrong (extracted 9/9, gold 10) outcome `agree_wrong`
- `ninety_div_nine` intended `decoding_config`: class `decoding config` step=21 grades wrong/wrong (extracted 2/90, gold 10) outcome `agree_wrong`

## Limits

- This is not a leaderboard and not a p-value.
- Completions are truncated at 64 new tokens.
  A restated prompt numeral can become the last whole number; that
  grades `wrong` or accidentally `correct` without a hidden fix-up.
- `no_answer` pairs are ungraded, not filled in.
- 0.5B-class instruct model on CPU, no chat template (same raw-prompt
  style as the portfolio demo).
- First-divergence class is the stored-event classifier (ADR 0006).
  Later token diffs are downstream and are not a second root cause
  (ADR 0007).
- tiny-gpt2 answers are unreadable; they are not this finding.
- Llama-3.2-1B-Instruct was not used (gated; no HF_TOKEN).
