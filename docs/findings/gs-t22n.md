# GS-T22n: Does first-divergence class predict wrong answers?

Status: measured (null)
Date: 2026-09-19T23:05:23.879349+00:00

## Question

When two recorded generations split, does the M5 first-divergence class
(`sampling` vs `decoding config`) predict whether the completions disagree
on correctness? This is a small descriptive count, like SunkeLo's GS-T7
absent-claim / grounding number: report the measured rates, including a
null, and do not invent a metric the run did not produce.

## Setup

- **N items:** 8 gradeable integer questions (16 pairs;
  2 pair kinds per item).
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- **Library default / CI smoke:** `sshleifer/tiny-gpt2` is not used
  for this finding.
- **Sampling pairs:** seed 1 vs 2,
  temperature 1.0.
- **Decoding-config pairs:** seed 1,
  temperature 0.7 vs 1.2.
- **max_new_tokens:** 64;
  **capture_k:** 5.
- **Grading rule:** Extract the last whole number in output_text. A token matching -?\d+(?:\.\d+)? is a whole number iff float(token).is_integer(). Grade correct if that integer equals gold, wrong if it differs, no_answer if none is found. No LLM judge.
- **Verdict rule:** positive if both sampling and decoding-config disagree_rate are defined, each n_gradeable_diverged >= 4, and the higher rate is at least 2.0x the lower (or lower is 0 and higher > 0); else null. Not a p-value.
- **Hardware:** Linux-6.12.94+-x86_64-with-glibc2.39; Python 3.12.3; cpus=4.
- **Library versions:** {'torch': '2.14.0+cpu', 'transformers': '5.17.0'}
- **Backend:** `hf` (live recorder + `compare_traces`
  on stored events; compare does not call a model).

Re-run:

```bash
python scripts/gs_t22n_divergence_accuracy.py --backend hf
```

CI smoke (fake adapter; must not overwrite this finding):

```bash
python scripts/gs_t22n_divergence_accuracy.py --backend fake --out /tmp/llmfr-gs-t22n-fake
```

Raw JSON: `docs/findings/gs-t22n-results.json`.

## Result

**null.** Among gradeable diverged pairs, sampling disagree rate is
0.375 (3/8);
decoding-config disagree rate is
0.375 (3/8).
Wrong-answer rate among gradeable traces is
0.6875 for sampling pairs and
0.6875 for decoding-config pairs.

## Table

| class | n pairs | n gradeable diverged | agree correct | agree wrong | disagree | ungraded | disagree rate | wrong-answer rate | wrong/gradeable traces |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sampling | 8 | 8 | 1 | 4 | 3 | 0 | 0.375 | 0.6875 | 11/16 |
| decoding config | 8 | 8 | 1 | 4 | 3 | 0 | 0.375 | 0.6875 | 11/16 |

Identical pairs (no first behavioral divergence): 0.
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

## Limits

- N is small on purpose. This is not a leaderboard and not a p-value.
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
- `--backend fake` is smoke. It must not be quoted as the table above.
