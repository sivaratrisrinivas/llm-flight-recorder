# Eval audit (eval-audit skill, 2026-10-03)

Routed by evals-start: an eval pipeline exists (answer grader vs hand labels, synthetic grader suite, divergence suite, CI gate), so `eval-audit`. Findings by impact.

## 6. Pipeline hygiene

### Grader validated on traces from the old study settings
**Status:** Fixed in part in this branch.
The grader's held-out check used 64-token traces. The study now records 256 tokens with stop strings, so the grader had never been checked on the traces it now grades. 20 of the 40 held-out prompts were re-recorded at the new default and labeled before scoring (`data/heldout256_labels.jsonl`).
Result: new rule TPR 0.85 (17/20), old last-number rule 0.90 (18/20). On clean runs the new rule is no better than the old one; it helps on truncated runs. The set is now gated in CI (floor 0.85). The misses were not tuned away, since this is test data.
**Remaining:** record the other 20 prompts, include runs with no answer so TNR is measured at 256 tokens, and re-run GS-T22q at 256 tokens.

## 5. Labeled data

### Held-out set is small
**Status:** Problem exists. 40 traces at 64 tokens (11 with an answer, 29 without) plus 20 at 256 tokens (all with an answer). That is about 31 positives and 29 negatives, below the roughly 50/50 needed for tight TPR/TNR. Recording takes about 1 to 1.5 minutes of CPU per trace on this box.
**Fix:** the monitoring loop in this README (sample about 20 traces per study run, label, append).

## 4. Human review

**Status:** Problem exists. All labels are agent-made (each row has a `labeler` field). The task is arithmetic, so no domain expert is needed, but a person should spot-check the ambiguous rows (truncated runs that change their answer).

## 1. Error analysis

**Status:** OK. Failure modes were observed by reading 120 real traces (truncation, text after the end token, bleeding into a new problem) before the grader was written.

## 2. Evaluator design

**Status:** OK. The grader is code, binary per trace (answer found or not, correct or not). No LLM judge and no similarity metric.

## 3. Judge validation

**Status:** OK for a code grader. Scored with TPR/TNR per class on held-out data labeled after the rule was frozen. The dev number is reported as optimistic.
