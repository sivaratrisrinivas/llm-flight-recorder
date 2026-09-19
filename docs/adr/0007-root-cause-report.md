# ADR 0007: Root-cause vs downstream-effect compare report

Status: accepted  
Date: 2026-09-19

## Context

Milestone 5 names the first causal split and tags later event diffs as
downstream. That classification is enough for tests, but the human report
still read like a dump: config rows, a first-divergence block, then later
steps. A reader can mistake a later logit or context diff for a second root
cause (Case E).

Milestone 6 is the report framing. It does not change the M5 class of the
first split. It does not invent logits. Polished CLI UX remains M7.

## Decision

1. **`llmfr compare` keeps one command.** The text report (and `--json`
   `CompareResult`) grow fields. No new subcommand.

2. **Fixed section order.** CONFIG DIFFERENCE, EXECUTION, FIRST BEHAVIORAL
   DIVERGENCE, LIKELY ENABLING CONFIG, then Steps N+: downstream effects.

3. **CONFIG DIFFERENCE stays complete.** Every run-level field of interest
   that differs is listed, including diffs that did not enable the first
   split.

4. **EXECUTION states the matching prefix.** How many leading steps still
   share sampled tokens, model-visible context, and captured logits, plus
   recorded lengths.

5. **LIKELY ENABLING CONFIG is a filter, not a new class.** It is the subset
   of `config_diffs` that likely enabled the M5 first-divergence class
   (seed for sampling, sampler fields for decoding config, model id/revision
   for model/version, prompt fields for prompt/history). Unrelated diffs
   (device next to a seed split, tokenizer next to a prompt split) stay out.
   Window truncation has no v1 generation_config field; the report says so
   rather than inventing one.

6. **Downstream language is explicit.** Later context, logit, and token
   diffs are labeled not a new root cause. The report never assigns
   `class: raw-logit` (or any first-divergence class) to those later steps.

## Consequences

- Tests can lock section presence/order and Case E wording without live
  models.
- M7 can restyle the same sections. It should not reclassify first
  divergence to make the narrative prettier.

## Alternatives considered

- A second `llmfr explain` command: splits the story from the classifier
  and invites drift.
- Re-running later prefixes to reclassify logit diffs: that is the Case E
  lie.
