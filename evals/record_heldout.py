"""Record one real trace per held-out prompt for grader validation.

Needs the [hf] extra. Writes JSONL rows with the output text and whether the
run hit max_new_tokens. These rows are labeled by hand after the grader rule
is frozen, so they act as a held-out test set.

python evals/record_heldout.py evals/data/heldout_prompts.jsonl out.jsonl --max-new-tokens 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llmfr.adapters.huggingface import (
    PORTFOLIO_DEMO_MODEL_ID,
    PORTFOLIO_DEMO_MODEL_REVISION,
    HuggingFaceCausalLMAdapter,
)
from llmfr.core.schema import GenerationConfig
from llmfr.record.recorder import record_generation
from llmfr.study.run import STUDY_STOP


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("prompts")
    p.add_argument("out")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--stop-at-eos", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    items = [json.loads(line) for line in Path(args.prompts).read_text().splitlines() if line]
    if args.limit:
        items = items[: args.limit]
    done = set()
    out = Path(args.out)
    if out.exists():
        done = {json.loads(line)["id"] for line in out.read_text().splitlines() if line}
    lm = HuggingFaceCausalLMAdapter(
        PORTFOLIO_DEMO_MODEL_ID, revision=PORTFOLIO_DEMO_MODEL_REVISION, device="cpu"
    )
    gen = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=1.0,
        seed=args.seed,
        stop=list(STUDY_STOP) if args.stop_at_eos else None,
    )
    with out.open("a", encoding="utf-8") as fh:
        for i, item in enumerate(items):
            if item["id"] in done:
                continue
            trace = record_generation(lm, item["prompt"], generation=gen, persist=False)
            row = {
                "id": item["id"],
                "prompt": item["prompt"],
                "gold": item["gold"],
                "tags": item.get("tags", {}),
                "output": trace.run_metadata.output_text,
                "n_tokens": len(trace.events),
                "truncated": len(trace.events) >= args.max_new_tokens,
                "max_new_tokens": args.max_new_tokens,
                "seed": args.seed,
                "model": PORTFOLIO_DEMO_MODEL_ID,
                "revision": PORTFOLIO_DEMO_MODEL_REVISION,
            }
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"{i + 1}/{len(items)} {item['id']} tokens={row['n_tokens']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
