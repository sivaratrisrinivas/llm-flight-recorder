#!/bin/sh
# GS-T26: run the GS-T22s latency job and always write /out artifacts.
# Extra args override argparse flags (last flag wins).
set -eu
OUT="${LLMFR_OUT:-/out}"
mkdir -p "$OUT/scratch"
if command -v python >/dev/null 2>&1; then
  PY=python
else
  PY=python3
fi
exec "$PY" scripts/gs_t22s_latency.py \
  --out "$OUT/scratch" \
  --results "$OUT/results.json" \
  --finding "$OUT/finding.md" \
  "$@"
