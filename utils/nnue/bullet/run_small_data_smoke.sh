#!/usr/bin/env bash
set -euo pipefail

resolve_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    printf '%s\n' "${PYTHON_BIN}"
    return
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
  if command -v pyenv >/dev/null 2>&1; then
    pyenv which python
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  echo "No usable Python interpreter found (python/pyenv/python3)." >&2
  exit 1
}

PYTHON_BIN="$(resolve_python_bin)"
RUN_ID="${RUN_ID:-smoke_$(date -u +%Y%m%dT%H%M%SZ)}"

"${PYTHON_BIN}" utils/nnue/bullet/build_mixed_dataset.py \
  --python-bin "${PYTHON_BIN}" \
  --run-id "${RUN_ID}" \
  --fishtest-hours 12 \
  --fishtest-tc-lower-limit 1 \
  --fishtest-tc-upper-limit 20 \
  --fishtest-max-runs 1 \
  --fishtest-max-games 120000 \
  --max-fishtest-pgns 1 \
  --include-existing-fishtest \
  --selfplay-games 8 \
  --selfplay-movetime-ms 60 \
  --selfplay-shuffle-openings \
  --teacher-depth 8 \
  --sample-rate 0.35 \
  --max-positions 800 \
  --skip-check \
  --skip-captures \
  --dedupe-fen \
  --shuffle-mem-mb 512 \
  --convert-threads 2 \
  "$@"
