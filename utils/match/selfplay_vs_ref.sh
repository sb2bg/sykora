#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  utils/match/selfplay_vs_ref.sh [--baseline <spec>] [selfplay.py args...]

Description:
  Builds the current working tree (candidate) in ReleaseFast and runs
  utils/match/selfplay.py against a baseline resolved from history/current_baseline.txt
  (or a user-provided baseline spec).

    baseline = resolved from baseline spec
    candidate = current working tree build

  Baseline resolution order:
    1) --baseline <spec>
    2) first non-empty, non-comment line in ./history/current_baseline.txt

  A baseline spec may be:
    - path to an engine binary
    - snapshot ID under history/engines/<id>/engine

Examples:
  utils/match/selfplay_vs_ref.sh --games 120 --movetime-ms 200
  utils/match/selfplay_vs_ref.sh --baseline history/engines/<snapshot_id>/engine --games 200 --depth 8
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sykora-selfplay.XXXXXX")"
BASE_BIN="$TMP_DIR/sykora_baseline"
CAND_BIN="$TMP_DIR/sykora_working"
BASELINE_FILE="$ROOT/history/current_baseline.txt"
BASELINE_SPEC=""
SELFPLAY_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline)
      if [[ $# -lt 2 ]]; then
        echo "error: --baseline requires a value" >&2
        exit 2
      fi
      BASELINE_SPEC="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      SELFPLAY_ARGS+=("$1")
      shift
      ;;
  esac
done

read_baseline_from_file() {
  local file="$1"
  if [[ ! -f "$file" ]]; then
    return 1
  fi
  sed -e 's/#.*$//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' "$file" | awk 'NF{print; exit}'
}

resolve_baseline_binary() {
  local spec="$1"
  local candidate=""

  if [[ -z "$spec" ]]; then
    echo "error: baseline spec is empty" >&2
    return 1
  fi

  if [[ -f "$spec" ]]; then
    candidate="$spec"
  elif [[ -f "$ROOT/$spec" ]]; then
    candidate="$ROOT/$spec"
  elif [[ -f "$ROOT/history/engines/$spec/engine" ]]; then
    candidate="$ROOT/history/engines/$spec/engine"
  else
    echo "error: could not resolve baseline spec '$spec'" >&2
    echo "  expected a binary path or history snapshot ID" >&2
    return 1
  fi

  printf '%s\n' "$candidate"
}

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

if [[ -z "$BASELINE_SPEC" ]]; then
  BASELINE_SPEC="$(read_baseline_from_file "$BASELINE_FILE" || true)"
fi
if [[ -z "$BASELINE_SPEC" ]]; then
  echo "error: no baseline configured" >&2
  echo "  set --baseline, or set history/current_baseline.txt" >&2
  exit 2
fi

BASELINE_PATH="$(resolve_baseline_binary "$BASELINE_SPEC")"
cp "$BASELINE_PATH" "$BASE_BIN"

has_name1=0
has_name2=0
for arg in "${SELFPLAY_ARGS[@]}"; do
  if [[ "$arg" == "--name1" || "$arg" == --name1=* ]]; then
    has_name1=1
  fi
  if [[ "$arg" == "--name2" || "$arg" == --name2=* ]]; then
    has_name2=1
  fi
done

echo "==> Building candidate (current working tree)"
(cd "$ROOT" && zig build -Doptimize=ReleaseFast >/dev/null)
cp "$ROOT/zig-out/bin/sykora" "$CAND_BIN"

if [[ $has_name1 -eq 0 ]]; then
  SELFPLAY_ARGS+=(--name1 "baseline:$BASELINE_SPEC")
fi
if [[ $has_name2 -eq 0 ]]; then
  SELFPLAY_ARGS+=(--name2 "working")
fi

PYTHON_BIN="${PYTHON_BIN:-$HOME/.pyenv/shims/python}"

echo "==> Running self-play"
"$PYTHON_BIN" "$ROOT/utils/match/selfplay.py" "$BASE_BIN" "$CAND_BIN" "${SELFPLAY_ARGS[@]}"
