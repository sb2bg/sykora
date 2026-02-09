#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  utils/match/selfplay_vs_ref.sh [--baseline <spec>] [--archive] [selfplay.py args...]

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

  With --archive:
    - snapshots baseline/candidate binaries into history/engines/
    - runs an archived match via utils/history/history.py match
    - updates ratings/network inputs under history/

Examples:
  utils/match/selfplay_vs_ref.sh --games 120 --movetime-ms 200
  utils/match/selfplay_vs_ref.sh --archive --games 200 --movetime-ms 60
  utils/match/selfplay_vs_ref.sh --baseline history/engines/<snapshot_id>/engine --games 200 --depth 8
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sykora-selfplay.XXXXXX")"
BASE_BIN="$TMP_DIR/sykora_baseline"
CAND_BIN="$TMP_DIR/sykora_working"
BASELINE_FILE="$ROOT/history/current_baseline.txt"
BASELINE_SPEC=""
ARCHIVE_MODE=0
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
    --archive)
      ARCHIVE_MODE=1
      shift
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

detect_baseline_snapshot_id() {
  local baseline_path="$1"
  local prefix="$ROOT/history/engines/"
  if [[ "$baseline_path" == "$prefix"*/engine ]]; then
    local rel="${baseline_path#"$prefix"}"
    printf '%s\n' "${rel%%/*}"
    return 0
  fi
  return 1
}

validate_archive_args() {
  local disallowed=("--name1" "--name2" "--output-dir" "--summary-json")
  for arg in "${SELFPLAY_ARGS[@]}"; do
    for flag in "${disallowed[@]}"; do
      if [[ "$arg" == "$flag" || "$arg" == "$flag="* ]]; then
        echo "error: $flag is not supported with --archive (history.py sets this automatically)" >&2
        exit 2
      fi
    done
  done
}

snapshot_engine() {
  local engine_path="$1"
  local label="$2"
  local notes="$3"
  local out

  out="$("$PYTHON_BIN" "$ROOT/utils/history/history.py" snapshot --engine "$engine_path" --label "$label" --notes "$notes")"
  printf '%s\n' "$out" >&2

  local id
  id="$(printf '%s\n' "$out" | sed -n 's/^Snapshot created: //p' | head -n1)"
  if [[ -z "$id" ]]; then
    echo "error: could not parse snapshot id from history.py output" >&2
    exit 1
  fi
  printf '%s\n' "$id"
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
PYTHON_BIN="${PYTHON_BIN:-$HOME/.pyenv/shims/python}"

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

if [[ $ARCHIVE_MODE -eq 1 ]]; then
  validate_archive_args
  echo "==> Archiving snapshots + match under history/"
  "$PYTHON_BIN" "$ROOT/utils/history/history.py" init >/dev/null

  baseline_id="$(detect_baseline_snapshot_id "$BASELINE_PATH" || true)"
  if [[ -z "$baseline_id" ]]; then
    baseline_id="$(snapshot_engine "$BASELINE_PATH" "baseline-ref" "selfplay_vs_ref baseline: $BASELINE_SPEC")"
  else
    echo "Using existing baseline snapshot: $baseline_id"
  fi

  candidate_id="$(snapshot_engine "$CAND_BIN" "working" "selfplay_vs_ref candidate from current working tree")"

  "$PYTHON_BIN" "$ROOT/utils/history/history.py" match "$baseline_id" "$candidate_id" --python "$PYTHON_BIN" "${SELFPLAY_ARGS[@]}"
else
  cp "$BASELINE_PATH" "$BASE_BIN"

  if [[ $has_name1 -eq 0 ]]; then
    SELFPLAY_ARGS+=(--name1 "baseline:$BASELINE_SPEC")
  fi
  if [[ $has_name2 -eq 0 ]]; then
    SELFPLAY_ARGS+=(--name2 "working")
  fi

  echo "==> Running self-play"
  "$PYTHON_BIN" "$ROOT/utils/match/selfplay.py" "$BASE_BIN" "$CAND_BIN" "${SELFPLAY_ARGS[@]}"
fi
