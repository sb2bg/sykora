#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  utils/selfplay_vs_ref.sh <git-ref> [selfplay.py args...]

Description:
  Builds engine at <git-ref> in a temporary worktree, builds current working tree,
  then runs utils/selfplay.py as:

    baseline = ref build
    candidate = current working tree build

Examples:
  utils/selfplay_vs_ref.sh HEAD~1 --games 120 --movetime-ms 200
  utils/selfplay_vs_ref.sh v0.2.0 --games 200 --depth 8 --openings default
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

REF="$1"
shift

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sykora-selfplay.XXXXXX")"
WT_DIR="$TMP_DIR/ref-worktree"
BASE_BIN="$TMP_DIR/sykora_ref"
CAND_BIN="$TMP_DIR/sykora_working"
SELFPLAY_ARGS=("$@")

cleanup() {
  git -C "$ROOT" worktree remove --force "$WT_DIR" >/dev/null 2>&1 || true
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

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

echo "==> Preparing baseline worktree at ref $REF"
git -C "$ROOT" worktree add --detach "$WT_DIR" "$REF" >/dev/null

echo "==> Building baseline (ref: $REF)"
(cd "$WT_DIR" && zig build -Doptimize=ReleaseFast >/dev/null)
cp "$WT_DIR/zig-out/bin/sykora" "$BASE_BIN"

echo "==> Building candidate (current working tree)"
(cd "$ROOT" && zig build -Doptimize=ReleaseFast >/dev/null)
cp "$ROOT/zig-out/bin/sykora" "$CAND_BIN"

if [[ $has_name1 -eq 0 ]]; then
  SELFPLAY_ARGS+=(--name1 "ref:$REF")
fi
if [[ $has_name2 -eq 0 ]]; then
  SELFPLAY_ARGS+=(--name2 "working")
fi

PYTHON_BIN="${PYTHON_BIN:-$HOME/.pyenv/shims/python}"

echo "==> Running self-play"
"$PYTHON_BIN" "$ROOT/utils/selfplay.py" "$BASE_BIN" "$CAND_BIN" "${SELFPLAY_ARGS[@]}"
