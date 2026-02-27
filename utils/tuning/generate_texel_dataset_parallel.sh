#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  utils/tuning/generate_texel_dataset_parallel.sh [options]

Goal:
  Run multiple dataset-generation shards in parallel, then merge into one output file.

Options:
  --engine PATH              Engine binary (default: zig-out/bin/sykora)
  --output PATH              Final merged dataset (default: datasets/texel/train_large.txt)
  --games N                  Total games across all shards (default: 30000)
  --shards N                 Number of parallel shard processes (default: cpu_count - 2)
  --work-dir PATH            Shard/log folder (default: history/tuning/texel_dataset_shards_<ts>)
  --python PATH              Python interpreter
  --seed N                   Base seed; shard i uses seed+ i (default: 1)
  --keep-shards              Keep shard text files after merge
  --dedupe-fen               Merge with FEN deduplication (keeps first occurrence)

Quality / filtering options:
  --stockfish PATH           Stockfish path (default: auto-detect if available)
  --no-stockfish             Disable Stockfish annotation (cp=0)
  --sf-depth N               Stockfish analysis depth (default: 10)
  --sf-threads N             Stockfish Threads per shard (default: 1)
  --sf-hash-mb N             Stockfish Hash MB per shard (default: 32)
  --sf-restart-retries N     Restart attempts after SF failure (default: 1)
  --depth N                  Self-play search depth (default: 6)
  --movetime-ms N            Self-play movetime ms (used only when depth is disabled)
  --min-move N               Default: 8
  --max-move N               Default: 80
  --min-pieces N             Default: 10
  --positions-per-game N     Default: 20
  --help                     Show this help
EOF
}

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

detect_cpu_count() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return
  fi
  if command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.ncpu
    return
  fi
  echo 4
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ENGINE="zig-out/bin/sykora"
OUTPUT="datasets/texel/train_large.txt"
GAMES=30000
SHARDS=""
WORK_DIR=""
PYTHON_PATH="${PYTHON_BIN:-}"
SEED=1
KEEP_SHARDS=0
DEDUPE_FEN=0

STOCKFISH=""
if [[ -x "/opt/homebrew/bin/stockfish" ]]; then
  STOCKFISH="/opt/homebrew/bin/stockfish"
fi
SF_DEPTH=10
SF_THREADS=1
SF_HASH_MB=32
SF_RESTART_RETRIES=1
DEPTH=6
MOVETIME_MS=50
USE_DEPTH=1
MIN_MOVE=8
MAX_MOVE=80
MIN_PIECES=10
POSITIONS_PER_GAME=20

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine) ENGINE="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --games) GAMES="$2"; shift 2 ;;
    --shards) SHARDS="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --python) PYTHON_PATH="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --keep-shards) KEEP_SHARDS=1; shift ;;
    --dedupe-fen) DEDUPE_FEN=1; shift ;;
    --stockfish) STOCKFISH="$2"; shift 2 ;;
    --no-stockfish) STOCKFISH=""; shift ;;
    --sf-depth) SF_DEPTH="$2"; shift 2 ;;
    --sf-threads) SF_THREADS="$2"; shift 2 ;;
    --sf-hash-mb) SF_HASH_MB="$2"; shift 2 ;;
    --sf-restart-retries) SF_RESTART_RETRIES="$2"; shift 2 ;;
    --depth) DEPTH="$2"; USE_DEPTH=1; shift 2 ;;
    --movetime-ms) MOVETIME_MS="$2"; USE_DEPTH=0; shift 2 ;;
    --min-move) MIN_MOVE="$2"; shift 2 ;;
    --max-move) MAX_MOVE="$2"; shift 2 ;;
    --min-pieces) MIN_PIECES="$2"; shift 2 ;;
    --positions-per-game) POSITIONS_PER_GAME="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${PYTHON_PATH}" ]]; then
  PYTHON_PATH="$(resolve_python_bin)"
fi
if [[ ! -x "${PYTHON_PATH}" ]]; then
  echo "Python interpreter not executable: ${PYTHON_PATH}" >&2
  exit 1
fi

if [[ ! -x "${ENGINE}" ]]; then
  echo "Engine not found/executable: ${ENGINE}" >&2
  exit 1
fi

if [[ -n "${STOCKFISH}" && ! -x "${STOCKFISH}" ]]; then
  echo "Stockfish not executable: ${STOCKFISH}" >&2
  exit 1
fi

if [[ -z "${SHARDS}" ]]; then
  cpu_count="$(detect_cpu_count)"
  if (( cpu_count > 2 )); then
    SHARDS="$((cpu_count - 2))"
  else
    SHARDS=1
  fi
fi

if (( SHARDS <= 0 )); then
  echo "--shards must be > 0" >&2
  exit 1
fi
if (( GAMES <= 0 )); then
  echo "--games must be > 0" >&2
  exit 1
fi

if [[ -z "${WORK_DIR}" ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
  WORK_DIR="history/tuning/texel_dataset_shards_${RUN_ID}"
fi
mkdir -p "${WORK_DIR}"
mkdir -p "$(dirname "${OUTPUT}")"

cat > "${WORK_DIR}/run_config.txt" <<EOF
engine=${ENGINE}
output=${OUTPUT}
games=${GAMES}
shards=${SHARDS}
python=${PYTHON_PATH}
stockfish=${STOCKFISH:-<none>}
sf_depth=${SF_DEPTH}
sf_threads=${SF_THREADS}
sf_hash_mb=${SF_HASH_MB}
sf_restart_retries=${SF_RESTART_RETRIES}
depth=$(if (( USE_DEPTH == 1 )); then echo "${DEPTH}"; else echo "<disabled>"; fi)
movetime_ms=$(if (( USE_DEPTH == 1 )); then echo "<ignored>"; else echo "${MOVETIME_MS}"; fi)
seed=${SEED}
min_move=${MIN_MOVE}
max_move=${MAX_MOVE}
min_pieces=${MIN_PIECES}
positions_per_game=${POSITIONS_PER_GAME}
dedupe_fen=${DEDUPE_FEN}
keep_shards=${KEEP_SHARDS}
EOF

echo "==> Parallel dataset generation"
echo "Engine:         ${ENGINE}"
echo "Stockfish:      ${STOCKFISH:-disabled}"
echo "Games total:    ${GAMES}"
echo "Shards:         ${SHARDS}"
echo "Work dir:       ${WORK_DIR}"
echo "Final output:   ${OUTPUT}"
echo "Target max rows: ~$((GAMES * POSITIONS_PER_GAME)) (before filtering)"
echo

base_games=$((GAMES / SHARDS))
extra_games=$((GAMES % SHARDS))

declare -a PIDS=()
declare -a SHARD_IDS=()
declare -a SHARD_FILES=()
declare -a SHARD_LOGS=()

start_ts="$(date +%s)"

for ((i = 0; i < SHARDS; i++)); do
  shard_games="${base_games}"
  if (( i < extra_games )); then
    shard_games=$((shard_games + 1))
  fi
  if (( shard_games <= 0 )); then
    continue
  fi

  shard_file="${WORK_DIR}/shard_${i}.txt"
  shard_log="${WORK_DIR}/shard_${i}.log"
  shard_seed=$((SEED + i))

  cmd=(
    "${PYTHON_PATH}" "utils/tuning/generate_texel_dataset.py"
    --engine "${ENGINE}"
    --output "${shard_file}"
    --games "${shard_games}"
    --sf-depth "${SF_DEPTH}"
    --sf-threads "${SF_THREADS}"
    --sf-hash-mb "${SF_HASH_MB}"
    --sf-restart-retries "${SF_RESTART_RETRIES}"
    --min-move "${MIN_MOVE}"
    --max-move "${MAX_MOVE}"
    --min-pieces "${MIN_PIECES}"
    --positions-per-game "${POSITIONS_PER_GAME}"
    --seed "${shard_seed}"
  )

  if (( USE_DEPTH == 1 )); then
    cmd+=(--depth "${DEPTH}")
  else
    cmd+=(--movetime-ms "${MOVETIME_MS}")
  fi

  if [[ -n "${STOCKFISH}" ]]; then
    cmd+=(--stockfish "${STOCKFISH}")
  fi

  echo "  Launch shard ${i}: games=${shard_games} seed=${shard_seed}"
  ("${cmd[@]}" >"${shard_log}" 2>&1) &
  pid=$!

  PIDS+=("${pid}")
  SHARD_IDS+=("${i}")
  SHARD_FILES+=("${shard_file}")
  SHARD_LOGS+=("${shard_log}")
done

echo
echo "==> Waiting for shards..."
failed=0
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  sid="${SHARD_IDS[$idx]}"
  sfile="${SHARD_FILES[$idx]}"
  slog="${SHARD_LOGS[$idx]}"
  if wait "${pid}"; then
    line_count="$(wc -l < "${sfile}")"
    echo "  shard ${sid}: ok (${line_count} rows)"
  else
    echo "  shard ${sid}: FAILED (log: ${slog})" >&2
    failed=1
  fi
done

if (( failed != 0 )); then
  echo "One or more shards failed. Keeping shard outputs/logs in ${WORK_DIR}." >&2
  exit 1
fi

echo
echo "==> Merging shards"
if (( DEDUPE_FEN == 1 )); then
  "${PYTHON_PATH}" - "${OUTPUT}" "${SHARD_FILES[@]}" <<'PY'
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
in_paths = [Path(p) for p in sys.argv[2:]]
seen_fens = set()
kept = 0
dropped = 0
with out_path.open("w") as out:
    for path in in_paths:
        with path.open() as handle:
            for line in handle:
                fen = line.split("|", 1)[0].strip()
                if fen in seen_fens:
                    dropped += 1
                    continue
                seen_fens.add(fen)
                out.write(line)
                kept += 1
print(f"dedupe_fen kept={kept} dropped={dropped}")
PY
else
  cat "${SHARD_FILES[@]}" > "${OUTPUT}"
fi

final_rows="$(wc -l < "${OUTPUT}")"
elapsed="$(( $(date +%s) - start_ts ))"

echo "Final rows:     ${final_rows}"
echo "Elapsed:        ${elapsed}s"
echo "Output:         ${OUTPUT}"

if (( KEEP_SHARDS == 0 )); then
  rm -f "${SHARD_FILES[@]}"
  echo "Shard text files removed (logs kept): ${WORK_DIR}"
else
  echo "Shard files kept: ${WORK_DIR}"
fi
