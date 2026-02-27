#!/usr/bin/env bash
#
# Parallel self-play data generation for Sykora NNUE training.
# Launches N sykora gensfen instances with different seeds, then concatenates output.
#
# Usage:
#   ./utils/nnue/bullet/generate_data_parallel.sh \
#       --shards 12 --games 100000 --depth 8 --output data/train.data --seed 1
#
# For i9-13900 (24C/32T): 12 shards recommended.

set -euo pipefail

# Defaults
SHARDS=12
TOTAL_GAMES=100000
DEPTH=8
OUTPUT="data/train.data"
BASE_SEED=1
RANDOM_PLIES=10
SAMPLE_PCT=25
MIN_PLY=16
MAX_PLY=400
SYKORA_BIN="./zig-out/bin/sykora"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --shards)      SHARDS="$2";       shift 2 ;;
        --games)       TOTAL_GAMES="$2";   shift 2 ;;
        --depth)       DEPTH="$2";         shift 2 ;;
        --output)      OUTPUT="$2";        shift 2 ;;
        --seed)        BASE_SEED="$2";     shift 2 ;;
        --random-plies) RANDOM_PLIES="$2"; shift 2 ;;
        --sample-pct)  SAMPLE_PCT="$2";    shift 2 ;;
        --min-ply)     MIN_PLY="$2";       shift 2 ;;
        --max-ply)     MAX_PLY="$2";       shift 2 ;;
        --bin)         SYKORA_BIN="$2";    shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 --shards N --games N --depth D --output FILE [--seed N] [--random-plies N] [--sample-pct N]" >&2
            exit 1
            ;;
    esac
done

PER_SHARD=$(( (TOTAL_GAMES + SHARDS - 1) / SHARDS ))
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

echo "Generating data: ${TOTAL_GAMES} games across ${SHARDS} shards (${PER_SHARD} games/shard)"
echo "Depth: ${DEPTH}, Random plies: ${RANDOM_PLIES}, Sample: ${SAMPLE_PCT}%"
echo "Output: ${OUTPUT}"
echo "Temp dir: ${TMPDIR}"
echo ""

PIDS=()
for i in $(seq 0 $((SHARDS - 1))); do
    SEED=$((BASE_SEED + i))
    SHARD_FILE="${TMPDIR}/shard_${i}.data"

    echo "  Starting shard ${i} (seed=${SEED}, games=${PER_SHARD}) ..."
    "${SYKORA_BIN}" gensfen \
        --output "${SHARD_FILE}" \
        --games "${PER_SHARD}" \
        --depth "${DEPTH}" \
        --random-plies "${RANDOM_PLIES}" \
        --seed "${SEED}" \
        --sample-pct "${SAMPLE_PCT}" \
        --min-ply "${MIN_PLY}" \
        --max-ply "${MAX_PLY}" \
        --report-interval 500 &
    PIDS+=($!)
done

echo ""
echo "Waiting for ${SHARDS} shards to complete ..."

FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "  Shard PID ${pid} failed!" >&2
        FAILED=$((FAILED + 1))
    fi
done

if [[ $FAILED -gt 0 ]]; then
    echo "ERROR: ${FAILED} shard(s) failed." >&2
    exit 1
fi

echo "All shards complete. Concatenating ..."

# Ensure output directory exists
mkdir -p "$(dirname "${OUTPUT}")"

cat "${TMPDIR}"/shard_*.data > "${OUTPUT}"

TOTAL_BYTES=$(stat -f%z "${OUTPUT}" 2>/dev/null || stat -c%s "${OUTPUT}" 2>/dev/null || echo "?")
TOTAL_RECORDS=$((TOTAL_BYTES / 32))

echo ""
echo "Done: ${OUTPUT} (${TOTAL_BYTES} bytes, ${TOTAL_RECORDS} positions)"
