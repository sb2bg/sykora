#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  utils/tuning/run_large_texel.sh [options]

Options:
  --dataset PATH            Dataset file (default: datasets/texel/train.txt)
  --engine PATH             Tune engine binary (default: zig-out/bin/sykora-tune)
  --run-dir PATH            Output run directory (default: history/tuning/texel_large_<timestamp>)
  --python PATH             Python interpreter to use
  --workers N               Engine eval workers (default: auto)
  --positions N             Max dataset rows to use (default: 1000000)
  --batch-size N            Sample size per epoch (default: 40000)
  --seed N                  Base random seed (default: 1)
  --resume-from PATH        Start from an existing params file
  --skip-build              Skip `zig build -Doptimize=ReleaseFast`
  --no-scalar-bounds        Pass --no-scalar-bounds to the tuner
  --help                    Show this help

Stage controls:
  --coarse-epochs N         Default: 6
  --coarse-delta N          Default: 8
  --coarse-full-eval N      Default: 4

  --mid-epochs N            Default: 10
  --mid-delta N             Default: 4
  --mid-full-eval N         Default: 2

  --fine-epochs N           Default: 14
  --fine-delta N            Default: 2
  --fine-full-eval N        Default: 1

Advanced:
  --min-improvement FLOAT   Default: 0.000001
  --scalar-bound-mult FLOAT Default: 3.0
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

DATASET="datasets/texel/train.txt"
ENGINE="zig-out/bin/sykora-tune"
RUN_DIR=""
SKIP_BUILD=0
NO_SCALAR_BOUNDS=0
RESUME_FROM=""

COARSE_EPOCHS=6
COARSE_DELTA=8
COARSE_FULL_EVAL=4

MID_EPOCHS=10
MID_DELTA=4
MID_FULL_EVAL=2

FINE_EPOCHS=14
FINE_DELTA=2
FINE_FULL_EVAL=1

POSITIONS=1000000
BATCH_SIZE=40000
SEED=1
MIN_IMPROVEMENT="0.000001"
SCALAR_BOUND_MULT="3.0"

PYTHON_PATH="${PYTHON_BIN:-}"
WORKERS="${WORKERS:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --engine) ENGINE="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --python) PYTHON_PATH="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --positions) POSITIONS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --resume-from) RESUME_FROM="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --no-scalar-bounds) NO_SCALAR_BOUNDS=1; shift ;;
    --coarse-epochs) COARSE_EPOCHS="$2"; shift 2 ;;
    --coarse-delta) COARSE_DELTA="$2"; shift 2 ;;
    --coarse-full-eval) COARSE_FULL_EVAL="$2"; shift 2 ;;
    --mid-epochs) MID_EPOCHS="$2"; shift 2 ;;
    --mid-delta) MID_DELTA="$2"; shift 2 ;;
    --mid-full-eval) MID_FULL_EVAL="$2"; shift 2 ;;
    --fine-epochs) FINE_EPOCHS="$2"; shift 2 ;;
    --fine-delta) FINE_DELTA="$2"; shift 2 ;;
    --fine-full-eval) FINE_FULL_EVAL="$2"; shift 2 ;;
    --min-improvement) MIN_IMPROVEMENT="$2"; shift 2 ;;
    --scalar-bound-mult) SCALAR_BOUND_MULT="$2"; shift 2 ;;
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

if [[ -z "${WORKERS}" ]]; then
  cpu_count="$(detect_cpu_count)"
  if (( cpu_count > 2 )); then
    WORKERS="$((cpu_count - 2))"
  else
    WORKERS=1
  fi
fi

if [[ -z "${RUN_DIR}" ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
  RUN_DIR="history/tuning/texel_large_${RUN_ID}"
fi

mkdir -p "${RUN_DIR}"

if [[ ! -f "${DATASET}" ]]; then
  echo "Dataset not found: ${DATASET}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_PATH}" ]]; then
  echo "Python interpreter not executable: ${PYTHON_PATH}" >&2
  exit 1
fi

if [[ ${SKIP_BUILD} -eq 0 ]]; then
  echo "==> Building ReleaseFast binaries"
  zig build -Doptimize=ReleaseFast
fi

if [[ ! -x "${ENGINE}" ]]; then
  echo "Engine not found/executable: ${ENGINE}" >&2
  exit 1
fi

if [[ -n "${RESUME_FROM}" && ! -f "${RESUME_FROM}" ]]; then
  echo "Resume params file not found: ${RESUME_FROM}" >&2
  exit 1
fi

cat > "${RUN_DIR}/run_config.txt" <<EOF
dataset=${DATASET}
engine=${ENGINE}
python=${PYTHON_PATH}
workers=${WORKERS}
positions=${POSITIONS}
batch_size=${BATCH_SIZE}
seed=${SEED}
resume_from=${RESUME_FROM:-<none>}
min_improvement=${MIN_IMPROVEMENT}
scalar_bound_mult=${SCALAR_BOUND_MULT}
coarse_epochs=${COARSE_EPOCHS}
coarse_delta=${COARSE_DELTA}
coarse_full_eval=${COARSE_FULL_EVAL}
mid_epochs=${MID_EPOCHS}
mid_delta=${MID_DELTA}
mid_full_eval=${MID_FULL_EVAL}
fine_epochs=${FINE_EPOCHS}
fine_delta=${FINE_DELTA}
fine_full_eval=${FINE_FULL_EVAL}
EOF

run_stage() {
  local stage_name="$1"
  local epochs="$2"
  local delta="$3"
  local full_eval_interval="$4"
  local seed_value="$5"
  local params_in="$6"
  local params_out="$7"
  local log_path="$8"

  if (( epochs <= 0 )); then
    echo "==> Skipping ${stage_name} (epochs=${epochs})"
    return
  fi

  echo "==> Stage ${stage_name}: epochs=${epochs} delta=${delta} full_eval_interval=${full_eval_interval}"
  local -a cmd=(
    "${PYTHON_PATH}" "utils/tuning/texel_tune.py"
    --dataset "${DATASET}"
    --engine "${ENGINE}"
    --output "${params_out}"
    --positions "${POSITIONS}"
    --epochs "${epochs}"
    --delta "${delta}"
    --batch-size "${BATCH_SIZE}"
    --workers "${WORKERS}"
    --seed "${seed_value}"
    --min-improvement "${MIN_IMPROVEMENT}"
    --full-eval-interval "${full_eval_interval}"
    --scalar-bound-mult "${SCALAR_BOUND_MULT}"
  )
  if [[ "${NO_SCALAR_BOUNDS}" -eq 1 ]]; then
    cmd+=(--no-scalar-bounds)
  fi
  if [[ -n "${params_in}" ]]; then
    cmd+=(--params "${params_in}")
  fi

  "${cmd[@]}" 2>&1 | tee "${log_path}"
}

stage1_params="${RUN_DIR}/params_stage1_coarse.txt"
stage2_params="${RUN_DIR}/params_stage2_mid.txt"
stage3_params="${RUN_DIR}/params_stage3_fine.txt"

current_params="${RESUME_FROM}"
final_params=""

if (( COARSE_EPOCHS > 0 )); then
  run_stage "coarse" "${COARSE_EPOCHS}" "${COARSE_DELTA}" "${COARSE_FULL_EVAL}" \
    "${SEED}" "${current_params}" "${stage1_params}" "${RUN_DIR}/stage1_coarse.log"
  current_params="${stage1_params}"
  final_params="${stage1_params}"
else
  echo "==> Skipping coarse stage (epochs=0)"
fi

if (( MID_EPOCHS > 0 )); then
  run_stage "mid" "${MID_EPOCHS}" "${MID_DELTA}" "${MID_FULL_EVAL}" \
    "$((SEED + 1))" "${current_params}" "${stage2_params}" "${RUN_DIR}/stage2_mid.log"
  current_params="${stage2_params}"
  final_params="${stage2_params}"
else
  echo "==> Skipping mid stage (epochs=0)"
fi

if (( FINE_EPOCHS > 0 )); then
  run_stage "fine" "${FINE_EPOCHS}" "${FINE_DELTA}" "${FINE_FULL_EVAL}" \
    "$((SEED + 2))" "${current_params}" "${stage3_params}" "${RUN_DIR}/stage3_fine.log"
  current_params="${stage3_params}"
  final_params="${stage3_params}"
else
  echo "==> Skipping fine stage (epochs=0)"
fi

if [[ -z "${final_params}" ]]; then
  echo "No stage ran (all epoch counts are 0)." >&2
  exit 1
fi

cp -f "${final_params}" "${RUN_DIR}/tune_params_final.txt"

echo
echo "Large tune completed."
echo "Run directory: ${RUN_DIR}"
echo "Final params:  ${RUN_DIR}/tune_params_final.txt"
echo "Apply with:"
echo "  python utils/tuning/apply_params.py ${RUN_DIR}/tune_params_final.txt"
