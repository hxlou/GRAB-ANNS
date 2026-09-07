#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CMAKE_BIN="${CMAKE_BIN:-/home/lhx/miniconda3/envs/faiss/bin/cmake}"
BUILD_DIR="${BUILD_DIR:-build_gpu0}"
GPU_DEVICE="${GPU_DEVICE:-0}"
OUTPUT_CSV="${1:-build/gpu_bruteforce_baseline_b1000.csv}"
LOG_PATH="${2:-logs/gpu_bruteforce_baseline_b1000.log}"

mkdir -p "$(dirname "${OUTPUT_CSV}")" "$(dirname "${LOG_PATH}")"
rm -f "${OUTPUT_CSV}"

"${CMAKE_BIN}" -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
"${CMAKE_BIN}" --build "${BUILD_DIR}" -j --target gpu_bruteforce_baseline

run_one() {
  local dataset="$1"
  local path="$2"

  echo "===== ${dataset}: ${path} ====="
  "./${BUILD_DIR}/gpu_bruteforce_baseline" \
    "${path}" "${dataset}" "${OUTPUT_CSV}" \
    --n 1000000 \
    --buckets 100 \
    --batch 1000 \
    --rounds 5 \
    --k 10 \
    --device "${GPU_DEVICE}" \
    --seed 20260717 \
    --ratios 0.01,0.10,0.20,1.0
}

{
  echo "GPU brute-force baseline"
  echo "Start: $(date '+%F %T')"
  echo "Device: ${GPU_DEVICE}"
  echo "Output: ${OUTPUT_CSV}"
  run_one "DEEP-96" "data/deep_base.fvecs"
  run_one "SIFT-128" "data/sift.fvecs"
  run_one "GIST-960" "data/GIST1M/gist_base.fvecs"
  run_one "WIT-2048" "data/wit-image.fvecs"
  echo "End: $(date '+%F %T')"
} 2>&1 | tee "${LOG_PATH}"