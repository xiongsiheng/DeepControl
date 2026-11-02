#!/usr/bin/env bash
# Run one dataset split as NUM_PROCS concurrent infer_fast.py processes over contiguous
# row ranges, all against the vLLM pool (scripts/run_local_vllm_pool.sh), then merge.
# With NUM_PROCS=8 and NUM_WORKERS=16, about 128 questions are in flight at once.
#
# Usage: CONFIG=hotpotqa SPLIT=dev bash scripts/infer_fast_parallel.sh
set -Eeuo pipefail
CONFIG="${CONFIG:-hotpotqa}"
SPLIT="${SPLIT:-dev}"
DATASET_REPO_PATH="${DATASET_REPO_PATH:-FlashRAG_datasets}"
MODEL="${MODEL:-deepcontrol-qwen25-3b}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8001/v1}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000}"
NUM_PROCS="${NUM_PROCS:-8}"
NUM_WORKERS="${NUM_WORKERS:-16}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
RESUME="${RESUME:-0}"   # 1: skip questions already in the part files (rerun after an interruption)
OUT="${OUT:-results/infer_${MODEL}_${CONFIG}_${SPLIT}.jsonl}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$(dirname "${OUT}")" "${LOG_DIR}"

total=$(python - "${DATASET_REPO_PATH}" "${CONFIG}" "${SPLIT}" <<'PY'
import sys
from datasets import load_dataset
print(len(load_dataset(sys.argv[1], sys.argv[2])[sys.argv[3]]))
PY
)
[[ -n "${MAX_SAMPLES}" ]] && (( MAX_SAMPLES < total )) && total="${MAX_SAMPLES}"
per=$(( (total + NUM_PROCS - 1) / NUM_PROCS ))
pids=()
for ((p = 0; p < NUM_PROCS; p++)); do
  start=$((p * per)); (( start >= total )) && break
  count=$(( start + per > total ? total - start : per ))
  python infer_fast.py \
    --mode dataset --source local --base_url "${BASE_URL}" --model "${MODEL}" \
    --dataset_repo_path "${DATASET_REPO_PATH}" --config_name "${CONFIG}" --split "${SPLIT}" \
    --start_index "${start}" --max_samples "${count}" \
    --topk 5 --retriever_url "${RETRIEVER_URL}" --max_turns 8 --max_tokens 512 \
    --temperature 0 --num_workers "${NUM_WORKERS}" --retriever_max_concurrency 16 \
    $([[ "${RESUME}" == 1 ]] && echo --resume) \
    --out_jsonl "${OUT%.jsonl}.part${p}.jsonl" > "${LOG_DIR}/infer_$(basename "${OUT%.jsonl}")_part${p}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
cat "${OUT%.jsonl}".part*.jsonl > "${OUT}"
echo "wrote ${OUT} ($(wc -l < "${OUT}") rows), status=${status}"
exit "${status}"
