#!/usr/bin/env bash
# DeepControl inference on one dataset split against an OpenAI-compatible server
# (e.g. scripts/run_local_vllm_service.sh). MODEL is the served model name.
#
# Usage: MODEL=deepcontrol-qwen25-7b CONFIG=hotpotqa SPLIT=dev bash scripts/infer_fast.sh
set -euo pipefail
MODEL="${MODEL:-deepcontrol-qwen25-3b}"
CONFIG="${CONFIG:-hotpotqa}"
SPLIT="${SPLIT:-dev}"
DATASET_REPO_PATH="${DATASET_REPO_PATH:-RUC-NLPIR/FlashRAG_datasets}"
MAX_SAMPLES="${MAX_SAMPLES:-}"   # e.g. 20 for a quick try; empty: the whole split
BASE_URL="${BASE_URL:-http://127.0.0.1:8001/v1}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000}"
NUM_WORKERS="${NUM_WORKERS:-16}"
OUT="${OUT:-results/infer_${MODEL##*/}_${CONFIG}_topk5_max_turns_8_temperature_0.jsonl}"

python infer_fast.py \
  --mode dataset \
  --source local \
  --base_url "${BASE_URL}" \
  --model "${MODEL}" \
  --dataset_repo_path "${DATASET_REPO_PATH}" \
  --config_name "${CONFIG}" \
  --split "${SPLIT}" \
  --topk 5 \
  --retriever_url "${RETRIEVER_URL}" \
  --max_turns 8 \
  --max_tokens 512 \
  --temperature 0 \
  --num_workers "${NUM_WORKERS}" \
  ${MAX_SAMPLES:+--max_samples "${MAX_SAMPLES}"} \
  --out_jsonl "${OUT}"
