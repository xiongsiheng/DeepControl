#!/usr/bin/env bash
# Baseline (search-only) inference on one dataset split against an OpenAI-compatible server
# (e.g. scripts/run_local_vllm_service_baseline.sh). MODEL is the served model name.
#
# Usage: MODEL=<served baseline model> CONFIG=nq SPLIT=test \
#          bash scripts/infer_vanilla_fast.sh
set -euo pipefail
MODEL="${MODEL:-PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo}"
CONFIG="${CONFIG:-hotpotqa}"
SPLIT="${SPLIT:-dev}"
DATASET_REPO_PATH="${DATASET_REPO_PATH:-RUC-NLPIR/FlashRAG_datasets}"
MAX_SAMPLES="${MAX_SAMPLES:-}"   # e.g. 20 for a quick try; empty: the whole split
BASE_URL="${BASE_URL:-http://127.0.0.1:8001/v1}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
OUT="${OUT:-results/infer_vanilla_${MODEL##*/}_${CONFIG}_topk3_max_turns_6_temperature_0.jsonl}"

python infer_vanilla_fast.py \
  --mode dataset \
  --source local \
  --base_url "${BASE_URL}" \
  --model "${MODEL}" \
  --dataset_repo_path "${DATASET_REPO_PATH}" \
  --config_name "${CONFIG}" \
  --split "${SPLIT}" \
  --topk 3 \
  --retriever_url "${RETRIEVER_URL}" \
  --max_turns 6 \
  --max_tokens 2048 \
  --temperature 0 \
  --num_workers "${NUM_WORKERS}" \
  ${MAX_SAMPLES:+--max_samples "${MAX_SAMPLES}"} \
  --out_jsonl "${OUT}"
