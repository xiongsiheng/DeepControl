#!/usr/bin/env bash
# One vLLM server for the baseline (search-only) model; clients pass MODEL as --model.
#
# Usage: MODEL=<Hugging Face id or path of the baseline model> \
#          bash scripts/run_local_vllm_service_baseline.sh
set -euo pipefail
MODEL="${MODEL:-PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo}"
PORT="${PORT:-8001}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

vllm serve "${MODEL}" \
  --dtype bfloat16 \
  --host 127.0.0.1 \
  --port "${PORT}"
