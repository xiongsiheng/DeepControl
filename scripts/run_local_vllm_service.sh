#!/usr/bin/env bash
# One vLLM server for DeepControl inference. MODEL is a local checkpoint or a Hugging Face
# model id; SERVED_NAME is the name clients pass as --model.
#
# Usage: MODEL=sxiong/DeepControl-Qwen2.5-7B SERVED_NAME=deepcontrol-qwen25-7b \
#          bash scripts/run_local_vllm_service.sh
set -euo pipefail
MODEL="${MODEL:-sxiong/DeepControl-Qwen2.5-3B}"
SERVED_NAME="${SERVED_NAME:-deepcontrol-qwen25-3b}"
PORT="${PORT:-8001}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --served-model-name "${SERVED_NAME}" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --host 127.0.0.1 \
  --port "${PORT}"
