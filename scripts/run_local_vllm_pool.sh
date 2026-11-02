#!/usr/bin/env bash
# One vLLM server per GPU plus a least-busy proxy in front of them, all on 127.0.0.1.
# Point inference at the proxy: --base_url http://127.0.0.1:${PROXY_PORT}/v1
#
# Usage: MODEL=sxiong/DeepControl-Qwen2.5-3B SERVED_NAME=deepcontrol-qwen25-3b \
#          NUM_GPUS=8 bash scripts/run_local_vllm_pool.sh
# Requires the inference environment (requirements-infer.txt: vllm 0.10.2,
# transformers 4.56.2). Ctrl-C stops every server and the proxy.
set -Eeuo pipefail
MODEL="${MODEL:-sxiong/DeepControl-Qwen2.5-3B}"
SERVED_NAME="${SERVED_NAME:-deepcontrol-qwen25-3b}"
NUM_GPUS="${NUM_GPUS:-8}"
BASE_PORT="${BASE_PORT:-8101}"
PROXY_PORT="${PROXY_PORT:-8001}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"
pids=()
cleanup() {
  trap - EXIT INT TERM
  kill "${pids[@]}" 2>/dev/null || true
  # vLLM servers exit on SIGTERM; give them 30 s, then SIGKILL anything left in this
  # script's process group (the servers' engine-core children included).
  for _ in $(seq 1 30); do
    alive=0; for p in "${pids[@]}"; do kill -0 "${p}" 2>/dev/null && alive=1; done
    (( alive )) || break; sleep 1
  done
  kill -KILL "${pids[@]}" 2>/dev/null || true
  pkill -KILL -g "$(ps -o pgid= $$ | tr -d ' ')" -f "vllm|VLLM::EngineCore" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

backends=""
# Refuse ports that are already serving something; otherwise a readiness check could be
# answered by a leftover server instead of the one started here.
for port in "${PROXY_PORT}" $(seq "${BASE_PORT}" $((BASE_PORT + NUM_GPUS - 1))); do
  if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:${port}/"; then
    echo "port ${port} is already in use; stop the old server or set BASE_PORT/PROXY_PORT" >&2
    exit 1
  fi
done
# vLLM reserves 90% of each GPU; stop early if another process already holds memory there.
for ((g = 0; g < NUM_GPUS; g++)); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${g}")
  if (( used > 1000 )); then
    echo "GPU ${g} already has ${used} MiB in use by another process; free it or lower NUM_GPUS" >&2
    exit 1
  fi
done
for ((g = 0; g < NUM_GPUS; g++)); do
  port=$((BASE_PORT + g))
  CUDA_VISIBLE_DEVICES="${g}" vllm serve "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --host 127.0.0.1 --port "${port}" \
    --dtype bfloat16 --max-model-len 8192 --enforce-eager --seed 0 \
    > "${LOG_DIR}/vllm_gpu${g}.log" 2>&1 &
  pids+=("$!")
  backends+="${backends:+,}http://127.0.0.1:${port}"
done
for ((g = 0; g < NUM_GPUS; g++)); do
  until curl -fsS "http://127.0.0.1:$((BASE_PORT + g))/health" >/dev/null 2>&1; do
    kill -0 "${pids[$g]}" 2>/dev/null || { echo "vLLM on GPU ${g} exited; see ${LOG_DIR}/vllm_gpu${g}.log" >&2; exit 1; }
    sleep 5
  done
done
LLM_BACKENDS="${backends}" PORT="${PROXY_PORT}" python scripts/llm_least_busy_proxy.py \
  > "${LOG_DIR}/llm_proxy.log" 2>&1 &
proxy=$!; pids+=("${proxy}")
until curl -fsS "http://127.0.0.1:${PROXY_PORT}/proxy_stats" >/dev/null 2>&1; do
  kill -0 "${proxy}" 2>/dev/null || { echo "proxy exited (port ${PROXY_PORT} in use?); see ${LOG_DIR}/llm_proxy.log" >&2; exit 1; }
  sleep 1
done
sleep 2
kill -0 "${proxy}" 2>/dev/null || { echo "proxy exited; see ${LOG_DIR}/llm_proxy.log" >&2; exit 1; }
echo "pool ready: http://127.0.0.1:${PROXY_PORT}/v1 over ${NUM_GPUS} GPUs"
wait
