#!/usr/bin/env bash
# CPU retriever for DeepControl (training and inference): NUM_BACKENDS server processes on
# ports BACKEND_PORT.. behind a round-robin proxy on PORT. Each backend loads the index and the
# corpus (about 80 GB of RAM); requests are batched (up to 128 queries, 20 ms wait).
#
# Usage: bash scripts/entryretrieval_launch_CPU.sh          # 3 backends, proxy on :8000
#        NUM_BACKENDS=1 bash scripts/entryretrieval_launch_CPU.sh
set -Eeuo pipefail
NUM_BACKENDS="${NUM_BACKENDS:-3}"
PORT="${PORT:-8000}"
BACKEND_PORT="${BACKEND_PORT:-8201}"   # clear of vLLM (8001) and the vLLM pool (8101+)
THREADS="${RETRIEVER_THREADS:-32}"   # CPU threads per backend
export TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=8
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" OMP_DYNAMIC=false MKL_DYNAMIC=false

pids=()
cleanup() {
  trap - EXIT INT TERM
  kill "${pids[@]}" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

backends=""
for ((i = 0; i < NUM_BACKENDS; i++)); do
  port=$((BACKEND_PORT + i))
  python -u deepcontrol/search/entryretrieval_server_multi.py \
    --retriever_model intfloat/e5-base-v2 \
    --retriever_name e5 \
    --max_length 512 \
    --batch_size 64 \
    --request_batch_size 128 \
    --request_batch_wait_ms 20 \
    --corpus_num_proc 16 \
    --topk 10 \
    --port "${port}" &
  pids+=("$!")
  backends+="${backends:+,}http://127.0.0.1:${port}"
done
for ((i = 0; i < NUM_BACKENDS; i++)); do
  until curl -fsS "http://127.0.0.1:$((BACKEND_PORT + i))/docs" >/dev/null 2>&1; do
    kill -0 "${pids[$i]}" 2>/dev/null || { echo "retriever backend on port $((BACKEND_PORT + i)) exited" >&2; exit 1; }
    sleep 10
  done
done
RETRIEVER_BACKENDS="${backends}" PORT="${PORT}" python -u scripts/retriever_round_robin_proxy.py &
pids+=("$!")
until curl -fsS "http://127.0.0.1:${PORT}/docs" >/dev/null 2>&1; do
  kill -0 "${pids[-1]}" 2>/dev/null || { echo "retriever proxy exited (port ${PORT} in use?)" >&2; exit 1; }
  sleep 2
done
echo "retriever ready: http://127.0.0.1:${PORT} over ${NUM_BACKENDS} backends"
wait
