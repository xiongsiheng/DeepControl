export CUDA_VISIBLE_DEVICES=0


python -m vllm.entrypoints.openai.api_server \
  --model outputs/verl_checkpoints/nq-deepcontrol-retrieve-grpo-qwen2.5-3b-it/actor \
  --served-model-name deepcontrol-qwen25-3b \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --host 127.0.0.1 \
  --port 8001