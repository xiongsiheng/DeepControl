export CUDA_VISIBLE_DEVICES=0

vllm serve PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo \
  --dtype bfloat16 \
  --port 8001