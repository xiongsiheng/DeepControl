export CUDA_VISIBLE_DEVICES=0

python deepcontrol/search/entryretrieval_server_multi.py \
  --retriever_model intfloat/e5-base-v2 \
  --retriever_name e5 \
  --max_length 512 \
  --batch_size 512 \
  --topk 10 \
  --faiss_gpu --gpu_use_fp16 \
  --gpu_add_batch 200000 \
  --port 8000
