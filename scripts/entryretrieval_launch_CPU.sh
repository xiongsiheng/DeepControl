python deepcontrol/search/entryretrieval_server_multi.py \
  --retriever_model intfloat/e5-base-v2 \
  --retriever_name e5 \
  --max_length 512 \
  --batch_size 64 \
  --topk 10 \
  --port 8000
