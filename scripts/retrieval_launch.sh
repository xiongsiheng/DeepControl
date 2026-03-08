file_path=data/wiki18
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki_dump.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

export CUDA_VISIBLE_DEVICES=0

python deepcontrol/search/retrieval_server.py --index_path $index_file \
                                              --corpus_path $corpus_file \
                                              --topk 10 \
                                              --retriever_name $retriever_name \
                                              --retriever_model $retriever_path \
                                              --faiss_gpu