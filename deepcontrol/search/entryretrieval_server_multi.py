import argparse
import queue
import threading
import time
import re
import os
from typing import List, Optional

import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
import datasets

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from gpu_flat import build_gpu_flat


manifest = [
    {
        "name": "wiki18",
        "corpus_path": "data/wiki18/wiki_dump.jsonl",
        "index_path": "data/wiki18/e5_Flat.index",
    },
]


#####################################
# Corpus
#####################################

def load_corpus(corpus_path: str, num_proc: int = 4):
    return datasets.load_dataset(
        "json",
        data_files=corpus_path,
        split="train",
        num_proc=num_proc,
    )


#####################################
# Encoder
#####################################

def pooling(pooler_output, last_hidden_state, attention_mask, method="mean"):
    if method == "mean":
        masked = last_hidden_state.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        denom = attention_mask.sum(dim=1).clamp(min=1)[..., None]
        return masked.sum(dim=1) / denom
    elif method == "cls":
        return last_hidden_state[:, 0]
    elif method == "pooler":
        return pooler_output if pooler_output is not None else last_hidden_state[:, 0]
    else:
        raise ValueError(method)


class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16, device):
        self.model_name = (model_name or "").lower()
        self.pooling_method = pooling_method
        self.max_length = max_length

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path, config=config, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, trust_remote_code=True
        )

        self.device = device
        self.model.eval().to(self.device)
        if use_fp16:
            self.model.half()

    @torch.no_grad()
    def encode(self, texts: List[str], is_query=True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        if "e5" in self.model_name:
            prefix = "query: " if is_query else "passage: "
            texts = [prefix + t for t in texts]

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        out = self.model(**inputs, return_dict=True)
        emb = pooling(
            out.pooler_output if hasattr(out, "pooler_output") else None,
            out.last_hidden_state,
            inputs["attention_mask"],
            self.pooling_method,
        )
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.float().cpu().numpy()


#####################################
# Snippet parsing
#####################################

def parse_title_and_first_sentence(contents: str, min_tokens: int = 20):
    if not contents:
        return "", ""

    lines = contents.strip().splitlines()
    title = lines[0].strip().strip('"')

    body = " ".join(lines[1:]).strip()
    if not body:
        return title, ""

    for m in re.finditer(r"\.", body):
        end = m.end()
        prefix = body[:end]
        if len(prefix.split()) >= min_tokens:
            return title, prefix

    tokens = body.split()
    return title, " ".join(tokens[:min_tokens])


#####################################
# FAISS helpers（Key：streaming GPU Flat）
#####################################

def set_faiss_nprobe(index: faiss.Index, nprobe: Optional[int]):
    if nprobe is None:
        return
    if hasattr(index, "nprobe"):
        index.nprobe = nprobe


def _unwrap_idmap_and_get_ids(index: faiss.Index):
    ids = None
    cur = index
    if isinstance(cur, (faiss.IndexIDMap, faiss.IndexIDMap2)):
        try:
            ids = faiss.vector_to_array(cur.id_map).astype(np.int64, copy=False)
        except Exception:
            ids = None
        cur = cur.index
    if isinstance(cur, faiss.IndexPreTransform):
        cur = cur.index
    return faiss.downcast_index(cur), ids


def _is_flat_index(index: faiss.Index) -> bool:
    base, _ = _unwrap_idmap_and_get_ids(index)
    return isinstance(base, faiss.IndexFlat)


def _parse_gpu_ids(gpu_ids_str: str):
    if not gpu_ids_str:
        return []
    return [int(x) for x in gpu_ids_str.split(",") if x.strip()]


def _build_gpu_flat_streaming(cpu_index, gpu_ids, use_fp16, add_batch):
    base, ids = _unwrap_idmap_and_get_ids(cpu_index)
    if ids is not None:
        raise ValueError("GPU Flat build does not support an ID map")
    return build_gpu_flat(base, gpu_ids, use_fp16=use_fp16, add_batch=add_batch)


#####################################
# Dense Retriever
#####################################

class DenseRetriever:
    def __init__(self, config, index_path, corpus_path, global_offset: int):
        self.batch_size = config.batch_size
        self.global_offset = global_offset

        cpu_index = faiss.read_index(index_path)
        set_faiss_nprobe(cpu_index, config.nprobe)

        if config.faiss_gpu:
            gpu_ids = _parse_gpu_ids(config.gpu_ids)
            if not gpu_ids:
                gpu_ids = list(range(faiss.get_num_gpus()))

            if _is_flat_index(cpu_index):
                self.index = _build_gpu_flat_streaming(
                    cpu_index,
                    gpu_ids=gpu_ids,
                    use_fp16=config.gpu_use_fp16,
                    add_batch=config.gpu_add_batch,
                )
            else:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(
                    res, gpu_ids[0], cpu_index
                )
        else:
            self.index = cpu_index

        self.faiss_lock = threading.Lock()
        self.corpus = load_corpus(corpus_path, config.corpus_num_proc)

        device = torch.device(
            "cuda" if config.faiss_gpu and torch.cuda.is_available() else "cpu"
        )
        self.encoder = Encoder(
            config.retriever_name,
            config.retriever_model,
            config.pooling_method,
            config.max_length,
            config.use_fp16,
            device=device
        )

    def batch_search(self, queries: List[str], topk: int):
        all_scores = []
        all_ids = []

        for s in range(0, len(queries), self.batch_size):
            batch = queries[s : s + self.batch_size]
            emb = self.encoder.encode(batch, is_query=True)

            with self.faiss_lock:
                scores, idxs = self.index.search(emb, topk)

            scores = scores.tolist()
            idxs = [[int(i) + self.global_offset for i in row] for row in idxs.tolist()]

            all_scores.extend(scores)
            all_ids.extend(idxs)

        return all_scores, all_ids


#####################################
# Multi Corpus Retriever
#####################################

class MultiCorpusRetriever:
    def __init__(self, retrievers):
        self.retrievers = retrievers

    def batch_search(self, queries, topk):
        all_scores = [[] for _ in queries]
        all_ids = [[] for _ in queries]

        for r in self.retrievers:
            scores, ids = r.batch_search(queries, topk)
            for i in range(len(queries)):
                all_scores[i].extend(scores[i])
                all_ids[i].extend(ids[i])

        merged = []
        for s, i in zip(all_scores, all_ids):
            merged.append(
                sorted(zip(s, i), reverse=True)[:topk]
            )
        return merged


#####################################
# Cross-request batching
#####################################

class RetrievalBatchItem:
    def __init__(self, queries, topk):
        self.queries = queries
        self.topk = topk
        self.done = threading.Event()
        self.result = None
        self.error = None


class RetrievalBatcher:
    def __init__(self, backend, max_queries, wait_ms):
        self.backend = backend
        self.max_queries = max(1, max_queries)
        self.wait_s = max(0.0, wait_ms / 1000.0)
        self.items = queue.Queue()
        self.stopping = False
        self.worker = threading.Thread(
            target=self._run,
            name="retrieval-batcher",
            daemon=True,
        )
        self.worker.start()

    def submit(self, queries, topk):
        item = RetrievalBatchItem(queries, topk)
        self.items.put(item)
        item.done.wait()
        if item.error is not None:
            raise item.error
        return item.result

    def stop(self):
        self.stopping = True
        self.items.put(None)
        self.worker.join(timeout=30)

    def _run(self):
        while True:
            first = self.items.get()
            if first is None:
                return

            batch = [first]
            query_count = len(first.queries)
            deadline = time.monotonic() + self.wait_s

            while query_count < self.max_queries:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    item = self.items.get(timeout=timeout)
                except queue.Empty:
                    break
                if item is None:
                    self.stopping = True
                    break
                batch.append(item)
                query_count += len(item.queries)

            try:
                flat_queries = [
                    query
                    for item in batch
                    for query in item.queries
                ]
                max_topk = max(item.topk for item in batch)
                merged = self.backend.batch_search(flat_queries, max_topk)

                offset = 0
                for item in batch:
                    size = len(item.queries)
                    item.result = [
                        row[: item.topk]
                        for row in merged[offset : offset + size]
                    ]
                    offset += size
            except Exception as exc:
                for item in batch:
                    item.error = exc
            finally:
                for item in batch:
                    item.done.set()

            if self.stopping:
                return


#####################################
# Global ID Mapper
#####################################

class GlobalIdMapper:
    def __init__(self):
        self.ranges = []

    def register(self, size: int):
        start = self.ranges[-1][1] if self.ranges else 0
        end = start + size
        self.ranges.append((start, end))
        return start

    def global_to_local(self, gid: int):
        for cid, (s, e) in enumerate(self.ranges):
            if s <= gid < e:
                return cid, gid - s
        raise ValueError(gid)


#####################################
# FastAPI
#####################################

class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


class ExpandRequest(BaseModel):
    doc_ids: List[int]


app = FastAPI()
retriever = None
retrievers = None
id_mapper = None
retrieval_batcher = None


@app.on_event("startup")
async def raise_request_thread_limit():
    # /retrieve is a sync handler, so Starlette runs it in anyio's thread pool (40 threads by
    # default): at most 40 requests reach the batcher at once. RETRIEVER_REQUEST_THREADS
    # raises the limit so a whole rollout turn of queries can share one index scan.
    n = int(os.environ.get("RETRIEVER_REQUEST_THREADS", "0") or 0)
    if n > 0:
        import anyio.to_thread
        anyio.to_thread.current_default_thread_limiter().total_tokens = n
        print(f"[server] request thread limit = {n}")


@app.on_event("startup")
def startup():
    global retriever, retrievers, id_mapper, retrieval_batcher

    cfg = app.state.config
    retrievers = []
    id_mapper = GlobalIdMapper()

    for item in manifest:
        corpus = load_corpus(item["corpus_path"], cfg.corpus_num_proc)
        print(f"[Corpus] {item['name']} size = {len(corpus)}")
        offset = id_mapper.register(len(corpus))

        r = DenseRetriever(
            cfg,
            item["index_path"],
            item["corpus_path"],
            offset,
        )
        retrievers.append(r)

    retriever = MultiCorpusRetriever(retrievers)
    retrieval_batcher = RetrievalBatcher(
        retriever,
        max_queries=cfg.request_batch_size,
        wait_ms=cfg.request_batch_wait_ms,
    )


@app.on_event("shutdown")
def shutdown():
    if retrieval_batcher is not None:
        retrieval_batcher.stop()


@app.post("/retrieve")
def retrieve(req: QueryRequest):
    topk = req.topk or app.state.config.topk
    merged = retrieval_batcher.submit(req.queries, topk)

    results = []
    for items in merged:
        out = []
        for score, gid in items:
            cid, local = id_mapper.global_to_local(gid)
            doc = retrievers[cid].corpus[local]

            contents = doc.get("contents", "")
            title, snippet = parse_title_and_first_sentence(contents)

            item = {
                "doc_id": gid,
                "title": title,
                "snippet": snippet,
                "full_text": contents
            }
            if req.return_scores:
                item["score"] = float(score)

            out.append(item)
        results.append(out)

    return {"result": results}



@app.post("/expand")
def expand(req: ExpandRequest):
    out = []
    for gid in req.doc_ids:
        cid, local = id_mapper.global_to_local(gid)
        doc = retrievers[cid].corpus[local]
        out.append({"doc_id": gid, "contents": doc.get("contents", "")})
    return {"expanded": out}


#####################################
# Main
#####################################

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--retriever_model", default="intfloat/e5-base-v2")
    parser.add_argument("--retriever_name", default="e5")

    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--request_batch_size", type=int, default=64)
    parser.add_argument("--request_batch_wait_ms", type=float, default=5.0)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--pooling_method", default="mean")
    parser.add_argument("--use_fp16", action="store_true")

    parser.add_argument("--corpus_num_proc", type=int, default=4)

    # FAISS GPU
    parser.add_argument("--faiss_gpu", action="store_true")
    parser.add_argument("--gpu_use_fp16", action="store_true")
    parser.add_argument("--gpu_ids", type=str, default="")
    parser.add_argument("--gpu_add_batch", type=int, default=200000)
    parser.add_argument("--nprobe", type=int, default=None)

    parser.add_argument("--port", type=int, default=8000)
    # Loopback by default: the evaluator reaches the retriever on the same node. A run
    # that serves another host must pass --host explicitly.
    parser.add_argument("--host", type=str, default="127.0.0.1")

    args = parser.parse_args()
    app.state.config = args

    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
