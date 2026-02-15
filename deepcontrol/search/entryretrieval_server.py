import argparse
import threading
import re
from typing import List, Optional, Tuple

import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
import datasets

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


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

def safe_load_docs(corpus, doc_idxs: List[int]) -> List[dict]:
    out = []
    for i in doc_idxs:
        i = int(i)
        if i < 0:
            out.append({"contents": ""})
        else:
            out.append(corpus[i])
    return out


#####################################
# Encoder
#####################################

def pooling(pooler_output, last_hidden_state, attention_mask, method="mean"):
    if method == "mean":
        masked = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        denom = attention_mask.sum(dim=1).clamp(min=1)[..., None]
        return masked.sum(dim=1) / denom
    elif method == "cls":
        return last_hidden_state[:, 0]
    elif method == "pooler":
        if pooler_output is None:
            return last_hidden_state[:, 0]
        return pooler_output
    else:
        raise ValueError(f"Unknown pooling method: {method}")

class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = (model_name or "").lower()
        self.pooling_method = pooling_method
        self.max_length = max_length

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)

        self.model.eval().cuda()
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
        ).to("cuda")

        output = self.model(**inputs, return_dict=True)
        emb = pooling(
            output.pooler_output if hasattr(output, "pooler_output") else None,
            output.last_hidden_state,
            inputs["attention_mask"],
            self.pooling_method,
        )

        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.to(torch.float32).detach().cpu().numpy()


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
# FAISS helpers
#####################################

def set_faiss_nprobe(index: faiss.Index, nprobe: Optional[int]):
    if nprobe is None:
        return
    if nprobe <= 0:
        raise ValueError(f"nprobe must be > 0, got {nprobe}")

    try:
        if hasattr(index, "nprobe"):
            index.nprobe = nprobe
            return
    except Exception:
        pass

    try:
        ps = faiss.ParameterSpace()
        ps.set_index_parameter(index, "nprobe", nprobe)
    except Exception:
        return

def _unwrap_idmap_and_get_ids(index: faiss.Index):
    """
    Returns: (base_index, ids or None)
    If index is IndexIDMap/IndexIDMap2, extract ids.
    """
    ids = None
    cur = index
    # unwrap IDMap
    if isinstance(cur, faiss.IndexIDMap) or isinstance(cur, faiss.IndexIDMap2):
        try:
            ids = faiss.vector_to_array(cur.id_map).astype(np.int64, copy=False)
        except Exception:
            ids = None
        cur = cur.index
    # unwrap PreTransform (rare in your setup, but safe)
    if isinstance(cur, faiss.IndexPreTransform):
        cur = cur.index
    return cur, ids

def _is_flat_index(index: faiss.Index) -> bool:
    base, _ = _unwrap_idmap_and_get_ids(index)
    base = faiss.downcast_index(base)
    return isinstance(base, faiss.IndexFlat)



def _build_gpu_flat_streaming(cpu_index, gpu_ids, use_fp16, add_batch):
    base, ids = _unwrap_idmap_and_get_ids(cpu_index)
    base = faiss.downcast_index(base)
    if not isinstance(base, faiss.IndexFlat):
        raise ValueError("streaming GPU build only supports IndexFlat/IndexFlatIP/IndexFlatL2")

    ntotal, d = base.ntotal, base.d
    if ids is None:
        ids = np.arange(ntotal, dtype=np.int64)
    else:
        ids = np.asarray(ids, dtype=np.int64)

    metric = base.metric_type
    if metric not in (faiss.METRIC_INNER_PRODUCT, faiss.METRIC_L2):
        raise ValueError(f"Unsupported metric for Flat streaming build: {metric}")
    if len(gpu_ids) == 0:
        raise RuntimeError("No GPU ids provided/visible for FAISS.")

    def make_gpu_idmap(gid: int):
        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.device = int(gid)
        cfg.useFloat16 = bool(use_fp16)
        if metric == faiss.METRIC_INNER_PRODUCT:
            gpu_flat = faiss.GpuIndexFlatIP(res, d, cfg)
        else:
            gpu_flat = faiss.GpuIndexFlatL2(res, d, cfg)
        return faiss.IndexIDMap2(gpu_flat)

    # ----- single GPU: no shards needed (most compatible) -----
    if len(gpu_ids) == 1:
        index = make_gpu_idmap(gpu_ids[0])
        add_batch = int(add_batch)
        for s in range(0, ntotal, add_batch):
            e = min(ntotal, s + add_batch)
            xb = base.reconstruct_n(int(s), int(e - s))
            xb = np.ascontiguousarray(xb, dtype=np.float32)
            idb = np.ascontiguousarray(ids[s:e], dtype=np.int64)
            index.add_with_ids(xb, idb)
        return index

    # ----- multi GPU: use IndexShards (avoid kwargs; set attrs if exist) -----
    try:
        shards = faiss.IndexShards(d)
    except TypeError:
        shards = faiss.IndexShards()
    if hasattr(shards, "threaded"):
        try: shards.threaded = False
        except Exception: pass
    if hasattr(shards, "successive_ids"):
        try: shards.successive_ids = False
        except Exception: pass

    shard_list = []
    for gid in gpu_ids:
        shard = make_gpu_idmap(gid)
        shards.add_shard(shard)
        shard_list.append(shard)

    n = len(shard_list)
    ranges = []
    for i in range(n):
        s0 = (ntotal * i) // n
        e0 = (ntotal * (i + 1)) // n
        ranges.append((s0, e0))

    add_batch = int(add_batch)
    for shard_i, (s0, e0) in enumerate(ranges):
        shard = shard_list[shard_i]
        for s in range(s0, e0, add_batch):
            e = min(e0, s + add_batch)
            xb = base.reconstruct_n(int(s), int(e - s))
            xb = np.ascontiguousarray(xb, dtype=np.float32)
            idb = np.ascontiguousarray(ids[s:e], dtype=np.int64)
            shard.add_with_ids(xb, idb)

    return shards



def _parse_gpu_ids(gpu_ids_str: str) -> List[int]:
    if gpu_ids_str is None:
        return []
    s = gpu_ids_str.strip()
    if not s:
        return []
    return [int(x) for x in s.split(",") if x.strip() != ""]


#####################################
# Retriever
#####################################

class DenseRetriever:
    def __init__(self, config):
        self.topk = config.topk
        self.batch_size = config.batch_size

        # 1) load CPU index
        cpu_index = faiss.read_index(config.index_path)
        if hasattr(cpu_index, "is_trained") and not cpu_index.is_trained:
            raise RuntimeError(
                f"FAISS index is not trained: {config.index_path}. "
                "IVF/IVFPQ must be trained before use."
            )

        # set nprobe on CPU (safe no-op for Flat)
        set_faiss_nprobe(cpu_index, config.nprobe)

        # 2) optional GPU
        if config.faiss_gpu:
            # figure out which GPUs to use
            gpu_ids = _parse_gpu_ids(config.gpu_ids)
            if not gpu_ids:
                # use all visible GPUs
                ng = faiss.get_num_gpus()
                gpu_ids = list(range(ng))

            # For Flat: use streaming build to avoid VRAM peak spike
            if _is_flat_index(cpu_index):
                self.index = _build_gpu_flat_streaming(
                    cpu_index=cpu_index,
                    gpu_ids=gpu_ids,
                    use_fp16=bool(config.gpu_use_fp16),
                    add_batch=int(config.gpu_add_batch),
                )
            else:
                # Non-flat: keep your old clone path (IVF/IVFPQ etc.)
                # If multiple GPUs, shard clone.
                if len(gpu_ids) > 1:
                    co = faiss.GpuMultipleClonerOptions()
                    co.useFloat16 = bool(config.gpu_use_fp16)
                    co.shard = True

                    res_vec = faiss.GpuResourcesVector()
                    dev_vec = faiss.IntVector()
                    for gid in gpu_ids:
                        res_vec.push_back(faiss.StandardGpuResources())
                        dev_vec.push_back(int(gid))

                    self.index = faiss.index_cpu_to_gpu_multiple(res_vec, dev_vec, cpu_index, co)
                else:
                    res = faiss.StandardGpuResources()
                    co = faiss.GpuClonerOptions()
                    co.useFloat16 = bool(config.gpu_use_fp16)
                    self.index = faiss.index_cpu_to_gpu(res, int(gpu_ids[0]), cpu_index, co)

                # set nprobe again (safe)
                set_faiss_nprobe(self.index, config.nprobe)
        else:
            self.index = cpu_index

        # FAISS GPU not thread-safe
        self.faiss_lock = threading.Lock()

        # corpus + encoder
        self.corpus = load_corpus(config.corpus_path, num_proc=config.corpus_num_proc)
        self.encoder = Encoder(
            model_name=config.retriever_name,
            model_path=config.retriever_model,
            pooling_method=config.pooling_method,
            max_length=config.max_length,
            use_fp16=config.use_fp16,
        )

    def batch_search(
        self, queries: List[str], topk: int
    ) -> Tuple[List[List[dict]], List[List[float]], List[List[int]]]:
        """
        Returns:
          results_docs:      List[ [doc, doc, ...] ] per query
          results_scores:    List[ [score, ...] ] per query
          results_row_idxs:  List[ [row_idx, ...] ] per query  (for /expand)
        """
        results_docs: List[List[dict]] = []
        results_scores: List[List[float]] = []
        results_row_idxs: List[List[int]] = []

        for start in range(0, len(queries), self.batch_size):
            batch = queries[start:start + self.batch_size]
            emb = self.encoder.encode(batch, is_query=True)

            with self.faiss_lock:
                batch_scores, batch_idxs = self.index.search(emb, topk)

            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            flat_idxs = [int(x) for row in batch_idxs for x in row]
            docs = safe_load_docs(self.corpus, flat_idxs)

            B = len(batch_idxs)
            docs = [docs[i * topk:(i + 1) * topk] for i in range(B)]

            results_docs.extend(docs)
            results_scores.extend(batch_scores)
            results_row_idxs.extend(batch_idxs)

        return results_docs, results_scores, results_row_idxs


#####################################
# FastAPI schema
#####################################

class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False

class ExpandRequest(BaseModel):
    doc_ids: List[str]


#####################################
# FastAPI App
#####################################

app = FastAPI()
retriever: Optional[DenseRetriever] = None

@app.on_event("startup")
def load_retriever():
    global retriever
    retriever = DenseRetriever(app.state.config)

@app.post("/retrieve")
def retrieve(req: QueryRequest):
    assert retriever is not None
    topk = req.topk or retriever.topk

    docs_per_q, scores_per_q, row_idxs_per_q = retriever.batch_search(req.queries, topk)

    merged = []
    for docs, scs, row_idxs in zip(docs_per_q, scores_per_q, row_idxs_per_q):
        items = []
        for d, s, row_idx in zip(docs, scs, row_idxs):
            row_idx = int(row_idx)
            if row_idx < 0:
                item = {"doc_id": "-1", "title": "", "snippet": ""}
                if req.return_scores:
                    item["score"] = float(s)
                items.append(item)
                continue

            contents = d.get("contents", "")
            title, snippet = parse_title_and_first_sentence(contents)

            item = {"doc_id": str(row_idx), "title": title, "snippet": snippet}
            if req.return_scores:
                item["score"] = float(s)
            items.append(item)

        merged.append(items)

    return {"result": merged}

@app.post("/expand")
def expand(req: ExpandRequest):
    assert retriever is not None
    expanded = []
    for doc_id in req.doc_ids:
        row_idx = int(doc_id)
        if row_idx < 0:
            expanded.append({"doc_id": "-1", "contents": ""})
            continue
        doc = retriever.corpus[row_idx]
        expanded.append({"doc_id": str(row_idx), "contents": doc.get("contents", "")})
    return {"expanded": expanded}


#####################################
# Main
#####################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--corpus_path", required=True)

    parser.add_argument("--retriever_model", default="intfloat/e5-base-v2")
    parser.add_argument("--retriever_name", default="e5")

    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--pooling_method", default="mean")
    parser.add_argument("--use_fp16", action="store_true")

    parser.add_argument("--corpus_num_proc", type=int, default=4)

    # FAISS GPU
    parser.add_argument("--faiss_gpu", action="store_true")
    parser.add_argument("--gpu_use_fp16", action="store_true")

    # multi-gpu control (empty -> all GPUs)
    parser.add_argument("--gpu_ids", type=str, default="",
                        help='Comma-separated GPU ids, e.g. "0,1,2,3". Empty => use all visible GPUs.')

    # streaming add batch size (controls peak VRAM during build)
    parser.add_argument("--gpu_add_batch", type=int, default=200000,
                        help="How many vectors per add_with_ids chunk when building GPU Flat.")

    # IVF only (safe no-op for Flat)
    parser.add_argument("--nprobe", type=int, default=None)

    parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    app.state.config = args

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        workers=1,  # FAISS GPU: keep 1
    )

if __name__ == "__main__":
    main()
