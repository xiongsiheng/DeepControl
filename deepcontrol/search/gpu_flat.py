import faiss
import numpy as np

# GPU indexes and resources must outlive the IndexShards that refers to them
_KEEP = []


def build_gpu_flat(base, gpu_ids, use_fp16=True, add_batch=200_000, slice_vectors=1_000_000):
    """Copy a CPU IndexFlat to the GPUs in gpu_ids.

    The vectors go into GpuIndexFlat slices of slice_vectors each, spread over the GPUs and searched
    as one index; each slice is filled add_batch vectors at a time. A GpuIndexFlat over 128 MB grows
    to the exact new size on every add and copies its data, and a single add stages the whole input
    on the GPU in float32, so the peak is the index plus one slice rather than several times the index.
    """
    if not isinstance(base, faiss.IndexFlat):
        raise ValueError("GPU build supports Flat indexes only")
    resources = {}
    for g in gpu_ids:
        resources[g] = faiss.StandardGpuResources()
        resources[g].setTempMemory(256 * 1024 * 1024)
    _KEEP.extend(resources.values())

    # serial search (slices on one GPU share its resources); ids continue from slice to slice
    index = faiss.IndexShards(base.d, False, True)
    for n, start in enumerate(range(0, base.ntotal, slice_vectors)):
        config = faiss.GpuIndexFlatConfig()
        config.device = gpu_ids[n % len(gpu_ids)]
        config.useFloat16 = bool(use_fp16)
        part = faiss.GpuIndexFlat(resources[config.device], base.d, base.metric_type, config)
        end = min(base.ntotal, start + slice_vectors)
        for s in range(start, end, add_batch):
            e = min(end, s + add_batch)
            part.add(np.ascontiguousarray(base.reconstruct_n(s, e - s), dtype=np.float32))
        index.add_shard(part)
        _KEEP.append(part)
        print(f"[FAISS GPU] uploaded {end}/{base.ntotal} vectors", flush=True)
    print(f"[FAISS GPU] {base.ntotal} vectors on GPUs {gpu_ids}", flush=True)
    return index
