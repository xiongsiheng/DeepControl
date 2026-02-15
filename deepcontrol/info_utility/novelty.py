from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, DefaultDict, Dict, List, Optional
from collections import defaultdict

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def group_retrieval_by_turn(retrieval_log: List[dict]) -> Dict[int, List[str]]:
    """Group retrieved full_text by retrieval turn."""
    turn_to_texts: DefaultDict[int, List[str]] = defaultdict(list)
    for item in retrieval_log or []:
        turn = item.get("turn")
        full_text = item.get("full_text")
        if turn is None or not full_text:
            continue
        turn_to_texts[int(turn)].append(full_text.strip())
    return dict(sorted(turn_to_texts.items(), key=lambda x: x[0]))


class PassageEncoder:
    """Sentence encoder used by novelty scoring."""

    def __init__(self, model_name: str, device: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def encode(self, texts: List[str], max_length: int = 256) -> np.ndarray:
        prefixed = ["passage: " + t for t in texts]
        inputs = self.tokenizer(
            prefixed,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        out = self.model(**inputs)
        embs = out.last_hidden_state.mean(dim=1)
        embs = F.normalize(embs, dim=-1)
        return embs.detach().cpu().numpy().astype(np.float32)


class NoveltyBank:
    """Cosine similarity on normalized vectors equals inner product."""

    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def add(self, embs: np.ndarray) -> None:
        if embs is not None and len(embs) > 0:
            self.index.add(embs)

    def novelty(self, embs: np.ndarray, k: int) -> np.ndarray:
        if self.index.ntotal == 0:
            return np.ones((len(embs),), dtype=np.float32)
        k = min(int(k), self.index.ntotal)
        sims, _ = self.index.search(embs, k)
        return 1.0 - sims.mean(axis=1)


@dataclass
class OfflineNoveltyConfig:
    knn_k: int = 5
    emb_dim: int = 768
    enc_max_length: int = 256


class OfflineNoveltyScorer:
    def __init__(self, encoder: PassageEncoder, config: OfflineNoveltyConfig):
        self.encoder = encoder
        self.config = config

    def novelty_curve(self, turn_to_texts: Dict[int, List[str]]) -> List[float]:
        bank = NoveltyBank(dim=self.config.emb_dim)
        curve: List[float] = []
        for _, texts in turn_to_texts.items():
            if not texts:
                curve.append(float("nan"))
                continue
            embs = self.encoder.encode(texts, max_length=self.config.enc_max_length)
            nov = bank.novelty(embs, k=self.config.knn_k).mean()
            curve.append(float(nov))
            bank.add(embs)
        return curve


@dataclass
class OnlineNoveltyState:
    turn_idx: int = 0
    seen_doc_ids: set = field(default_factory=set)


@dataclass
class OnlineNoveltyResult:
    novelty: float
    num_passages: int
    new_doc_ratio: Optional[float]
    turn_idx: int


class OnlineNoveltyScorer:
    """
    Incremental novelty scorer for rollout-time control.
    """

    def __init__(self, encoder: PassageEncoder, knn_k: int, emb_dim: int, enc_max_length: int = 256):
        self.encoder = encoder
        self.knn_k = knn_k
        self.enc_max_length = enc_max_length
        self.bank = NoveltyBank(dim=emb_dim)

    def update_from_texts(self, texts: List[str], state: OnlineNoveltyState) -> OnlineNoveltyResult:
        state.turn_idx += 1
        if not texts:
            return OnlineNoveltyResult(
                novelty=0.0,
                num_passages=0,
                new_doc_ratio=None,
                turn_idx=state.turn_idx,
            )

        embs = self.encoder.encode(texts, max_length=self.enc_max_length)
        nov = float(self.bank.novelty(embs, k=self.knn_k).mean())
        self.bank.add(embs)
        return OnlineNoveltyResult(
            novelty=nov,
            num_passages=len(texts),
            new_doc_ratio=None,
            turn_idx=state.turn_idx,
        )

    def update_from_entries(
        self,
        entries: List[Dict[str, Any]],
        state: OnlineNoveltyState,
        text_key: str = "full_text",
        doc_id_key: str = "doc_id",
    ) -> OnlineNoveltyResult:
        texts = [str(e.get(text_key, "")).strip() for e in entries if str(e.get(text_key, "")).strip()]
        doc_ids = [e.get(doc_id_key) for e in entries if e.get(doc_id_key) is not None]
        prev_seen = len(state.seen_doc_ids)
        state.seen_doc_ids.update(doc_ids)

        out = self.update_from_texts(texts, state=state)
        if doc_ids:
            new_seen = len(state.seen_doc_ids) - prev_seen
            out.new_doc_ratio = float(new_seen / max(len(doc_ids), 1))
        return out
