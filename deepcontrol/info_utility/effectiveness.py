from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def total_variation(p_prev: torch.Tensor, p_cur: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.abs(p_cur - p_prev).sum()


@torch.no_grad()
def generate_cot_once(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    device: str,
    max_new_tokens: int,
) -> str:
    prompt = (
        "<|im_start|>user\n"
        f"Question: {question}\n\n"
        "Please reason step by step.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    if text.endswith("<|im_end|>"):
        text = text[:-len("<|im_end|>")]
    return text.strip()


@torch.no_grad()
def answer_distribution_with_cot_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_with_cot: str,
    candidates: List[str],
    device: str,
) -> List[float]:
    """
    Compute P(candidate | prompt_with_cot) for all candidates in one forward pass.
    """
    prefix_ids = tokenizer(
        prompt_with_cot,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)

    ans_token_ids = [tokenizer(c, add_special_tokens=False).input_ids for c in candidates]
    max_ans_len = max(len(x) for x in ans_token_ids)

    padded_ans, ans_mask = [], []
    for ids in ans_token_ids:
        pad_len = max_ans_len - len(ids)
        padded_ans.append(ids + [tokenizer.pad_token_id] * pad_len)
        ans_mask.append([1] * len(ids) + [0] * pad_len)

    ans_ids = torch.tensor(padded_ans, device=device)
    ans_mask = torch.tensor(ans_mask, device=device)

    B = ans_ids.size(0)
    prefix_ids = prefix_ids.repeat(B, 1)
    input_ids = torch.cat([prefix_ids, ans_ids], dim=1)
    logits = model(input_ids=input_ids).logits

    start = prefix_ids.shape[1] - 1
    ans_logits = logits[:, start : start + max_ans_len, :]

    log_probs = F.log_softmax(ans_logits, dim=-1)
    token_logps = log_probs.gather(2, ans_ids.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * ans_mask
    seq_logps = token_logps.sum(dim=1)

    probs = torch.softmax(seq_logps, dim=0)
    return probs.detach().cpu().tolist()


@dataclass
class OfflineEffectivenessConfig:
    cot_tokens: int = 256


class OfflineEffectivenessScorer:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str,
        config: OfflineEffectivenessConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config

    def effectiveness_curve(
        self,
        question: str,
        turn_to_texts: dict,
        candidates: List[str],
    ) -> List[float]:
        cot_text = generate_cot_once(
            self.model, self.tokenizer, question, self.device, max_new_tokens=self.config.cot_tokens
        )
        base_prompt = cot_text + "\nAnswer:"
        p_prev = torch.tensor(
            answer_distribution_with_cot_batch(
                self.model, self.tokenizer, base_prompt, candidates, self.device
            ),
            device="cpu",
            dtype=torch.float32,
        )

        curve: List[float] = []
        for _, texts in turn_to_texts.items():
            evidence = "\n\n".join(texts)
            prompt_t = cot_text + "\nEvidence:\n" + evidence + "\nAnswer:"
            p_cur = torch.tensor(
                answer_distribution_with_cot_batch(
                    self.model, self.tokenizer, prompt_t, candidates, self.device
                ),
                device="cpu",
                dtype=torch.float32,
            )
            tv = total_variation(p_prev, p_cur).item()
            curve.append(float(tv))
            p_prev = p_cur
        return curve


@dataclass
class OnlineEffectivenessState:
    question: str
    candidates: List[str]
    cot_text: str
    p_prev: torch.Tensor
    turn_idx: int = 0
    evidence_history: List[str] = field(default_factory=list)


@dataclass
class OnlineEffectivenessResult:
    effectiveness: float
    turn_idx: int
    answer_dist: List[float]


class OnlineEffectivenessScorer:
    """
    Incremental effectiveness scorer for rollout-time control.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str,
        cot_tokens: int = 256,
        evidence_mode: Literal["incremental", "cumulative"] = "incremental",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.cot_tokens = cot_tokens
        self.evidence_mode = evidence_mode

    def init_state(self, question: str, candidates: List[str]) -> OnlineEffectivenessState:
        cot_text = generate_cot_once(
            self.model,
            self.tokenizer,
            question,
            self.device,
            max_new_tokens=self.cot_tokens,
        )
        base_prompt = cot_text + "\nAnswer:"
        p_prev = torch.tensor(
            answer_distribution_with_cot_batch(
                self.model, self.tokenizer, base_prompt, candidates, self.device
            ),
            device="cpu",
            dtype=torch.float32,
        )
        return OnlineEffectivenessState(
            question=question,
            candidates=candidates,
            cot_text=cot_text,
            p_prev=p_prev,
        )

    def update(self, state: OnlineEffectivenessState, evidence_text: str) -> OnlineEffectivenessResult:
        state.turn_idx += 1
        evidence_text = evidence_text.strip()
        if evidence_text:
            state.evidence_history.append(evidence_text)

        if self.evidence_mode == "cumulative":
            evidence_block = "\n\n".join(state.evidence_history)
        else:
            evidence_block = evidence_text

        if evidence_block:
            prompt_t = state.cot_text + "\nEvidence:\n" + evidence_block + "\nAnswer:"
        else:
            prompt_t = state.cot_text + "\nAnswer:"

        p_cur = torch.tensor(
            answer_distribution_with_cot_batch(
                self.model, self.tokenizer, prompt_t, state.candidates, self.device
            ),
            device="cpu",
            dtype=torch.float32,
        )
        tv = total_variation(state.p_prev, p_cur).item()
        state.p_prev = p_cur
        return OnlineEffectivenessResult(
            effectiveness=float(tv),
            turn_idx=state.turn_idx,
            answer_dist=p_cur.tolist(),
        )
