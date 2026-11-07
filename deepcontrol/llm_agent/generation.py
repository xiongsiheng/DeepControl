import os
import re
import math
import json
import ast
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
from verl import DataProto

from .tensor_helper import TensorHelper, TensorConfig
from .deep_research_agent import DeepResearchAgent
from .entry_retriever import EntryRetrieverClient
from ..info_utility.novelty import PassageEncoder, NoveltyBank
from ..protocol import (
    CONTROL_CONTINUE, CONTROL_NO_EXPAND, CONTROL_STOP, EMPTY_INFORMATION, EpisodeState,
    control_expand, information_observation, invalid_action, search_observation,
)
from ..info_utility.utility import (
    OnlineUtilityConfig,
    OnlineUtilityController,
    OnlineUtilityState,
)

# control_mode=normal: Expand is sent only when the best document's gold-answer score (mean
# log-likelihood) beats the runner-up by at least this margin
EXPAND_SCORE_MARGIN = 1.0


@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int
    max_response_length: int
    max_obs_length: int
    num_gpus: int

    retriever_url: str
    topk: int = 8

    # if you already have these flags in config, keep them
    no_think_rl: bool = False
    # utility controller
    enable_utility_controller: bool = False
    # Information control mode (with enable_utility_controller):
    #   off    - no control (plain PPO rollouts)
    #   normal - the utility controller (novelty and effectiveness), with expansion
    #            guidance only when the best document's gold-answer score beats the runner-up
    #            by EXPAND_SCORE_MARGIN, and continuation by the gold answer
    #   fast   - every control signal decided by string match of the gold answer against
    #            the evidence and the retrieved documents
    control_mode: str = "fast"
    utility_gamma: float = 0.5
    utility_low_threshold: float = 0.10
    utility_low_patience: int = 2
    utility_min_turns_before_stop: int = 2
    utility_force_stop_after_turn: Optional[int] = None
    utility_continue_threshold: float = 0.25
    utility_continue_patience: int = 2
    utility_continue_score_max: float = -2.0
    utility_continue_prob: float = 0.10
    utility_intervention_schedule: Optional[List[float]] = None
    utility_random_seed: Optional[int] = None
    utility_novelty_encoder: str = "intfloat/e5-base-v2"
    utility_novelty_knn_k: int = 5
    utility_novelty_emb_dim: int = 768
    utility_novelty_max_length: int = 256
    # CPU threads for the novelty encoder (the driver task otherwise runs it on one thread)
    utility_novelty_threads: int = 16
    utility_cot_tokens: int = 512
    utility_eff_tau_low: float = 0.05
    utility_eff_tau_high: float = 0.20
    # granularity control
    granularity_control: bool = True
    # at most this many documents per expand action; 0 = no cap
    max_expand_docs: int = 3
    # train on each turn's sampled token ids up to the token that closes its first action,
    # then <|im_end|> (false: decode, cut the text to [think] + action and re-encode it)
    sampled_tokens: bool = True
    # concurrent retriever requests per environment step
    retrieval_workers: int = 64
    # sync the actor weights into vLLM once per multi-turn rollout and keep them there for
    # all of its generate calls
    rollout_weight_cache: bool = True


class LLMGenerationManager:
    """
    Batched rollout loop:
      model emits <think> + one action tag
      env executes action:
        - search -> append <search_results>...</search_results> as observation (masked)
        - expand -> append <information>...</information> as observation (masked), remember span(expand+information)
        - answer -> done
    """

    def __init__(self, tokenizer, actor_rollout_wg, config: GenerationConfig, is_validation: bool = False):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        import dataclasses
        mode = config.control_mode
        if mode == "off":
            config = dataclasses.replace(config, enable_utility_controller=False)
        elif mode not in ("normal", "fast"):
            raise ValueError(f"control_mode must be off, normal or fast, got {mode!r}")
        self.fast_control = mode == "fast"
        self.config = config
        self.is_validation = is_validation

        self.agent = DeepResearchAgent()
        self.retriever = EntryRetrieverClient(base_url=config.retriever_url, topk=config.topk)
        self.novelty_encoder = None
        self.utility_controller = None
        # The controller is a training-time signal only: validation rollouts run without it,
        # as the final policy does at test time.
        if config.enable_utility_controller and not is_validation:
            if not self.fast_control:
                novelty_device = "cuda" if torch.cuda.is_available() else "cpu"
                self.novelty_encoder = PassageEncoder(config.utility_novelty_encoder, novelty_device,
                                                      num_threads=config.utility_novelty_threads)
            self.utility_controller = OnlineUtilityController(
                OnlineUtilityConfig(
                    gamma=config.utility_gamma,
                    low_utility_threshold=config.utility_low_threshold,
                    min_turns_before_stop=config.utility_min_turns_before_stop,
                    low_utility_patience=config.utility_low_patience,
                    force_stop_after_turn=config.utility_force_stop_after_turn,
                    continue_threshold=config.utility_continue_threshold,
                    continue_patience=config.utility_continue_patience,
                    continue_score_max=config.utility_continue_score_max,
                    continue_prob=config.utility_continue_prob,
                    random_seed=config.utility_random_seed,
                )
            )

        self.tensor_fn = TensorHelper(
            TensorConfig(
                pad_token_id=tokenizer.pad_token_id,
                max_prompt_length=config.max_prompt_length,
                max_obs_length=config.max_obs_length,
                max_start_length=config.max_start_length,
            )
        )

    # -----------------------
    # token helpers
    # -----------------------
    def _batch_tokenize(self, texts: List[str]) -> torch.Tensor:
        return self.tokenizer(
            texts,
            add_special_tokens=False,
            return_tensors="pt",
            padding="longest",
        )["input_ids"]

    def _left_padded(self, seqs: List[List[int]], length: Optional[int] = None) -> torch.Tensor:
        """Token lists -> (B, length) tensor, left-padded with pad_token_id (the tokenizer pads on
        the right; vLLM strips only left padding, and the actor scores the last positions)."""
        pad_id = self.tokenizer.pad_token_id
        L = max([len(x) for x in seqs] + [length or 0])
        out = torch.full((len(seqs), L), pad_id, dtype=torch.long)
        for i, x in enumerate(seqs):
            if x:
                out[i, L - len(x):] = torch.tensor(x, dtype=torch.long)
        return out

    def _decode_batch(self, ids: torch.Tensor) -> List[str]:
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)

    @contextmanager
    def _timed(self, name: str):
        """Accumulate wall time of one rollout phase into the trainer's timing_raw as gen_<name>."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            timing = getattr(self, "timing_raw", None)
            if timing is not None:
                timing[f"gen_{name}"] = timing.get(f"gen_{name}", 0.0) + time.perf_counter() - t0

    def _truncate_model_outputs(self, responses: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        if self.config.sampled_tokens:
            return self._cut_sampled_responses(responses)
        outs = self._decode_batch(responses)
        outs = [self.agent.truncate_to_one_action(x) + '<|im_end|>' for x in outs]

        ids = self._batch_tokenize(outs)
        return ids, outs

    def _cut_sampled_responses(self, responses: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """Sampled ids up to the turn's end, cut after the token that closes the first action
        block, then <|im_end|>; right-padded like the tokenizer. Returns ids and their text."""
        tok = self.tokenizer
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        stop = {tok.pad_token_id, im_end, tok.eos_token_id}
        has_action = lambda ids: any(p.search(tok.decode(ids)) for p in self.agent.ACTION_BLOCK_PATTERNS.values())
        seqs = []
        for row in responses.tolist():
            n = next((k for k, t in enumerate(row) if t in stop), len(row))
            ids = row[:n]
            if has_action(ids):
                lo, hi = 1, len(ids)   # smallest prefix that contains a closed action block
                while lo < hi:
                    mid = (lo + hi) // 2
                    if has_action(ids[:mid]):
                        hi = mid
                    else:
                        lo = mid + 1
                ids = ids[:lo]
            seqs.append(ids + [im_end])
        out = torch.full((len(seqs), max(len(x) for x in seqs)), tok.pad_token_id, dtype=torch.long)
        for i, x in enumerate(seqs):
            out[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        return out, [tok.decode(x) for x in seqs]

    def _process_obs(self, obs: List[str], alpha: float=0.4) -> torch.Tensor:
        obs_ids = self.tokenizer(
            obs,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]

        return obs_ids

    # -----------------------
    # padding / concat
    # -----------------------
    def _info_masked_concatenate_with_padding(
        self,
        prompt: torch.Tensor,
        prompt_with_mask: torch.Tensor,
        response: torch.Tensor,
        info: torch.Tensor = None,
        pad_to_left: bool = True,  # not used but keep for interface consistency
    ):
        pad_id = self.tokenizer.pad_token_id
        device = prompt.device
        B = prompt.size(0)

        def _last_nonpad_len(x2d: torch.Tensor) -> torch.Tensor:
            # x2d: [B, L]
            if x2d.size(1) == 0:
                return torch.zeros(B, device=x2d.device, dtype=torch.long)
            mask = (x2d != pad_id).to(torch.long)                    # [B, L]
            idx = torch.arange(x2d.size(1), device=x2d.device).view(1, -1) + 1
            # last position+1 of non-pad; 0 if all-pad
            return (mask * idx).max(dim=1).values                    # [B]

        # Concatenate per sample, first remove trailing padding from each segment to avoid padding becoming "internal pad"
        out_rows = []
        out_mask_rows = []

        p_len = _last_nonpad_len(prompt)
        r_len = _last_nonpad_len(response)
        i_len = _last_nonpad_len(info) if info is not None else None

        for i in range(B):
            a = prompt[i, : p_len[i]]                    # Retain internal padding of fold (it is before p_len[i])
            am = prompt_with_mask[i, : p_len[i]]

            b = response[i, : r_len[i]]
            bm = response[i, : r_len[i]]                 # response follows real tokens in mask

            if info is not None:
                c = info[i, : i_len[i]]
                cm = torch.full((c.numel(),), pad_id, device=device, dtype=prompt_with_mask.dtype)
                seq = torch.cat([a, b, c], dim=0)
                seqm = torch.cat([am, bm, cm], dim=0)
            else:
                seq = torch.cat([a, b], dim=0)
                seqm = torch.cat([am, bm], dim=0)

            out_rows.append(seq)
            out_mask_rows.append(seqm)

        # pad back to matrix
        maxL = max((x.numel() for x in out_rows), default=0)
        out = torch.full((B, maxL), pad_id, device=device, dtype=prompt.dtype)
        out_m = torch.full((B, maxL), pad_id, device=device, dtype=prompt_with_mask.dtype)

        for i in range(B):
            L = out_rows[i].numel()
            if L > 0:
                out[i, :L] = out_rows[i]
                out_m[i, :L] = out_mask_rows[i]

        assert out.shape == out_m.shape
        return out, out_m


    def _update_right_side(
        self,
        right_side: Dict[str, torch.Tensor],
        cur_responses: torch.Tensor,
        next_obs_ids: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        if next_obs_ids is not None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                right_side["responses"],
                right_side["responses_with_info_mask"],
                cur_responses,
                next_obs_ids,
                pad_to_left=False,
            )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                right_side["responses"],
                right_side["responses_with_info_mask"],
                cur_responses,
                pad_to_left=False,
            )

        return {
            "responses": responses,
            "responses_with_info_mask": responses_with_info_mask,
        }

    # -----------------------
    # multi-gpu padding wrapper
    # -----------------------
    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        # Rollout log-probs are recomputed once on the final trajectories by the trainer, so the
        # per-call recompute in generate_sequences is skipped.
        meta_info = dict(active_batch.meta_info or {})
        meta_info["recompute_log_prob"] = False
        active_batch.meta_info = meta_info
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        batch_size = active_batch.batch["input_ids"].shape[0]
        remainder = batch_size % num_gpus

        for k in active_batch.batch.keys():
            active_batch.batch[k] = active_batch.batch[k].long()

        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        padding_size = num_gpus - remainder
        padded_batch = {}
        for k, v in active_batch.batch.items():
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch, meta_info=meta_info)
        for k in padded_active_batch.batch.keys():
            padded_active_batch.batch[k] = padded_active_batch.batch[k].long()

        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}

        if hasattr(padded_output, "meta_info") and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta

        padded_output.batch = trimmed_batch
        return padded_output

    # -----------------------
    # env helpers
    # -----------------------
    def _postprocess_action(self, text: str) -> Tuple[Optional[str], str]:
        text = re.sub(r"<\|im_end\|>\s*$", "", text).strip()
        # parse exactly as inference does (infer_fast: truncate_to_one_action, detect_action)
        action, content = self.agent.detect_action(self.agent.truncate_to_one_action(text))
        return action, content

    def truncate_block_tokens(
        self,
        text: str,
        tokenizer,
        max_tokens: int,
        alpha: float = 0.4,
        marker: str = "\n...[TRUNCATED]...\n",
        head_only: bool = False,
        tail_only: bool = False,
    ) -> str:
        """
        Token-aware head+tail truncation.
        - tokenizer: HF tokenizer
        - max_tokens: budget in tokens
        - alpha: fraction kept for head; tail gets the rest (after marker)
        - marker: inserted between head and tail (counted in tokens)
        """
        if max_tokens <= 0:
            return ""

        ids: List[int] = tokenizer.encode(text, add_special_tokens=False)
        L = len(ids)
        if L <= max_tokens:
            return text

        if head_only:
            marker = "\n...[TRUNCATED]"
            marker_ids = tokenizer.encode(marker, add_special_tokens=False)
            out_ids = ids[:max_tokens] + marker_ids
            return tokenizer.decode(out_ids, skip_special_tokens=True)

        if tail_only:
            out_ids = ids[-max_tokens:]
            return tokenizer.decode(out_ids, skip_special_tokens=True)
        
        marker_ids: List[int] = tokenizer.encode(marker, add_special_tokens=False)
        m = len(marker_ids)

        # If marker itself is too long, just hard crop tokens
        if m >= max_tokens:
            kept = ids[:max_tokens]
            return tokenizer.decode(kept, skip_special_tokens=True)

        budget = max_tokens - m

        head_len = int(budget * alpha)
        head_len = max(1, min(head_len, budget - 1))
        tail_len = budget - head_len

        head_ids = ids[:head_len]
        tail_ids = ids[-tail_len:] if tail_len > 0 else []

        out_ids = head_ids + marker_ids + tail_ids
        # out_ids length == max_tokens (or very close due to rounding guarantees)
        return tokenizer.decode(out_ids, skip_special_tokens=True)

    def _content_knn_novelty(
        self,
        contents: List[str],
        bank: Optional[NoveltyBank],
    ) -> float:
        if self.novelty_encoder is None or bank is None:
            return 0.0
        texts = [c.strip() for c in contents if isinstance(c, str) and c.strip()]
        if not texts:
            return 0.0
        with self._timed("novelty"):
            embs = self.novelty_encoder.encode(texts, max_length=self.config.utility_novelty_max_length)
        nov = float(bank.novelty(embs, k=self.config.utility_novelty_knn_k).mean())
        bank.add(embs)
        return nov

    def _extract_question_from_prompt(self, prompt: str) -> str:
        prompt = re.sub(r"<\|endoftext\|>", "", prompt or "")
        m = re.search(r"Question:\s*(.*)", prompt)
        if not m:
            return (prompt or "").strip()
        q = m.group(1)
        q = re.split(r"\n<\|im_end\|>|\n<\|im_start\|>|\n<control>|</control>", q, maxsplit=1)[0]
        return q.strip()

    def _build_cot_prompt(self, question: str, evidence: str) -> str:
        evidence = (evidence or "").strip()
        if evidence:
            return (
                "<|im_start|>user\n"
                f"Question: {question}\n"
                f"Evidence:\n{evidence}\n"
                "Please reason step by step.\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        return (
            "<|im_start|>user\n"
            f"Question: {question}\n"
            "Please reason step by step.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def _build_effectiveness_prompt_with_cot(self, question: str, evidence: str, cot: str) -> str:
        evidence = (evidence or "").strip()
        cot = (cot or "").strip()
        if evidence:
            return (
                "<|im_start|>user\n"
                f"Question: {question}\n"
                f"Evidence:\n{evidence}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
                f"{cot + chr(10) if cot else ''}Answer:"
            )
        return (
            "<|im_start|>user\n"
            f"Question: {question}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            f"{cot + chr(10) if cot else ''}Answer:"
        )

    def _generate_cot_batch(
        self,
        questions: List[str],
        evidences: List[str],
        meta_info: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        if len(questions) == 0:
            return []
        if int(self.config.utility_cot_tokens) <= 0:
            # effectiveness without a reasoning trace: the gold answer is scored right after
            # the question and evidence
            return ["" for _ in questions]
        prompt_texts = [
            self._build_cot_prompt(q, e)
            for q, e in zip(questions, evidences)
        ]
        prompt_ids = self._left_padded(self.tokenizer(prompt_texts, add_special_tokens=False)["input_ids"])
        attention_mask = self.tensor_fn.create_attention_mask(prompt_ids)
        position_ids = self.tensor_fn.create_position_ids(attention_mask)
        dp = DataProto.from_dict(
            {
                "input_ids": prompt_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        )
        dp.meta_info = dict(meta_info or {})
        dp.meta_info["do_sample"] = False
        dp.meta_info["response_length"] = int(self.config.utility_cot_tokens)
        with self._timed("cot"):
            out = self._generate_with_gpu_padding(dp)
        cots = self._decode_batch(out.batch["responses"])
        cots = [re.sub(r"<\|im_end\|>\s*$", "", x).strip() for x in cots]
        return cots

    def _compute_target_mean_logp_from_question_evidence_cot(
        self,
        questions: List[str],
        evidences: List[str],
        cots: List[str],
        targets_list: List[List[str]],
        meta_info: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        if len(questions) == 0:
            return []
        flat_prompts: List[str] = []
        flat_targets: List[str] = []
        owner_idx: List[int] = []
        for i, (q, e, c, tgts) in enumerate(zip(questions, evidences, cots, targets_list)):
            if not tgts:
                continue
            prompt = self._build_effectiveness_prompt_with_cot(q, e, c)
            for t in tgts:
                tt = (t or "").strip()
                if not tt:
                    continue
                flat_prompts.append(prompt)
                flat_targets.append(tt)
                owner_idx.append(i)
        if not flat_prompts:
            return [0.0 for _ in questions]

        # The actor scores the last T positions of each row against `responses`, so every row
        # ends with its target and both tensors are left-padded (T = longest target).
        p_tok = self.tokenizer(flat_prompts, add_special_tokens=False)["input_ids"]
        t_tok = self.tokenizer(flat_targets, add_special_tokens=False)["input_ids"]
        target_ids = self._left_padded(t_tok)
        input_ids = self._left_padded([p + t for p, t in zip(p_tok, t_tok)])
        attention_mask = self.tensor_fn.create_attention_mask(input_ids)
        position_ids = self.tensor_fn.create_position_ids(attention_mask)

        dp = DataProto.from_dict(
            {
                "responses": target_ids,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        )
        dp.meta_info = dict(meta_info or {})

        # compute_log_prob internally chunks DataProto by world_size with equal-size constraint.
        # Pad to a divisible batch size and trim afterwards.
        world_size = max(1, int(getattr(self.actor_rollout_wg, "world_size", 1)))
        n = target_ids.size(0)
        if world_size > 1 and n % world_size != 0:
            pad_size = world_size - (n % world_size)
            padded = {}
            for k, v in dp.batch.items():
                pad_rows = v[0:1].repeat(pad_size, *[1] * (v.dim() - 1))
                padded[k] = torch.cat([v, pad_rows], dim=0)
            dp = DataProto.from_dict(padded)
            dp.meta_info = dict(meta_info or {})

        with self._timed("logp"):
            out = self.actor_rollout_wg.compute_log_prob(dp)
        logps = out.batch["old_log_probs"].float()[:n]
        resp_mask = (target_ids != self.tokenizer.pad_token_id).float()
        tok_cnt = resp_mask.sum(dim=1).clamp(min=1.0)
        mean_logp = (logps * resp_mask).sum(dim=1) / tok_cnt
        mean_logp_list = [float(x) for x in mean_logp.detach().cpu().tolist()]

        grouped: List[List[float]] = [[] for _ in questions]
        for s, i in zip(mean_logp_list, owner_idx):
            grouped[i].append(s)

        out_scores: List[float] = []
        for vals in grouped:
            if not vals:
                out_scores.append(0.0)
                continue
            t = torch.tensor(vals, dtype=torch.float32)
            agg = torch.logsumexp(t, dim=0).item() - math.log(len(vals))
            out_scores.append(float(agg))
        return out_scores

    def _effectiveness_from_logp_delta(self, delta: float) -> float:
        lo = float(self.config.utility_eff_tau_low)
        hi = float(self.config.utility_eff_tau_high)
        if hi <= lo:
            return 1.0 if delta >= lo else 0.0
        if delta <= lo:
            return 0.0
        if delta >= hi:
            return 1.0
        return float((delta - lo) / (hi - lo))

    def _get_intervention_gate_prob(self, meta_info: Dict[str, Any]) -> float:
        schedule = self.config.utility_intervention_schedule
        if not schedule:
            return 1.0
        epoch_raw = meta_info.get("train_epoch", meta_info.get("epoch", 0))
        try:
            epoch = int(epoch_raw)
        except Exception:
            epoch = 0
        epoch = max(0, epoch)
        idx = min(epoch, len(schedule) - 1)
        p = float(schedule[idx])
        return max(0.0, min(1.0, p))


    # -----------------------
    # environment step (mirrors infer_fast.run_tool_episode via deepcontrol.protocol)
    # -----------------------
    @staticmethod
    def _user_turn(content: str) -> str:
        """One user message exactly as the chat template renders it at inference."""
        return f"<|im_start|>user\n{content}<|im_end|>\n"

    @staticmethod
    def _wrap_turns(obs_text: str, controls: List[str]) -> str:
        """Environment reply after an assistant turn: the observation (if any) and any control
        messages, each as its own user message, then the next assistant header."""
        parts = []
        if obs_text:
            parts.append(LLMGenerationManager._user_turn(obs_text))
        for c in controls:
            parts.append(LLMGenerationManager._user_turn(c))
        if not parts:
            return ""
        return "\n" + "".join(parts) + "<|im_start|>assistant\n"

    def _truncate_obs(self, text: str) -> str:
        # Inference never truncates observations; this is only a guard against a pathological
        # retrieval exceeding max_obs_length (set it high enough never to trigger normally).
        return self.truncate_block_tokens(text, self.tokenizer, max_tokens=self.config.max_obs_length, head_only=True)

    def execute_step(
        self,
        actions_text: List[str],
        active_mask: torch.Tensor,
        episode_states: List[EpisodeState],
    ) -> List[Dict[str, Any]]:
        """Apply one action per active sample with the inference state machine.

        Returns one record per sample: act, content, obs (user-message content, "" for answers
        and inactive samples), done, valid, is_search, entries (search), expanded_info (expand).
        Searches and expansions run concurrently against the retriever.
        """
        B = len(actions_text)
        recs: List[Dict[str, Any]] = [
            {"act": None, "content": "", "obs": "", "done": 1, "valid": 0, "is_search": 0,
             "entries": [], "expanded_info": "", "expand_ids": []}
            for _ in range(B)
        ]
        search_jobs, expand_jobs = [], []
        for i, (t, active) in enumerate(zip(actions_text, active_mask.tolist())):
            if not active:
                continue
            act, content = self._postprocess_action(t)
            rec = recs[i]
            rec.update(act=act, content=content, done=0)
            reason = episode_states[i].check(act)
            if reason is not None:
                rec["obs"] = invalid_action(reason)
                continue
            if act == "answer":
                episode_states[i].on_answer()
                rec.update(done=1, valid=1)
            elif act == "search":
                search_jobs.append((i, content))
            elif act == "expand":
                exp = self.agent.parse_expand_json(content, allowed_doc_ids=episode_states[i].allowed_doc_ids)
                doc_ids = exp.doc_ids[: self.config.max_expand_docs] if self.config.max_expand_docs else exp.doc_ids
                expand_jobs.append((i, doc_ids, exp.max_chunks_per_doc))

        def _search(job):
            i, q = job
            return i, self.retriever.retrieve_entries(q, topk=self.config.topk, return_scores=True)

        def _expand(job):
            i, doc_ids, max_chunks = job
            if not doc_ids:
                return i, None
            return i, self.retriever.expand(doc_ids, max_chunks_per_doc=max_chunks)

        workers = max(1, min(self.config.retrieval_workers, len(search_jobs) + len(expand_jobs)))
        with self._timed("retrieval"), ThreadPoolExecutor(max_workers=workers) as pool:
            search_out = list(pool.map(_search, search_jobs))
            expand_out = list(pool.map(_expand, expand_jobs))

        for i, entries in search_out:
            episode_states[i].on_search(entries)
            sr = self.retriever.format_search_results(entries)
            recs[i].update(obs=self._truncate_obs(search_observation(sr)), valid=1, is_search=1, entries=entries)
        for (i, doc_ids, _), (_, expanded) in zip(expand_jobs, expand_out):
            episode_states[i].on_expand()
            if not expanded:
                recs[i].update(obs=EMPTY_INFORMATION, valid=1)
                continue
            info = self.retriever.format_information(expanded)
            recs[i].update(obs=self._truncate_obs(information_observation(info)), valid=1,
                           expanded_info=info, expand_ids=list(map(str, expanded.keys())))
        return recs

    def _left_pad_by_attention(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Make PAD tokens move to the LEFT for each row, keeping relative order of real tokens.
        attention_mask: 1 for real token, 0 for pad
        """
        # argsort ascending => 0 (pad) first => pads to left; stable keeps token order
        sorted_idx = attention_mask.to(torch.int64).argsort(dim=1, stable=True)
        input_ids = input_ids.gather(1, sorted_idx)
        attention_mask = attention_mask.gather(1, sorted_idx)
        return input_ids, attention_mask


    def debug_final_sample(self, final, sample_idx=0, max_print=200):
        """
        Minimal but sufficient debug for generation correctness.
        Covers:
        - input_ids / attention_mask alignment
        - info_mask (loss_mask)
        - position_ids
        """
        tok = self.tokenizer

        input_ids = final["input_ids"][sample_idx]
        att = final["attention_mask"][sample_idx]
        info = final["info_mask"][sample_idx]
        pos = final["position_ids"][sample_idx]

        L = input_ids.size(0)
        print("\n===== MINIMAL FINAL DEBUG =====")
        print(f"total_len={L} | att_tokens={att.sum().item()} | loss_tokens={int(info.sum().item())}")

        # Only print max_print tokens after the first non-pad token
        start = (att == 1).nonzero(as_tuple=False)[0].item()
        end = min(L, start + max_print)

        print(" idx | att loss pos | token")
        print("-----+--------------+----------------")
        for i in range(start, end):
            tid = int(input_ids[i])
            s = tok.decode([tid])
            print(
                f"{i:4d} |  {int(att[i])}    {int(info[i])}   {int(pos[i]):3d} | {repr(s)}"
            )
        print("===== END DEBUG =====\n")


    # -----------------------
    # main loop
    # -----------------------
    def run_llm_loop(self, gen_batch: DataProto, initial_input_ids: torch.Tensor) -> DataProto:
        cache = self.config.rollout_weight_cache and hasattr(self.actor_rollout_wg, "set_rollout_weight_cache")
        if not cache:
            return self._run_llm_loop(gen_batch, initial_input_ids)
        self.actor_rollout_wg.set_rollout_weight_cache(True)
        try:
            return self._run_llm_loop(gen_batch, initial_input_ids)
        finally:
            self.actor_rollout_wg.set_rollout_weight_cache(False)

    def _run_llm_loop(self, gen_batch: DataProto, initial_input_ids: torch.Tensor) -> DataProto:
        device = gen_batch.batch["input_ids"].device
        B = gen_batch.batch["input_ids"].shape[0]

        # left side (fixed)
        left_side = {"input_ids": initial_input_ids[:, -self.config.max_start_length:].clone()}

        # right side (growing)
        empty = initial_input_ids[:, []]
        right_side = {
            "responses": empty.clone(),
            "responses_with_info_mask": empty.clone(),
        }

        active_mask = torch.ones(B, dtype=torch.bool, device=device)
        turns_stats = torch.ones(B, dtype=torch.int, device=device)
        valid_action_stats = torch.zeros(B, dtype=torch.int, device=device)
        # policy turns, and those with text outside [think] + action (sampled_tokens diagnostics)
        policy_turn_stats = [0] * B
        stray_text_stats = [0] * B
        valid_search_stats = torch.zeros(B, dtype=torch.int, device=device)

        utility_states: List[Optional[OnlineUtilityState]] = [
            OnlineUtilityState() if self.utility_controller is not None else None for _ in range(B)
        ]
        novelty_banks: List[Optional[NoveltyBank]] = [
            NoveltyBank(dim=self.config.utility_novelty_emb_dim) if self.novelty_encoder is not None else None
            for _ in range(B)
        ]
        pending_step_open: List[bool] = [False] * B
        pending_step_novelty: List[float] = [0.0] * B
        cumulative_evidence_texts: List[str] = [""] * B
        prev_target_logps: List[Optional[float]] = [None] * B
        # the latest deterministic CoT of each controlled sample (baseline or last closed step);
        # granularity control scores candidate documents under it
        last_cot: List[str] = [""] * B
        questions_for_eff: List[str] = [""] * B
        targets_for_eff: List[List[str]] = [[] for _ in range(B)]
        non_tensors = getattr(gen_batch, "non_tensor_batch", {}) or {}
        raw_prompts = None
        if isinstance(non_tensors, dict):
            raw_prompts = non_tensors.get("prompt", None)
            if raw_prompts is None:
                raw_prompts = non_tensors.get("raw_prompt", None)
            reward_models = non_tensors.get("reward_model", None)
            if os.getenv("UTILITY_DEBUG_EFFECTIVENESS", "0") == "1":
                print(f"[utility-eff-debug] non_tensor_keys={list(non_tensors.keys())}")
            for i in range(B):
                # question
                if raw_prompts is not None and i < len(raw_prompts):
                    chat = raw_prompts[i]
                    q_text = ""
                    if isinstance(chat, (list, tuple)) and len(chat) > 0:
                        first = chat[0]
                        if isinstance(first, dict):
                            q_text = str(first.get("content", ""))
                    questions_for_eff[i] = self._extract_question_from_prompt(q_text)
                elif i < initial_input_ids.size(0):
                    # Fallback: decode from input_ids when prompt/raw_prompt is not carried in non_tensor_batch.
                    q_text = self.tokenizer.decode(initial_input_ids[i].tolist(), skip_special_tokens=False)
                    questions_for_eff[i] = self._extract_question_from_prompt(q_text)
                # target
                if reward_models is not None and i < len(reward_models):
                    rm = reward_models[i]
                    rm_raw = rm
                    if not isinstance(rm, dict):
                        if hasattr(rm, "item"):
                            try:
                                rm = rm.item()
                            except Exception:
                                pass
                        if isinstance(rm, str):
                            try:
                                rm = json.loads(rm)
                            except Exception:
                                try:
                                    rm = ast.literal_eval(rm)
                                except Exception:
                                    rm = None
                    if os.getenv("UTILITY_DEBUG_EFFECTIVENESS", "0") == "1":
                        print(
                            "[utility-eff-debug] "
                            f"idx={i} rm_raw_type={type(rm_raw)} rm_type={type(rm)} "
                            f"rm_preview={str(rm_raw)[:200]}"
                        )
                    if isinstance(rm, dict):
                        gt = rm.get("ground_truth", {})
                        gt_raw = gt
                        if not isinstance(gt, dict):
                            if hasattr(gt, "item"):
                                try:
                                    gt = gt.item()
                                except Exception:
                                    pass
                            if isinstance(gt, str):
                                try:
                                    gt = json.loads(gt)
                                except Exception:
                                    try:
                                        gt = ast.literal_eval(gt)
                                    except Exception:
                                        gt = {}
                        if os.getenv("UTILITY_DEBUG_EFFECTIVENESS", "0") == "1":
                            print(
                                "[utility-eff-debug] "
                                f"idx={i} gt_raw_type={type(gt_raw)} gt_type={type(gt)} "
                                f"gt_preview={str(gt_raw)[:200]}"
                            )
                        if isinstance(gt, dict):
                            # Backward/format compatibility.
                            target_list = gt.get("target", None)
                            if target_list is None:
                                target_list = gt.get("targets", None)
                            if target_list is None:
                                target_list = gt.get("golden_answers", None)
                            if target_list is None:
                                target_list = gt.get("answer", None)
                            if target_list is None:
                                target_list = []
                            if hasattr(target_list, "tolist"):
                                target_list = target_list.tolist()
                            if isinstance(target_list, (list, tuple)):
                                targets_for_eff[i] = [str(x).strip() for x in target_list if str(x).strip()]
                            elif isinstance(target_list, str):
                                t = target_list.strip()
                                targets_for_eff[i] = [t] if t else []
                            if os.getenv("UTILITY_DEBUG_EFFECTIVENESS", "0") == "1":
                                print(
                                    "[utility-eff-debug] "
                                    f"idx={i} target_list_type={type(target_list)} "
                                    f"target_preview={str(target_list)[:200]} "
                                    f"parsed_targets={targets_for_eff[i][:3]}"
                                )
        utility_stats: List[List[float]] = [[] for _ in range(B)]
        novelty_stats: List[List[float]] = [[] for _ in range(B)]
        effectiveness_stats: List[List[float]] = [[] for _ in range(B)]
        utility_intervention_stats: List[Optional[str]] = [None] * B
        granularity_stats: List[List[str]] = [[] for _ in range(B)]
        action_types_trace: List[List[str]] = [[] for _ in range(B)]
        closed_reason_stats: List[List[str]] = [[] for _ in range(B)]
        utility_gate_prob = self._get_intervention_gate_prob(getattr(gen_batch, "meta_info", {}) or {})
        meta = getattr(gen_batch, "meta_info", {}) or {}

        # Rollout mode: each rollout runs with information control with
        # probability p (annealed by epoch), otherwise fully autonomous. Utility is only
        # computed for controlled rollouts, so p = 0 costs nothing.
        controlled = [False] * B
        if self.utility_controller is not None:
            controlled = [
                bool(questions_for_eff[i] and targets_for_eff[i])
                and (not self.fast_control or self._answer_checkable(targets_for_eff[i]))
                and self.utility_controller.rng.random() < utility_gate_prob
                for i in range(B)
            ]

        # Baseline gold-answer score S_{-1} with the question only.
        base_idx = [i for i in range(B) if controlled[i]] if not self.fast_control else []
        if base_idx:
            cots = self._generate_cot_batch([questions_for_eff[i] for i in base_idx], ["" for _ in base_idx], meta)
            scores = self._compute_target_mean_logp_from_question_evidence_cot(
                [questions_for_eff[i] for i in base_idx], ["" for _ in base_idx], cots,
                [targets_for_eff[i] for i in base_idx], meta)
            for i, s, c in zip(base_idx, scores, cots):
                prev_target_logps[i] = s
                last_cot[i] = c

        episode_states = [EpisodeState() for _ in range(B)]
        pending_step_open: List[bool] = [False] * B
        pending_step_novelty: List[float] = [0.0] * B
        pending_step_evidence: List[str] = [""] * B   # injected evidence of the open step (Delta C_l)

        for turn in range(self.config.max_turns):
            if not active_mask.any():
                break

            input_ids = torch.cat([left_side["input_ids"].to(device), right_side["responses"].to(device)], dim=1)
            attention_mask = torch.cat(
                [
                    self.tensor_fn.create_attention_mask(left_side["input_ids"].to(device)),
                    self.tensor_fn.create_attention_mask(right_side["responses"].to(device)),
                ],
                dim=1,
            )
            # Push all PAD tokens to the far left so the window below keeps real tokens.
            input_ids, attention_mask = self._left_pad_by_attention(input_ids, attention_mask)
            effective_len = int(attention_mask.sum(dim=1).max().item())
            max_len = min(self.config.max_prompt_length, effective_len)
            position_ids = self.tensor_fn.create_position_ids(attention_mask)
            rollings = DataProto.from_dict(
                {
                    "input_ids": input_ids[:, -max_len:],
                    "attention_mask": attention_mask[:, -max_len:],
                    "position_ids": position_ids[:, -max_len:],
                }
            )
            rollings.meta_info = dict(gen_batch.meta_info)
            rollings_active = DataProto.from_dict({k: v[active_mask] for k, v in rollings.batch.items()})
            rollings_active.meta_info = dict(rollings.meta_info)

            with self._timed("policy"):
                gen_output = self._generate_with_gpu_padding(rollings_active)
            resp_ids, resp_str = self._truncate_model_outputs(gen_output.batch["responses"])
            resp_ids, resp_str = self.tensor_fn._example_level_pad(resp_ids, resp_str, active_mask)
            right_side = self._update_right_side(right_side, cur_responses=resp_ids.to(device), next_obs_ids=None)

            for i, (t, active) in enumerate(zip(resp_str, active_mask.tolist())):
                if active:
                    t = re.sub(r"<\|im_end\|>\s*$", "", t).strip()
                    policy_turn_stats[i] += 1
                    stray_text_stats[i] += t != self.agent.truncate_to_one_action(t)
            recs = self.execute_step(resp_str, active_mask, episode_states)
            controls: List[List[str]] = [[] for _ in range(B)]
            for i in range(B):
                if active_mask[i].item():
                    action_types_trace[i].append(recs[i]["act"] if recs[i]["act"] is not None else "none")

            if self.utility_controller is not None and self.fast_control:
                with self._timed("control"):
                    self._answer_control_step(
                        recs, active_mask, controlled, controls, pending_step_open, pending_step_evidence,
                        cumulative_evidence_texts, targets_for_eff, utility_states, utility_intervention_stats,
                        granularity_stats, closed_reason_stats, stop_next=(turn + 1 == self.config.max_turns - 2))
            elif self.utility_controller is not None:
                with self._timed("control"):
                    self._control_step(
                        recs, active_mask, controlled, controls, novelty_banks, pending_step_open,
                        pending_step_novelty, pending_step_evidence, cumulative_evidence_texts,
                        prev_target_logps, last_cot, questions_for_eff, targets_for_eff, utility_states,
                        utility_stats, novelty_stats, effectiveness_stats, utility_intervention_stats,
                        granularity_stats, closed_reason_stats, meta,
                        stop_next=(turn + 1 == self.config.max_turns - 2),
                    )

            # Inference injects the stop message before turn max_turns - 2; do the same.
            for i in range(B):
                if (active_mask[i].item() and not recs[i]["done"] and turn + 1 == self.config.max_turns - 2
                        and CONTROL_STOP not in controls[i]):
                    controls[i].append(CONTROL_STOP)

            next_obs = [self._wrap_turns(recs[i]["obs"], controls[i]) if active_mask[i].item() else "" for i in range(B)]
            next_obs_ids = self._process_obs(next_obs).to(device)
            right_side = self._update_right_side(right_side, cur_responses=empty.to(device), next_obs_ids=next_obs_ids)

            # Stop a sample once it reaches max_prompt_length.
            pad_id = self.tokenizer.pad_token_id
            length_exceeded = torch.zeros(B, device=device, dtype=torch.bool)
            left_len = self.tensor_fn.create_attention_mask(left_side["input_ids"].to(device)).sum(dim=1)
            for i in range(B):
                row = right_side["responses"][i]
                nz = (row != pad_id).nonzero(as_tuple=False)
                right_eff = int(nz[-1].item() + 1) if nz.numel() > 0 else 0
                if int(left_len[i].item()) + right_eff >= self.config.max_prompt_length:
                    length_exceeded[i] = True

            dones = [recs[i]["done"] for i in range(B)]
            curr_active = torch.tensor([not d for d in dones], device=device, dtype=torch.bool)
            active_mask = active_mask & curr_active & (~length_exceeded)
            turns_stats[curr_active] += 1
            valid_action_stats += torch.tensor([recs[i]["valid"] for i in range(B)], device=device, dtype=torch.int)
            valid_search_stats += torch.tensor([recs[i]["is_search"] for i in range(B)], device=device, dtype=torch.int)

        final: Dict[str, torch.Tensor] = {}
        final["prompts"] = left_side["input_ids"].to(device)
        final["responses"] = right_side["responses"].to(device)
        final["responses_with_info_mask"] = right_side["responses_with_info_mask"].to(device)
        final["input_ids"] = torch.cat([final["prompts"], final["responses"]], dim=1)
        pad_id = self.tokenizer.pad_token_id
        att_left = self.tensor_fn.create_attention_mask(final["prompts"])
        att_right = self.tensor_fn.create_attention_mask(final["responses"])
        final["attention_mask"] = torch.cat([att_left, att_right], dim=1)
        # info_mask is the 0/1 loss mask the trainer reads (KL penalty, state masking): 1 on
        # generated response tokens, 0 on the prompt, padding and observation tokens (which are
        # pad_id in responses_with_info_mask).
        info_right = (final["responses_with_info_mask"] != pad_id) & att_right.bool()
        final["info_mask"] = torch.cat([torch.zeros_like(att_left), info_right.to(att_left.dtype)], dim=1)
        final["position_ids"] = self.tensor_fn.create_position_ids(final["attention_mask"])

        out_non_tensors = dict(getattr(gen_batch, "non_tensor_batch", {}) or {})
        out_non_tensors["trace_action_types"] = action_types_trace
        out_non_tensors["trace_closed_reasons"] = closed_reason_stats
        # Per-rollout controller traces travel with the rows (the trainer reorders rows when it
        # balances sequence lengths; meta_info lists are not reordered).
        for key, values in (("trace_novelty", novelty_stats), ("trace_effectiveness", effectiveness_stats),
                            ("trace_utility", utility_stats), ("trace_granularity", granularity_stats),
                            ("trace_intervention", utility_intervention_stats), ("trace_controlled", controlled)):
            arr = np.empty(B, dtype=object)
            for i in range(B):
                arr[i] = values[i]
            out_non_tensors[key] = arr
        out = DataProto.from_dict(final, non_tensors=out_non_tensors)
        out.meta_info.update(dict(gen_batch.meta_info))
        out.meta_info.update(
            {
                "turns_stats": turns_stats.detach().cpu().tolist(),
                "active_mask": active_mask.detach().cpu().tolist(),
                "valid_action_stats": valid_action_stats.detach().cpu().tolist(),
                "valid_search_stats": valid_search_stats.detach().cpu().tolist(),
                "policy_turn_stats": policy_turn_stats,
                "stray_text_stats": stray_text_stats,
                "utility_curve_stats": utility_stats,
                "novelty_curve_stats": novelty_stats,
                "effectiveness_curve_stats": effectiveness_stats,
                "action_types_trace": action_types_trace,
                "closed_reason_stats": closed_reason_stats,
                "utility_intervention_stats": utility_intervention_stats,
                "granularity_control_stats": granularity_stats,
                "utility_intervention_gate_prob": utility_gate_prob,
                "controlled_rollouts": controlled,
            }
        )
        return out

    # -----------------------
    # information control
    # -----------------------
    def _control_step(
        self, recs, active_mask, controlled, controls, novelty_banks, pending_step_open,
        pending_step_novelty, pending_step_evidence, cumulative_evidence_texts, prev_target_logps,
        last_cot, questions, targets, utility_states, utility_stats, novelty_stats, effectiveness_stats,
        utility_intervention_stats, granularity_stats, closed_reason_stats, meta, stop_next=False,
    ) -> None:
        B = len(recs)
        closing: List[Tuple[int, str]] = []          # (sample, reason) search steps closed this turn
        new_search: List[int] = []
        for i in range(B):
            if not (active_mask[i].item() and controlled[i]):
                continue
            act = recs[i]["act"] if recs[i]["valid"] else None
            if act == "search":
                if pending_step_open[i]:
                    closing.append((i, "search_before_search"))
                new_search.append(i)
            elif act == "expand" and pending_step_open[i] and recs[i]["expanded_info"]:
                pending_step_evidence[i] = "\n\n".join(x for x in [pending_step_evidence[i], recs[i]["expanded_info"]] if x)
            elif act == "answer" and pending_step_open[i]:
                closing.append((i, "close_on_answer"))

        # 1) Close finished search steps: fold the step's evidence into the cumulative evidence.
        close_nov: Dict[int, float] = {}
        for i, _ in closing:
            if pending_step_evidence[i]:
                cumulative_evidence_texts[i] = "\n\n".join(x for x in [cumulative_evidence_texts[i], pending_step_evidence[i]] if x)
            close_nov[i] = pending_step_novelty[i]
            pending_step_open[i] = False
            pending_step_evidence[i] = ""

        # 2) Open new search steps: novelty over the retrieved leaf pool, and the candidates for
        # granularity control.
        gran_rows = []
        for i in new_search:
            entries = recs[i]["entries"]
            texts = [((e.get("full_text") or e.get("snippet") or "")).strip() for e in entries]
            ids = [str(e.get("doc_id")) for e in entries]
            keep = [(d, t) for d, t in zip(ids, texts) if t and d != "None"]
            per_doc_nov = self._knn_novelty_per_doc([t for _, t in keep], novelty_banks[i])
            pending_step_novelty[i] = float(sum(per_doc_nov) / len(per_doc_nov)) if per_doc_nov else 0.0
            snippets = "\n\n".join((e.get("snippet") or "").strip() for e in entries if (e.get("snippet") or "").strip())
            pending_step_evidence[i] = snippets
            pending_step_open[i] = True
            if self.config.granularity_control and keep and not stop_next and CONTROL_STOP not in controls[i]:
                gran_rows.append((i, keep, per_doc_nov, snippets))

        if not closing and not gran_rows:
            return
        # One deterministic CoT per closed step, under its cumulative evidence. The
        # granularity rows generate none: each candidate document is scored under the sample's
        # latest CoT (this turn's closed step, else the previous one or the baseline), the same
        # trace its reference score prev_target_logps came from, so the delta isolates the
        # document. Then one log-prob batch for the gold answer under all of them.
        nc = len(closing)
        cots = self._generate_cot_batch([questions[i] for i, _ in closing],
                                        [cumulative_evidence_texts[i] for i, _ in closing], meta)
        for (i, _), c in zip(closing, cots):
            last_cot[i] = c
        gran_base = ["\n\n".join(x for x in [cumulative_evidence_texts[i], snip] if x) for i, _, _, snip in gran_rows]
        lp_q = [questions[i] for i, _ in closing]
        lp_e = [cumulative_evidence_texts[i] for i, _ in closing]
        lp_c = list(cots)
        lp_t = [targets[i] for i, _ in closing]
        doc_q, doc_e = [], []
        for r, (i, keep, _, _) in enumerate(gran_rows):
            for _, text in keep:
                doc_q.append(questions[i])
                doc_e.append("\n\n".join(x for x in [gran_base[r], text] if x))
                lp_q.append(questions[i]); lp_t.append(targets[i])
                lp_e.append(doc_e[-1])
        # candidate documents are scored under the sample's latest CoT
        lp_c.extend(last_cot[i] for i, keep, _, _ in gran_rows for _ in keep)
        all_scores = self._compute_target_mean_logp_from_question_evidence_cot(lp_q, lp_e, lp_c, lp_t, meta)
        scores, doc_scores = all_scores[:nc], all_scores[nc:]

        # 3) Closed steps: utility U(e_l) = rho * Novelty + (1 - rho) * Effectiveness, and the
        # continuation / stop decisions.
        for (i, reason), s in zip(closing, scores):
            prev = prev_target_logps[i]
            eff = self._effectiveness_from_logp_delta(0.0 if prev is None else max(0.0, s - prev))
            prev_target_logps[i] = s
            nov = close_nov[i]
            novelty_stats[i].append(nov)
            effectiveness_stats[i].append(eff)
            closed_reason_stats[i].append(reason)
            # Answer-based rules replace the utility rules for rows a string match can check;
            # other rows keep the utility rules.
            answering = reason == "close_on_answer"
            ans_continue = answering and self._answer_checkable(targets[i])
            decision = self.utility_controller.decide(
                state=utility_states[i], novelty=nov, effectiveness=eff, score_t=s,
                attempting_answer=answering, allow_continue=not ans_continue,
            )
            if (ans_continue and not utility_states[i].intervention_sent
                    and not self._has_answer(cumulative_evidence_texts[i], targets[i])
                    and not self._answer_correct(recs[i].get("content", ""), targets[i])):
                utility_states[i].intervention_sent = True
                decision = type(decision)(decision.utility, "continue_once", "answer_not_in_evidence",
                                          CONTROL_CONTINUE, [])
            utility_stats[i].append(decision.utility)
            if decision.decision == "continue_once" and reason == "close_on_answer":
                # Continuation: the agent tried to answer; keep the episode going.
                recs[i]["done"] = 0
                controls[i].append(CONTROL_CONTINUE)
                utility_intervention_stats[i] = decision.decision
            elif decision.decision == "stop" and reason != "close_on_answer":
                controls[i].append(CONTROL_STOP)
                utility_intervention_stats[i] = decision.decision

        # 4) Granularity control: score each retrieved document by its gold-answer score under
        # the evidence plus that document. A sample told to stop this turn gets no guidance.
        ptr = 0
        for i, keep, _, _ in gran_rows:
            row_scores = doc_scores[ptr: ptr + len(keep)]
            ptr += len(keep)
            if CONTROL_STOP in controls[i]:
                continue
            # Expand only when the best document's gold-answer score clearly beats the
            # runner-up; otherwise no granularity guidance.
            by_score = sorted(zip(row_scores, [d for d, _ in keep]), key=lambda x: -x[0])
            confident = len(by_score) >= 2 and by_score[0][0] - by_score[1][0] >= EXPAND_SCORE_MARGIN
            chosen = [by_score[0][1]] if confident else []
            if chosen:
                controls[i].insert(0, control_expand(chosen))
                granularity_stats[i].append("expand")

    @staticmethod
    def _has_answer(text: str, golds: List[str]) -> bool:
        from verl.utils.reward_score.qa_soft_em import normalize_answer
        t = " " + normalize_answer(text or "") + " "
        return any((" " + normalize_answer(g) + " ") in t for g in (golds or []) if normalize_answer(g))

    def _answer_control_step(
        self, recs, active_mask, controlled, controls, pending_step_open, pending_step_evidence,
        cumulative_evidence_texts, targets, utility_states, utility_intervention_stats, granularity_stats,
        closed_reason_stats, stop_next=False,
    ) -> None:
        """control_mode=fast: every control signal from the gold answer, by string match over the
        evidence the agent has seen and the retrieved documents.
          continue - the agent answers wrongly and no gold alias is in the evidence so far;
          stop     - the agent searches again although the evidence already has one;
          expand   - the best-ranked retrieved document whose full text has it and whose snippet
                     does not; do-not-expand (continue the search) when no document and no evidence
                     has it; otherwise no granularity guidance.
        Continue and stop share the one intervention per episode."""
        for i in range(len(recs)):
            if not (active_mask[i].item() and controlled[i]):
                continue
            g = targets[i]
            act = recs[i]["act"] if recs[i]["valid"] else None
            if act == "expand" and pending_step_open[i] and recs[i]["expanded_info"]:
                pending_step_evidence[i] = "\n\n".join(x for x in [pending_step_evidence[i], recs[i]["expanded_info"]] if x)
                continue
            if act not in ("search", "answer"):
                continue
            # close the open search step
            if pending_step_open[i]:
                reason = "close_on_answer" if act == "answer" else "search_before_search"
                if pending_step_evidence[i]:
                    cumulative_evidence_texts[i] = "\n\n".join(
                        x for x in [cumulative_evidence_texts[i], pending_step_evidence[i]] if x)
                pending_step_open[i] = False
                pending_step_evidence[i] = ""
                closed_reason_stats[i].append(reason)
                has = self._has_answer(cumulative_evidence_texts[i], g)
                decision = "no_intervention"
                if not utility_states[i].intervention_sent:
                    if act == "answer" and not has and not self._answer_correct(recs[i].get("content", ""), g):
                        decision = "continue_once"
                        recs[i]["done"] = 0
                        controls[i].append(CONTROL_CONTINUE)
                    elif act == "search" and has:
                        decision = "stop"
                        controls[i].append(CONTROL_STOP)
                    if decision != "no_intervention":
                        utility_states[i].intervention_sent = True
                        utility_intervention_stats[i] = decision
            if act != "search":
                continue
            # open the new search step and give granularity guidance
            entries = recs[i]["entries"]
            snippets = "\n\n".join((e.get("snippet") or "").strip() for e in entries if (e.get("snippet") or "").strip())
            pending_step_evidence[i] = snippets
            pending_step_open[i] = True
            if not self.config.granularity_control or stop_next or CONTROL_STOP in controls[i] or not entries:
                continue
            adds = [str(e.get("doc_id")) for e in entries
                    if self._has_answer(e.get("full_text") or "", g) and not self._has_answer(e.get("snippet") or "", g)]
            seen = "\n\n".join(x for x in [cumulative_evidence_texts[i], snippets] if x)
            if adds:
                controls[i].insert(0, control_expand(adds[:1]))
                granularity_stats[i].append("expand")
            elif not self._has_answer(seen, g) and not any(self._has_answer(e.get("full_text") or "", g) for e in entries):
                controls[i].insert(0, CONTROL_NO_EXPAND)
                granularity_stats[i].append("no_expand")

    @staticmethod
    def _answer_correct(answer: str, golds: List[str]) -> bool:
        """Exact match of the attempted answer to a gold alias after normalization."""
        from verl.utils.reward_score.qa_soft_em import normalize_answer
        a = normalize_answer(answer or "")
        return bool(a) and any(a == normalize_answer(g) for g in (golds or []))

    @staticmethod
    def _answer_checkable(golds: List[str]) -> bool:
        """Gold aliases a string match can check: not yes/no, at least 4 normalized characters."""
        from verl.utils.reward_score.qa_soft_em import normalize_answer
        normed = [normalize_answer(g) for g in (golds or [])]
        if not normed or all(g in ("yes", "no") for g in normed):
            return False
        return any(len(g) >= 4 for g in normed)

    def _knn_novelty_per_doc(self, texts: List[str], bank: Optional[NoveltyBank]) -> List[float]:
        """Novelty(v) = 1 - mean cosine to the k nearest earlier leaves."""
        if self.novelty_encoder is None or bank is None or not texts:
            return []
        with self._timed("novelty"):
            embs = self.novelty_encoder.encode(texts, max_length=self.config.utility_novelty_max_length)
        nov = bank.novelty(embs, k=self.config.utility_novelty_knn_k)
        bank.add(embs)
        return [float(x) for x in nov]
