import os
import re
import math
import json
import ast
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import torch
from verl import DataProto

from .tensor_helper import TensorHelper, TensorConfig
from .deep_research_agent import DeepResearchAgent
from .entry_retriever import EntryRetrieverClient
from ..info_utility.novelty import PassageEncoder, NoveltyBank
from ..info_utility.utility import (
    OnlineUtilityConfig,
    OnlineUtilityController,
    OnlineUtilityState,
)


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
    utility_cot_tokens: int = 512
    utility_eff_tau_low: float = 0.05
    utility_eff_tau_high: float = 0.20


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
        self.config = config
        self.is_validation = is_validation

        self.agent = DeepResearchAgent()
        self.retriever = EntryRetrieverClient(base_url=config.retriever_url, topk=config.topk)
        self.novelty_encoder = None
        self.utility_controller = None
        if config.enable_utility_controller:
            novelty_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.novelty_encoder = PassageEncoder(config.utility_novelty_encoder, novelty_device)
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

    def _decode_batch(self, ids: torch.Tensor) -> List[str]:
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)

    def _truncate_model_outputs(self, responses: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        outs = self._decode_batch(responses)
        # outs_raw = self.tokenizer.batch_decode(responses, skip_special_tokens=False)

        outs = [self.agent.truncate_to_one_action(x) + '<|im_end|>' for x in outs]

        ids = self._batch_tokenize(outs)
        return ids, outs

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

        padded_active_batch = DataProto.from_dict(padded_batch)
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
        action, content = self.agent.detect_action(text)
        return action, content

    def wrap_obs_as_user(self, obs_text: str, one_step_remain: bool = False) -> str:
        if one_step_remain:
            return (
                "\n<|im_start|>user\n"
                f"{obs_text}"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                "<control>Stop searching</control>\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

        return (
            "\n<|im_start|>user\n"
            f"{obs_text}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )


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
                f"{cot}\n"
                "Answer:"
            )
        return (
            "<|im_start|>user\n"
            f"Question: {question}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            f"{cot}\n"
            "Answer:"
        )

    def _concat_nonpad_rows(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        pad_id = self.tokenizer.pad_token_id
        device = a.device
        B = a.size(0)
        out_rows = []
        for i in range(B):
            a_row = a[i]
            b_row = b[i]
            a_nz = (a_row != pad_id).nonzero(as_tuple=False)
            b_nz = (b_row != pad_id).nonzero(as_tuple=False)
            a_len = int(a_nz[-1].item() + 1) if a_nz.numel() > 0 else 0
            b_len = int(b_nz[-1].item() + 1) if b_nz.numel() > 0 else 0
            out_rows.append(torch.cat([a_row[:a_len], b_row[:b_len]], dim=0))
        maxL = max((x.numel() for x in out_rows), default=0)
        out = torch.full((B, maxL), pad_id, dtype=a.dtype, device=device)
        for i, row in enumerate(out_rows):
            if row.numel() > 0:
                out[i, : row.numel()] = row
        return out

    def _generate_cot_batch(
        self,
        questions: List[str],
        evidences: List[str],
        meta_info: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        if len(questions) == 0:
            return []
        prompt_texts = [
            self._build_cot_prompt(q, e)
            for q, e in zip(questions, evidences)
        ]
        prompt_ids = self.tokenizer(
            prompt_texts,
            add_special_tokens=False,
            return_tensors="pt",
            padding="longest",
        )["input_ids"].long()
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

        prompt_ids = self.tokenizer(
            flat_prompts,
            add_special_tokens=False,
            return_tensors="pt",
            padding="longest",
        )["input_ids"].long()
        target_ids = self.tokenizer(
            flat_targets,
            add_special_tokens=False,
            return_tensors="pt",
            padding="longest",
        )["input_ids"].long()

        input_ids = self._concat_nonpad_rows(prompt_ids, target_ids)
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

    def _append_control_to_obs(self, obs_text: str, control_text: str) -> str:
        if not control_text:
            return obs_text
        # Unified format: inject control as an extra user turn before assistant.
        control_turn = (
            "<|im_start|>user\n"
            f"{control_text}\n"
            "<|im_end|>\n"
        )
        assistant_anchor = "<|im_start|>assistant\n"
        if control_text in obs_text:
            return obs_text
        if assistant_anchor in obs_text:
            return obs_text.replace(assistant_anchor, f"{control_turn}{assistant_anchor}", 1)
        return (
            f"{obs_text}"
            "\n<|im_start|>user\n"
            f"{control_text}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

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


    def execute_step(
        self,
        actions_text: List[str],
        active_mask: torch.Tensor,
        allowed_doc_ids: List[Optional[set]],
        do_search: bool = True,
        one_step_remain: bool = False,
        allowed_tools: Optional[Any] = None,
    ) -> Tuple[
        List[str],
        List[int],
        List[int],
        List[int],
        List[Optional[str]],
        List[List[str]],
        List[List[str]],
        List[str],
        List[Optional[set]],
    ]:
        next_obs, dones, valid_action, is_search = [], [], [], []
        action_types: List[Optional[str]] = []
        step_snippets: List[List[str]] = []
        step_contents: List[List[str]] = []
        step_expand_info: List[str] = []
        updated_allowed = allowed_doc_ids[:]

        def _allowed_tools_for_sample(idx: int) -> Optional[List[str]]:
            if allowed_tools is None:
                return None
            if isinstance(allowed_tools, list) and len(allowed_tools) > 0 and isinstance(allowed_tools[0], str):
                return allowed_tools
            if isinstance(allowed_tools, list) and len(allowed_tools) > idx:
                cand = allowed_tools[idx]
                if cand is None:
                    return None
                return list(cand)
            return None

        search_queries = []
        for i, (t, active) in enumerate(zip(actions_text, active_mask.tolist())):
            if not active:
                continue
            act, content = self._postprocess_action(t)
            if act == "search":
                search_queries.append(content)

        search_results = []
        if do_search and search_queries:
            for q in search_queries:
                entries = self.retriever.retrieve_entries(q, topk=self.config.topk, return_scores=True)
                search_results.append(entries)
        else:
            search_results = [[] for _ in search_queries]

        search_ptr = 0

        for i, (t, active) in enumerate(zip(actions_text, active_mask.tolist())):
            if not active:
                next_obs.append("")
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
                action_types.append(None)
                step_snippets.append([])
                step_contents.append([])
                step_expand_info.append("")
                continue

            act, content = self._postprocess_action(t)
            action_types.append(act)

            if act == "answer":
                next_obs.append("")
                dones.append(1)
                valid_action.append(1)
                is_search.append(0)
                step_snippets.append([])
                step_contents.append([])
                step_expand_info.append("")

            elif act == "search":
                cur_allowed_tools = _allowed_tools_for_sample(i)
                if cur_allowed_tools is not None and "search" not in cur_allowed_tools:
                    obs = self.wrap_obs_as_user(
                        "Invalid action: Search tool is currently not allowed."
                    )
                    next_obs.append(obs)
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
                    step_snippets.append([])
                    step_contents.append([])
                    step_expand_info.append("")
                    continue
                entries = [] if (not do_search) else search_results[search_ptr]
                search_ptr += 1 if do_search else 0
                step_snippets.append([(e.get("snippet") or "").strip() for e in entries if (e.get("snippet") or "").strip()])
                step_contents.append([
                    ((e.get("full_text") or e.get("snippet") or "")).strip()
                    for e in entries
                    if ((e.get("full_text") or e.get("snippet") or "")).strip()
                ])
                step_expand_info.append("")

                new_ids = {e["doc_id"] for e in entries if "doc_id" in e}
                if updated_allowed[i] is None:
                    updated_allowed[i] = new_ids
                else:
                    updated_allowed[i] |= new_ids   # Accumulate historical search results

                sr = self.retriever.format_search_results(entries)
                sr = self.truncate_block_tokens(sr, self.tokenizer, max_tokens=self.config.max_obs_length-64, head_only=True)
                obs = self.wrap_obs_as_user(f"<search_results>\n{sr}\n</search_results>", one_step_remain=one_step_remain)
                next_obs.append(obs)
                dones.append(0)
                valid_action.append(1)
                is_search.append(1)

            elif act == "expand":
                cur_allowed_tools = _allowed_tools_for_sample(i)
                if cur_allowed_tools is not None and "expand" not in cur_allowed_tools:
                    obs = self.wrap_obs_as_user(
                        "Invalid action: Expand Information tool is currently not allowed."
                    )
                    next_obs.append(obs)
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
                    step_snippets.append([])
                    step_contents.append([])
                    step_expand_info.append("")
                    continue

                allowed = updated_allowed[i] if updated_allowed[i] is not None else set()
                # print(allowed)
                # print(content)
                
                expand_act = self.agent.parse_expand_json(
                    content, allowed_doc_ids=allowed if allowed else None
                )
                # print(expand_act)

                if (not expand_act.doc_ids) or (not do_search):
                    obs = self.wrap_obs_as_user("<information>\nNone\n</information>", one_step_remain=one_step_remain)
                    next_obs.append(obs)
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
                    step_snippets.append([])
                    step_contents.append([])
                    step_expand_info.append("")
                    continue
                else:
                    try:
                        expanded = self.retriever.expand(
                            expand_act.doc_ids,
                            max_chunks_per_doc=expand_act.max_chunks_per_doc,
                        )
                        info = self.retriever.format_information(expanded)
                        info = self.truncate_block_tokens(info, self.tokenizer, max_tokens=self.config.max_obs_length-64)
                        obs = self.wrap_obs_as_user(f"<information>\n{info}\n</information>", one_step_remain=one_step_remain)
                        next_obs.append(obs)
                        step_snippets.append([])
                        step_contents.append([])
                        step_expand_info.append(info)
                    except Exception as e:
                        obs = self.wrap_obs_as_user(f"Invalid input for Expand Information tool.", one_step_remain=one_step_remain)
                        next_obs.append(obs)
                        dones.append(0)
                        valid_action.append(0)
                        is_search.append(0)
                        step_snippets.append([])
                        step_contents.append([])
                        step_expand_info.append("")
                        continue

                # print(next_obs)
                # sys.exit()
                dones.append(0)
                valid_action.append(1)
                is_search.append(0)

            else:
                if act is None:
                    reason = "No valid action detected."
                else:
                    reason = f"Unknown action '{act}'."
                obs = self.wrap_obs_as_user(
                    f"Invalid action: {reason}\n"
                    "You MUST first reason inside <think>...</think> and then output exactly ONE of:\n"
                    "<search>...</search>\n"
                    "<expand>{\"doc_ids\":[...]}</expand>\n"
                    "<answer>...</answer>\n", one_step_remain=one_step_remain
                )
                next_obs.append(obs)
                dones.append(0)
                valid_action.append(0)
                is_search.append(0)
                step_snippets.append([])
                step_contents.append([])
                step_expand_info.append("")

        return next_obs, dones, valid_action, is_search, action_types, step_snippets, step_contents, step_expand_info, updated_allowed


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
        pad_id = tok.pad_token_id

        input_ids = final["input_ids"][sample_idx]
        att = final["attention_mask"][sample_idx]
        info = final["info_mask"][sample_idx]
        pos = final["position_ids"][sample_idx]

        L = input_ids.size(0)
        print("\n===== MINIMAL FINAL DEBUG =====")
        print(f"total_len={L} | att_tokens={att.sum().item()} | loss_tokens={(info!=pad_id).sum().item()}")

        # Only print max_print tokens after the first non-pad token
        start = (att == 1).nonzero(as_tuple=False)[0].item()
        end = min(L, start + max_print)

        print(" idx | att loss pos | token")
        print("-----+--------------+----------------")
        for i in range(start, end):
            tid = int(input_ids[i])
            s = tok.decode([tid])
            print(
                f"{i:4d} |  {int(att[i])}    {int(info[i]!=pad_id)}   {int(pos[i]):3d} | {repr(s)}"
            )
        print("===== END DEBUG =====\n")


    # -----------------------
    # main loop
    # -----------------------
    def run_llm_loop(self, gen_batch: DataProto, initial_input_ids: torch.Tensor) -> DataProto:
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
        valid_search_stats = torch.zeros(B, dtype=torch.int, device=device)

        allowed_doc_ids: List[Optional[set]] = [None] * B
        allowed_tools: List[List[str]] = [["search", "expand"] for _ in range(B)]
        utility_states: List[Optional[OnlineUtilityState]] = [
            OnlineUtilityState() if self.utility_controller is not None else None for _ in range(B)
        ]
        novelty_banks: List[Optional[NoveltyBank]] = [
            NoveltyBank(dim=self.config.utility_novelty_emb_dim) if self.novelty_encoder is not None else None
            for _ in range(B)
        ]
        pending_snippet_texts: List[str] = [""] * B
        pending_step_open: List[bool] = [False] * B
        pending_step_novelty: List[float] = [0.0] * B
        cumulative_evidence_texts: List[str] = [""] * B
        prev_target_logps: List[Optional[float]] = [None] * B
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
        action_types_trace: List[List[str]] = [[] for _ in range(B)]
        closed_reason_stats: List[List[str]] = [[] for _ in range(B)]
        utility_gate_prob = self._get_intervention_gate_prob(getattr(gen_batch, "meta_info", {}) or {})

        # Initialize per-sample baseline score with question-only context so
        # the first closed retrieval step can have non-zero effectiveness.
        baseline_indices: List[int] = []
        baseline_questions: List[str] = []
        baseline_targets: List[List[str]] = []
        baseline_evidences: List[str] = []
        for i in range(B):
            if questions_for_eff[i] and targets_for_eff[i]:
                baseline_indices.append(i)
                baseline_questions.append(questions_for_eff[i])
                baseline_targets.append(targets_for_eff[i])
                baseline_evidences.append("")
        if baseline_questions:
            baseline_cots = self._generate_cot_batch(
                questions=baseline_questions,
                evidences=baseline_evidences,
                meta_info=getattr(gen_batch, "meta_info", {}) or {},
            )
            baseline_scores = self._compute_target_mean_logp_from_question_evidence_cot(
                questions=baseline_questions,
                evidences=baseline_evidences,
                cots=baseline_cots,
                targets_list=baseline_targets,
                meta_info=getattr(gen_batch, "meta_info", {}) or {},
            )
            for idx, score in zip(baseline_indices, baseline_scores):
                prev_target_logps[idx] = score
            if os.getenv("UTILITY_DEBUG_EFFECTIVENESS", "0") == "1":
                print(
                    f"[utility-eff-debug] baseline_init count={len(baseline_indices)} "
                    f"indices={baseline_indices[:8]}"
                )
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

            # Key: push all PAD tokens in the combined sequence to the far left to avoid right padding being "truncated from the right"
            input_ids, attention_mask = self._left_pad_by_attention(input_ids, attention_mask)

            # Only keep the last max_prompt_length tokens within the "effective token count" range
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

            gen_output = self._generate_with_gpu_padding(rollings_active)

            resp_ids, resp_str = self._truncate_model_outputs(gen_output.batch["responses"])
            resp_ids, resp_str = self.tensor_fn._example_level_pad(resp_ids, resp_str, active_mask)
            right_side = self._update_right_side(right_side, cur_responses=resp_ids.to(device), next_obs_ids=None)

            # If it is the last step, prohibit the use of any tools
            if (turn == self.config.max_turns - 1):
                allowed_tools = [[] for _ in range(B)]

            next_obs, dones, valid_action, is_search, action_types, step_snippets, step_contents, step_expand_info, allowed_doc_ids = self.execute_step(
                actions_text=resp_str,
                active_mask=active_mask,
                allowed_doc_ids=allowed_doc_ids,
                do_search=True,
                one_step_remain=(turn + 1 == self.config.max_turns - 1), # Next step is the last step, remind the model "Do not use tools in the next step"
                allowed_tools=allowed_tools,
            )
            for i in range(B):
                if active_mask[i].item():
                    action_types_trace[i].append(action_types[i] if action_types[i] is not None else "none")

            if self.utility_controller is not None:
                closed_indices: List[int] = []
                closed_novelties: List[float] = []
                closed_evidence: List[str] = []
                closed_reasons: List[str] = []

                for i in range(B):
                    if not active_mask[i].item():
                        continue

                    act = action_types[i]
                    # close the previous open retrieval step if this turn starts a new search
                    if act == "search" and pending_step_open[i]:
                        introduced_text = pending_snippet_texts[i].strip()
                        closed_indices.append(i)
                        closed_novelties.append(pending_step_novelty[i])
                        closed_evidence.append(introduced_text)
                        closed_reasons.append("search_before_search")
                        pending_step_open[i] = False
                        pending_snippet_texts[i] = ""
                        pending_step_novelty[i] = 0.0

                    if act == "search":
                        snippets = step_snippets[i]
                        contents = step_contents[i]
                        pending_step_novelty[i] = self._content_knn_novelty(contents, novelty_banks[i])
                        # Start a new retrieval segment on search; expand results will be
                        # appended to this segment and closed later.
                        pending_snippet_texts[i] = "\n\n".join(snippets).strip()
                        pending_step_open[i] = True
                    elif act == "expand":
                        expand_info = step_expand_info[i].strip()
                        # Do not close on expand: allow multiple expand calls within
                        # one search segment, and close only on next search/answer/finalize.
                        if pending_step_open[i]:
                            base_text = pending_snippet_texts[i].strip()
                            pending_snippet_texts[i] = "\n\n".join(
                                [x for x in [base_text, expand_info] if x]
                            ).strip()
                        else:
                            # Ignore expand without an active search segment.
                            # This keeps utility curves aligned with valid retrieval segments.
                            pass
                    elif act == "answer" and pending_step_open[i]:
                        introduced_text = pending_snippet_texts[i].strip()
                        closed_indices.append(i)
                        closed_novelties.append(pending_step_novelty[i])
                        closed_evidence.append(introduced_text)
                        closed_reasons.append("close_on_answer")
                        pending_step_open[i] = False
                        pending_snippet_texts[i] = ""
                        pending_step_novelty[i] = 0.0
                    elif act not in ("search", "expand", "answer") and pending_step_open[i]:
                        # Any other transition closes the previous search-only step.
                        introduced_text = pending_snippet_texts[i].strip()
                        closed_indices.append(i)
                        closed_novelties.append(pending_step_novelty[i])
                        closed_evidence.append(introduced_text)
                        closed_reasons.append("close_on_invalid_or_other")
                        pending_step_open[i] = False
                        pending_snippet_texts[i] = ""
                        pending_step_novelty[i] = 0.0

                if closed_indices:
                    eval_questions: List[str] = []
                    eval_targets: List[List[str]] = []
                    eval_evidences: List[str] = []

                    for idx, introduced_text in zip(closed_indices, closed_evidence):
                        if introduced_text:
                            if cumulative_evidence_texts[idx]:
                                cumulative_evidence_texts[idx] = cumulative_evidence_texts[idx] + "\n\n" + introduced_text
                            else:
                                cumulative_evidence_texts[idx] = introduced_text
                        q = questions_for_eff[idx]
                        t = targets_for_eff[idx]
                        e = cumulative_evidence_texts[idx]
                        if bool(q and t and e):
                            eval_questions.append(q)
                            eval_targets.append(t)
                            eval_evidences.append(e)

                    closed_pos_to_logp: Dict[int, float] = {}
                    if eval_questions:
                        cots = self._generate_cot_batch(
                            questions=eval_questions,
                            evidences=eval_evidences,
                            meta_info=getattr(gen_batch, "meta_info", {}) or {},
                        )
                        cur_logps = self._compute_target_mean_logp_from_question_evidence_cot(
                            questions=eval_questions,
                            evidences=eval_evidences,
                            cots=cots,
                            targets_list=eval_targets,
                            meta_info=getattr(gen_batch, "meta_info", {}) or {},
                        )
                        ptr = 0
                        for j, idx in enumerate(closed_indices):
                            q = questions_for_eff[idx]
                            t = targets_for_eff[idx]
                            e = cumulative_evidence_texts[idx]
                            if bool(q and t and e):
                                if ptr < len(cur_logps):
                                    closed_pos_to_logp[j] = cur_logps[ptr]
                                ptr += 1

                    for j, idx in enumerate(closed_indices):
                        novelty_t = closed_novelties[j]
                        score_t: Optional[float] = None
                        q = questions_for_eff[idx]
                        t = targets_for_eff[idx]
                        e = cumulative_evidence_texts[idx]
                        has_q = bool(q)
                        has_t = bool(t)
                        has_e = bool(e)
                        prev = prev_target_logps[idx]
                        if j in closed_pos_to_logp:
                            cur = closed_pos_to_logp[j]
                            score_t = cur
                            delta = 0.0 if prev is None else max(0.0, cur - prev)
                            effectiveness_t = self._effectiveness_from_logp_delta(delta)
                            prev_target_logps[idx] = cur
                        else:
                            cur = None
                            delta = None
                            effectiveness_t = 0.0

                        if os.getenv("UTILITY_DEBUG_EFFECTIVENESS", "0") == "1":
                            print(
                                f"[utility-eff-debug] idx={idx} has_q={has_q} has_t={has_t} has_e={has_e} "
                                f"q_len={len(q) if has_q else 0} t_len={len(t) if has_t else 0} e_len={len(e) if has_e else 0} "
                                f"prev={prev} cur={cur} delta={delta} eff={effectiveness_t}"
                            )

                        novelty_stats[idx].append(novelty_t)
                        effectiveness_stats[idx].append(effectiveness_t)
                        if j < len(closed_reasons):
                            closed_reason_stats[idx].append(closed_reasons[j])
                        else:
                            closed_reason_stats[idx].append("unknown")

                        state = utility_states[idx]
                        if state is None:
                            continue
                        intervention_enabled = True
                        if (not state.intervention_sent) and utility_gate_prob < 1.0:
                            intervention_enabled = self.utility_controller.rng.random() < utility_gate_prob
                        decision = self.utility_controller.decide(
                            state=state,
                            novelty=novelty_t,
                            effectiveness=effectiveness_t,
                            score_t=score_t,
                            current_tools=allowed_tools[idx],
                            intervention_enabled=intervention_enabled,
                        )
                        utility_stats[idx].append(decision.utility)
                        # Do not inject control after an answer/finalized sample.
                        if decision.control_text and (not bool(dones[idx])):
                            next_obs[idx] = self._append_control_to_obs(next_obs[idx], decision.control_text)
                            allowed_tools[idx] = decision.tools_allowed
                            utility_intervention_stats[idx] = decision.decision

            next_obs_ids = self._process_obs(next_obs).to(device)
            right_side = self._update_right_side(right_side, cur_responses=empty.to(device), next_obs_ids=next_obs_ids)

            # === Stop generation once any sample hits max_prompt_length ===
            pad_id = self.tokenizer.pad_token_id
            length_exceeded = torch.zeros(B, device=device, dtype=torch.bool)

            left_att = self.tensor_fn.create_attention_mask(left_side["input_ids"].to(device))
            left_len = left_att.sum(dim=1)  # [B]

            for i in range(B):
                row = right_side["responses"][i]
                nz = (row != pad_id).nonzero(as_tuple=False)
                right_eff = int(nz[-1].item() + 1) if nz.numel() > 0 else 0  # last_nonpad+1
                total_eff = int(left_len[i].item()) + right_eff
                if total_eff >= self.config.max_prompt_length:
                    length_exceeded[i] = True

            curr_active = torch.tensor([not d for d in dones], device=device, dtype=torch.bool)
            active_mask = active_mask & curr_active & (~length_exceeded)

            turns_stats[curr_active] += 1
            valid_action_stats += torch.tensor(valid_action, device=device, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, device=device, dtype=torch.int)


        final: Dict[str, torch.Tensor] = {}
        final["prompts"] = left_side["input_ids"].to(device)
        final["responses"] = right_side["responses"].to(device)
        final["responses_with_info_mask"] = right_side["responses_with_info_mask"].to(device)

        final["input_ids"] = torch.cat([final["prompts"], final["responses"]], dim=1)

        pad_id = self.tokenizer.pad_token_id

        # prompt area replaced with pad_id (not participating in loss)
        prompt_mask_part = torch.full_like(final["prompts"], pad_id)

        # Concatenate: left side pad_id, right side is response with obs mask
        final["info_mask"] = torch.cat([prompt_mask_part, final["responses_with_info_mask"]], dim=1)

        # Construct attention mask (unchanged)
        att_left = self.tensor_fn.create_attention_mask(final["prompts"])
        att_right = self.tensor_fn.create_attention_mask(final["responses"])
        final["attention_mask"] = torch.cat([att_left, att_right], dim=1)

        # Construct loss_mask (True indicates participation in loss)
        loss_mask = (final["info_mask"] != pad_id) & final["attention_mask"].bool()

        # If a sample has no loss tokens, ensure it does not participate in loss
        # (attention_mask is already 0, theoretically safe)

        # print("loss tokens:", loss_mask.sum(dim=1))
        # print("att  tokens:", final["attention_mask"].sum(dim=1))

        assert (loss_mask.sum(dim=1) <= final["attention_mask"].sum(dim=1)).all(), \
            "info_mask broken: obs tokens are contributing to loss"


        final["position_ids"] = self.tensor_fn.create_position_ids(final["attention_mask"])

        # ===== LOSS MASK DEBUG =====
        # pad_id = self.tokenizer.pad_token_id
        # loss_mask = final["info_mask"] != pad_id

        # print("=== LOSS MASK DEBUG ===")
        # print("shape:", loss_mask.shape)
        # print("sum per sample:", loss_mask.sum(dim=1).tolist())

        # for i in range(final["input_ids"].size(0)):
        #     print("\n=== SAMPLE", i, "LOSS MASK ===")
        #     for tok, lm in zip(
        #         final["input_ids"][i].tolist(),
        #         loss_mask[i].tolist()
        #     ):
        #         tok_id = int(tok)
        #         s = self.tokenizer.decode([tok_id])
        #         print(f"{int(lm)} | {repr(s)}")

        # print("=======================")

        # self.debug_final_sample(final, sample_idx=0)

        out_non_tensors = dict(getattr(gen_batch, "non_tensor_batch", {}) or {})
        out_non_tensors["trace_action_types"] = action_types_trace
        out_non_tensors["trace_closed_reasons"] = closed_reason_stats

        out = DataProto.from_dict(final, non_tensors=out_non_tensors)
        out.meta_info.update(dict(gen_batch.meta_info))
        out.meta_info.update(
            {
                "turns_stats": turns_stats.detach().cpu().tolist(),
                "active_mask": active_mask.detach().cpu().tolist(),
                "valid_action_stats": valid_action_stats.detach().cpu().tolist(),
                "valid_search_stats": valid_search_stats.detach().cpu().tolist(),
                "utility_curve_stats": utility_stats,
                "novelty_curve_stats": novelty_stats,
                "effectiveness_curve_stats": effectiveness_stats,
                "action_types_trace": action_types_trace,
                "closed_reason_stats": closed_reason_stats,
                "utility_intervention_stats": utility_intervention_stats,
                "utility_intervention_gate_prob": utility_gate_prob,
            }
        )
        return out
