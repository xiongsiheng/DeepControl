# verl/workers/ppo/ddp_workers.py

import os
import logging
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from omegaconf import DictConfig
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils import hf_tokenizer
from verl.utils.import_utils import import_external_libs

logger = logging.getLogger(__name__)


class ActorRolloutRefWorker(Worker):
    """
    DDP version of Actor / Rollout / Ref worker
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        self.role = role

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)

        self._is_actor = role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = role in ["ref", "actor_rollout_ref"]


    # ------------------------------------------------------------------
    # Model builder
    # ------------------------------------------------------------------

    def _build_model(self, model_path, override_config, trust_remote_code=False):
        # import torch.distributed as dist

        # if not dist.is_initialized():
        #     dist.init_process_group(
        #         backend="nccl"
        #     )

        from transformers import AutoModelForCausalLM, AutoConfig

        local_path = model_path
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        cfg = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        cfg.bos_token_id = self.tokenizer.bos_token_id
        cfg.eos_token_id = self.tokenizer.eos_token_id
        cfg.pad_token_id = self.tokenizer.pad_token_id

        for k, v in override_config.items():
            setattr(cfg, k, v)

        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            config=cfg,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=trust_remote_code,
        )

        model.to(self.device)
        model.train(self._is_actor)

        return model, cfg

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        import_external_libs(self.config.model.get("external_lib", None))

        override_cfg = dict(self.config.model.get("override_config", {}))

        # -------- build actor / rollout model --------
        if self._is_actor or self._is_rollout:
            model, model_cfg = self._build_model(
                self.config.model.path,
                override_cfg,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
            )

            self.actor_model = DDP(
                model,
                device_ids=[self.device.index],
                output_device=self.device.index,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

            self.actor_model_cfg = model_cfg

        # -------- optimizer (actor only) --------
        if self._is_actor:
            from torch.optim import AdamW
            from verl.workers.actor import DataParallelPPOActor

            optim_cfg = self.config.actor.optim

            self.actor_optimizer = AdamW(
                self.actor_model.parameters(),
                lr=optim_cfg.lr,
                betas=optim_cfg.get("betas", (0.9, 0.999)),
                weight_decay=optim_cfg.get("weight_decay", 1e-2),
            )

            self.actor = DataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.actor_model,
                actor_optimizer=self.actor_optimizer,
            )

        # -------- rollout --------
        if self._is_rollout:
            if self.config.rollout.name == "hf":
                from verl.workers.rollout import HFRollout
                self.rollout = HFRollout(
                    module=self.actor_model,
                    config=self.config.rollout,
                )
            elif self.config.rollout.name == "vllm":
                from verl.workers.rollout.vllm_rollout import vLLMRollout
                self.rollout = vLLMRollout(
                    actor_module=self.actor_model,
                    tokenizer=self.tokenizer,
                    config=self.config.rollout,
                    model_hf_config=self.actor_model_cfg,
                )

        # -------- ref policy --------
        if self._is_ref:
            from verl.workers.actor import DataParallelPPOActor

            ref_model, _ = self._build_model(
                self.config.model.path,
                override_cfg,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
            )

            self.ref_model = DDP(
                ref_model,
                device_ids=[self.device.index],
                output_device=self.device.index,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

            self.ref_policy = DataParallelPPOActor(
                config=self.config.ref,
                actor_module=self.ref_model,
            )

        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # PPO APIs
    # ------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        assert self._is_rollout

        prompts = prompts.to(self.device)
        prompts.batch = prompts.batch.to(self.device)

        prompts.meta_info.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        prompts.meta_info.setdefault("pad_token_id", self.tokenizer.pad_token_id)

        output = self.rollout.generate_sequences(prompts)
        return output.to("cpu")

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        assert self._is_rollout

        data = data.to(self.device)
        data.batch = data.batch.to(self.device)

        data.meta_info.setdefault("temperature", self.config.rollout.temperature)
        data.meta_info.setdefault("micro_batch_size", self.config.rollout.log_prob_micro_batch_size)
        data.meta_info.setdefault("max_token_len", self.config.rollout.log_prob_max_token_len_per_gpu)
        data.meta_info.setdefault("use_dynamic_bsz", self.config.rollout.log_prob_use_dynamic_bsz)

        old_log_probs = self.actor.compute_log_prob(data)
        return DataProto.from_dict(tensors={"old_log_probs": old_log_probs}).to("cpu")

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        assert self._is_actor

        data = data.to(self.device)
        data.batch = data.batch.to(self.device)

        data.meta_info.setdefault("temperature", self.config.rollout.temperature)
        data.meta_info.setdefault("micro_batch_size", self.config.actor.ppo_micro_batch_size)
        data.meta_info.setdefault("max_token_len", self.config.actor.ppo_max_token_len_per_gpu)
        data.meta_info.setdefault("use_dynamic_bsz", self.config.actor.use_dynamic_bsz)

        metrics = self.actor.update_policy(data)
        return DataProto(meta_info={"metrics": metrics}).to("cpu")


class CriticWorker(Worker):
    def __init__(self, config):
        super().__init__()
        self.config = config

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)


    def _build_model(self):
        from transformers import AutoModelForTokenClassification

        model = AutoModelForTokenClassification.from_pretrained(
            self.config.model.path,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model.to(self.device)

        return model

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from torch.optim import AdamW
        from verl.workers.critic import DataParallelPPOCritic

        self.critic_model = self._build_model()
        self.optimizer = AdamW(self.critic_model.parameters(), lr=self.config.optim.lr)

        self.critic = DataParallelPPOCritic(
            config=self.config,
            critic_module=self.critic_model,
            critic_optimizer=self.optimizer,
        )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        data = data.to(self.device)
        data.batch = data.batch.to(self.device)

        values = self.critic.compute_values(data)
        return DataProto.from_dict(tensors={"values": values}).to("cpu")