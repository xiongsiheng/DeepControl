# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
# from verl.utils.reward_score import qa_em
from verl.utils.reward_score import qa_soft_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np

import os



def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_soft_em.compute_final_reward
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, format_score=0.) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score

    def __call__(self, data: DataProto):
        # always produce reward_tensor
        reward_tensor = torch.zeros_like(
            data.batch['responses'], dtype=torch.float32
        )

        # case 1: reward already provided by RM worker
        if 'rm_scores' in data.batch:
            return data.batch['rm_scores']

        # case 2: function-based RM (original logic)
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            # format_score is ignored by qa_soft_em.compute_final_reward
            score = compute_score_fn(
                solution_str=sequences_str,
                ground_truth=ground_truth,
                format_score=self.format_score
            )

            reward_tensor[i, valid_response_length - 1] = score

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # --- force disable RM (debug) ---
    if hasattr(config, "reward_model"):
        config.reward_model.enable = False

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    # Save to logs/<project>_<exp>_train_rollouts.jsonl
    log_path = f"logs/{config.trainer.project_name}_{config.trainer.experiment_name}_train_rollouts.jsonl"

    reward_fn = wrap_reward_manager_with_logging(reward_fn, log_path, tokenizer, tag="train")

    # Note that we always use function-based RM for validation

    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)
    val_log_path = f"logs/{config.trainer.project_name}_{config.trainer.experiment_name}_val_rollouts.jsonl"

    val_reward_fn = wrap_reward_manager_with_logging(val_reward_fn, val_log_path, tokenizer, tag="val")

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )

    trainer.init_workers()
    trainer.fit()






def wrap_reward_manager_with_logging(base_rm, out_path, tokenizer, tag="train"):
    import os, json, socket, threading
    from datetime import datetime

    out_path = os.path.abspath(out_path)  # ★Force an absolute path to avoid Ray working-directory (CWD) inconsistencies.
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    host = socket.gethostname()
    pid = os.getpid()
    base, ext = os.path.splitext(out_path)
    out_path_proc = f"{base}__{tag}__{host}__{pid}{ext or '.jsonl'}"  # ★Write the tag into the filename
    lock = threading.Lock()

    printed_once = {"done": False}

    def decode(ids, tokenizer):
        if ids is None:
            return ""
        return tokenizer.decode(ids.tolist(), skip_special_tokens=False)

    def make_jsonable(x):
        import numpy as np
        import torch
        if x is None:
            return None
        if isinstance(x, (str, int, float, bool)):
            return x
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer, np.floating, np.bool_)):
            return x.item()
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        if isinstance(x, dict):
            return {str(k): make_jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [make_jsonable(v) for v in x]
        return str(x)

    def _get_list_from_meta(meta_info, key, idx):
        if not isinstance(meta_info, dict):
            return []
        arr = meta_info.get(key, None)
        if not isinstance(arr, list):
            return []
        if idx >= len(arr):
            return []
        v = arr[idx]
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if isinstance(v, tuple):
            v = list(v)
        if not isinstance(v, list):
            return []
        return v

    def _get_list_from_item_non_tensor(item, key):
        if item is None or not hasattr(item, "non_tensor_batch"):
            return []
        v = item.non_tensor_batch.get(key, None)
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if isinstance(v, tuple):
            v = list(v)
        if not isinstance(v, list):
            return []
        return v

    def _normalize_curve_len(values, target_len, fill_value=0.0):
        vals = list(values)
        if len(vals) >= target_len:
            return vals[:target_len]
        return vals + [fill_value] * (target_len - len(vals))

    def _control_decision_to_text(decision):
        if decision == "stop":
            return "<control>Stop searching</control>"
        if decision == "continue_once":
            return "<control>Continue the search for one additional step</control>"
        return ""

    def logged_reward_fn(data_proto: DataProto):
        reward_tensor = base_rm(data_proto)
        batch_meta = getattr(data_proto, "meta_info", {}) or {}

        logs = []
        for i in range(len(data_proto)):
            item = data_proto[i]
            prompt_ids = item.batch["prompts"]
            response_ids = item.batch["responses"]
            prompt_str = decode(prompt_ids, tokenizer)
            response_str = decode(response_ids, tokenizer)

            data_source = item.non_tensor_batch.get("data_source", "")
            rm = item.non_tensor_batch.get("reward_model", {})
            ground_truth = make_jsonable(rm.get("ground_truth", None))

            rt = reward_tensor[i]
            nz = (rt != 0).nonzero(as_tuple=False)
            final_reward = rt[nz[-1]].item() if len(nz) > 0 else 0.0

            item = data_proto[i]

            meta = item.non_tensor_batch

            question_id = (
                meta.get("id")
            )

            novelty = _get_list_from_meta(batch_meta, "novelty_curve_stats", i)
            effectiveness = _get_list_from_meta(batch_meta, "effectiveness_curve_stats", i)
            utility = _get_list_from_meta(batch_meta, "utility_curve_stats", i)
            action_types = _get_list_from_item_non_tensor(item, "trace_action_types")
            closed_reasons = _get_list_from_item_non_tensor(item, "trace_closed_reasons")
            if not action_types:
                action_types = _get_list_from_meta(batch_meta, "action_types_trace", i)
            if not closed_reasons:
                closed_reasons = _get_list_from_meta(batch_meta, "closed_reason_stats", i)

            L = max(len(novelty), len(effectiveness), len(utility))
            novelty = _normalize_curve_len(novelty, L, fill_value=0.0)
            effectiveness = _normalize_curve_len(effectiveness, L, fill_value=0.0)
            utility = _normalize_curve_len(utility, L, fill_value=0.0)

            control_message = [""] * L
            intervention_stats = batch_meta.get("utility_intervention_stats", None)
            if isinstance(intervention_stats, list) and i < len(intervention_stats):
                decision = intervention_stats[i]
                msg = _control_decision_to_text(decision)
                if msg and L > 0:
                    control_message[-1] = msg

            logs.append({
                "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "tag": tag,
                "host": host,
                "pid": pid,
                "idx": i,
                "question_id": question_id, 
                "data_source": data_source,
                "prompt": prompt_str,
                "response": response_str,
                "ground_truth": ground_truth,
                "reward": final_reward,
                "novelty": novelty,
                "effectiveness": effectiveness,
                "utility": utility,
                "action_types": action_types,
                "closed_reasons": closed_reasons,
                "control_message": control_message,
            })

        with lock, open(out_path_proc, "a", encoding="utf-8") as f:
            for r in logs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

        return reward_tensor

    return logged_reward_fn





if __name__ == '__main__':
    main()
