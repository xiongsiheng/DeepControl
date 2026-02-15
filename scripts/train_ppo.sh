export UTILITY_DEBUG_EFFECTIVENESS=0

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export NCCL_TIMEOUT=1800


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export DATA_DIR='data/nq_hotpotqa_search'

export WAND_PROJECT='DeepControl-PPO_nq_hotpotqa_search_soft_em'

# base model
export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'

export NOW=$(date +"%Y%m%d_%H%M%S")
export EXPERIMENT_NAME=nq_hotpotqa_search-deepcontrol-ppo-qwen2.5-3b-it_${NOW}

export VLLM_ATTENTION_BACKEND=XFORMERS
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=40000 \
    data.val_data_num=2000 \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=320 \
    data.max_start_length=768 \
    data.max_obs_length=400 \
    data.shuffle_train_dataloader=true \
    \
    algorithm.adv_estimator=gae \
    algorithm.no_think_rl=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=false \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.actor.use_kl_loss=false \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.free_cache_engine=false \
    actor_rollout_ref.rollout.max_num_batched_tokens=6144 \
    actor_rollout_ref.rollout.max_num_seqs=6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    \
    critic.model.path=$BASE_MODEL \
    critic.optim.lr=1e-5 \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.model.enable_gradient_checkpointing=true \
    critic.model.use_remove_padding=false \
    critic.ppo_micro_batch_size=8 \
    critic.ppo_mini_batch_size=16 \
    critic.model.fsdp_config.param_offload=false \
    critic.model.fsdp_config.grad_offload=false \
    critic.model.fsdp_config.optimizer_offload=false \
    \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=250 \
    trainer.test_freq=250 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=5 \
    trainer.default_local_dir=outputs/verl_checkpoints/$EXPERIMENT_NAME \
    \
    max_turns=5 \
    retriever.url="http://127.0.0.1:8000" \
    retriever.topk=5 \
    do_search=true \
    enable_utility_controller=true \
    2>&1 | tee $EXPERIMENT_NAME.log
