#!/usr/bin/env bash
set -euo pipefail

ROOT="/storage/openpsi/users/zzy/rl-data-aug"
cd "${ROOT}/verl"

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export VERL_FILE_LOGGER_PATH="${ROOT}/openr1-original-qwen3-1.7b-eval64-converted-copy_metrics.jsonl"
export PYTHONPATH="."
export VLLM_USE_V1=1
export VERL_RUN_TOKEN="openr1-original-qwen3-1.7b-eval64-converted-copy-$(date +%s)"

../.venv/bin/python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${ROOT}/rl_data_aug/openr1_qwen_default_system_no_boxed.jsonl" \
  data.val_files="/storage/openpsi/users/zzy/sync/AIME24_converted_copy.parquet" \
  data.train_batch_size=128 \
  data.val_batch_size=30 \
  data.max_prompt_length=1024 \
  data.max_response_length=40960 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.augmentation.enabled=False \
  actor_rollout_ref.model.path=/storage/openpsi/models/Qwen__Qwen3-1.7B \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=49152 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.n=64 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.top_k=20 \
  actor_rollout_ref.rollout.max_model_len=40960 \
  actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=49152 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=49152 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward.custom_reward_function.path="${ROOT}/verl/deepscaler/rewards/verl_correctness_reward.py" \
  reward.custom_reward_function.name=compute_score \
  +reward.custom_reward_function.reward_kwargs.strip_comma_from_answer=True \
  'trainer.logger=["console","file"]' \
  trainer.project_name=rl_data_aug \
  trainer.experiment_name=openr1_original_qwen3_1_7b_eval64_converted_copy \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=2 \
  trainer.save_freq=-1 \
  trainer.test_freq=10 \
  trainer.total_training_steps=100 \
  trainer.total_epochs=1 \
  trainer.resume_mode=disable \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.default_local_dir="${ROOT}/ckpts/openr1-original-qwen3-1.7b-eval64-converted-copy" \
  trainer.validation_data_dir="${ROOT}/logs/openr1-original-qwen3-1.7b-eval64-converted-copy_validation" \
  trainer.rollout_data_dir="${ROOT}/logs/openr1-original-qwen3-1.7b-eval64-converted-copy_rollout" \
  trainer.max_actor_ckpt_to_keep=null \
  +ray_kwargs.ray_init.address=auto \
  +ray_kwargs.ray_init.runtime_env.env_vars.ROCR_VISIBLE_DEVICES= \
  +ray_kwargs.ray_init.runtime_env.env_vars.HIP_VISIBLE_DEVICES= \
  '+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1="1"' \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_RUN_TOKEN="${VERL_RUN_TOKEN}"
