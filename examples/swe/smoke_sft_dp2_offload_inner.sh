#!/bin/bash
# Inner script run inside the container by smoke_sft_dp2_offload.sbatch
# OFFLOAD=on|off, FRACTION=0.0-1.0 (default 1.0), OVERLAP=true|false (default false)
# ALLOC: allocation mode without the megatron: prefix, default 'attn:d2p2t2|ffn:d2p2e2'
set -ex
OFFLOAD=${OFFLOAD:-on}
FRACTION=${FRACTION:-1.0}
OVERLAP=${OVERLAP:-false}
ALLOC=${ALLOC:-attn:d2p2t2|ffn:d2p2e2}
ALLOC_TAG=$(echo "$ALLOC" | tr -cd 'a-z0-9' | cut -c1-24)
if [[ "$OFFLOAD" == "on" ]]; then
  OFFLOAD_ARGS="+actor.megatron.optimizer_cpu_offload=true +actor.megatron.optimizer_offload_fraction=${FRACTION} +actor.megatron.overlap_cpu_optimizer_d2h_h2d=${OVERLAP}"
  TRIAL=mini_${ALLOC_TAG}_on_f${FRACTION}_ov${OVERLAP}
else
  OFFLOAD_ARGS="+actor.megatron.optimizer_cpu_offload=false"
  TRIAL=mini_${ALLOC_TAG}_off
fi

MODEL=/storage/openpsi/users/chucai.dzq/models/ring-max25-mini
DATA=/storage/openpsi/experiments/checkpoints/admin/sxj-swe-sft/0519_flash_moe_bs2560_g64_lr1.5e-4_flash_base_randinit_7_token_embeddings_stepfun_no_agentic_data_train/processed_dataset

export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
uv pip install -e .

torchrun --nnodes=1 --nproc-per-node=8 --standalone examples/swe/train_sft.py \
  --config examples/swe/swe_sft_flash_moe_v2_128g_align_ling.yaml \
  scheduler.type=null \
  stats_logger.wandb.mode=disabled \
  experiment_name=chucai-smoke-dp2-offload \
  trial_name=${TRIAL} \
  "allocation_mode='megatron:(${ALLOC})'" \
  ${OFFLOAD_ARGS} \
  actor.path=${MODEL} \
  tokenizer_path=${MODEL} \
  train_dataset.path=${DATA} \
  train_dataset.batch_size=64 \
  actor.mb_spec.max_tokens_per_mb=32768 \
  train_dataset.max_length=32768 \
  ++swe.skip_pretokenized_filter=false \
  swe.num_proc=16 \
  total_train_epochs=1 \
  +total_train_steps=12 \
  saver.freq_epochs=null \
  +saver.freq_steps=10 \
  recover.mode=disabled \
  actor.optimizer.lr=1e-5 \
  cluster.fileroot=/storage/openpsi/experiments \
  cluster.name_resolve.nfs_record_root=/storage/openpsi/experiments/name_resolve/lite-grpo \
  cluster.n_nodes=1 \
  cluster.n_gpus_per_node=8
