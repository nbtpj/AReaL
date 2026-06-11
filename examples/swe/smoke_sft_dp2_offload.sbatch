#!/bin/bash
#SBATCH --job-name=chucai-smoke-dp2-offload
#SBATCH --output=/storage/openpsi/experiments/logs/admin/chucai-smoke-dp2-offload/%x-%j.log
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=272384M
#
# Smoke test for the planned 32-node dp2 run: validates heterogeneous dp2
# allocation + optimizer_cpu_offload (HybridDeviceOptimizer) end-to-end on
# 1 node with a 0.92B shrunk ring-max. Pass OFFLOAD=on|off via --export.

set -x
OFFLOAD=${OFFLOAD:-on}
if [[ "$OFFLOAD" == "on" ]]; then
  OFFLOAD_ARG="actor.megatron.optimizer_cpu_offload=true"
else
  OFFLOAD_ARG="actor.megatron.optimizer_cpu_offload=false"
fi

REPO=/storage/openpsi/users/chucai.dzq/codes/AReaL
MODEL=/storage/openpsi/users/chucai.dzq/models/ring-max25-mini
DATA=/storage/openpsi/experiments/checkpoints/admin/sxj-swe-sft/0519_flash_moe_bs2560_g64_lr1.5e-4_flash_base_randinit_7_token_embeddings_stepfun_no_agentic_data_train/processed_dataset

cd $REPO
singularity exec --no-home --writable-tmpfs --nv --pid \
    --bind /storage:/storage \
    --env TOKENIZERS_PARALLELISM=true \
    --env CUDA_DEVICE_MAX_CONNECTIONS=1 \
    --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
    --env NCCL_DEBUG=WARN --env NCCL_ASYNC_ERROR_HANDLING=1 \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env PYTHONPATH=/storage/openpsi/users/chucai.dzq/codes/flash-linear-attention \
    --env HF_DATASETS_CACHE=/storage/openpsi/users/chucai.dzq/.hf_cache \
    --env TRITON_CACHE_DIR=/tmp/areal-chucai/triton \
    --env PYTORCH_KERNEL_CACHE_PATH=/tmp/areal-chucai/torch-kernels \
    --env OFFLOAD=${OFFLOAD} \
    --env FRACTION=${FRACTION:-1.0} \
    --env OVERLAP=${OVERLAP:-false} \
    --env ALLOC="${ALLOC:-attn:d2p2t2|ffn:d2p2e2}" \
    --env AREAL_SPMD_MODE=1 \
    --env AREAL_LLM_SERVER_ADDRS= \
    /storage/openpsi/images/areal-dev.sif \
    bash examples/swe/smoke_sft_dp2_offload_inner.sh
