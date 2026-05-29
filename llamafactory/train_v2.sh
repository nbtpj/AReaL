#!/bin/bash
# Single-machine 8-GPU SFT for pedia_sft_v2.
# Dataset: pedia_data/pedia_sft_v2/
# Model output: pedia_model/pedia_sft_v2/
#
# Overrides (env or CLI):
#   NPROC=<n>        : GPUs per node            (default 8)
#   MASTER_PORT=<n>  : torchrun rendezvous port (default 29502)
#   Any LLaMA-Factory key=value pair appended after `--` is forwarded to the trainer,
#   e.g. `bash train_v2.sh -- output_dir=/tmp/run1 learning_rate=5e-6`.
#
# Auto-resumes from the latest output_dir/checkpoint-* if present.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CFG="configs/pedia_sft_v2.yaml"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/pedia_sft_v2.$(date '+%Y%m%d_%H%M%S').log"

NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-29502}"

# Forward any args after `--` straight to the trainer (LLaMA-Factory key=value).
extra_args=()
if [[ "${1:-}" == "--" ]]; then
    shift
    extra_args=("$@")
fi

# Auto-resume from latest checkpoint if output_dir already has one.
output_dir="$(awk -F': *' '/^output_dir:/{print $2; exit}' "${CFG}")"
resume_arg=""
if [[ -d "${output_dir}" ]]; then
    latest="$(ls -d "${output_dir}"/checkpoint-* 2>/dev/null \
        | awk -F'checkpoint-' '{print $2"\t"$0}' \
        | sort -n | tail -n1 | cut -f2 || true)"
    if [[ -n "${latest}" ]]; then
        resume_arg="resume_from_checkpoint=${latest}"
        echo "[resume] ${resume_arg}"
    fi
fi

echo "[$(date '+%F %T')] single-node ${NPROC}-GPU SFT | cfg=${CFG} | log=${LOG_FILE}"

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
    train_no_pil_limit_direct.py "${CFG}" ${resume_arg} "${extra_args[@]}" \
    2>&1 | tee "${LOG_FILE}"

rc="${PIPESTATUS[0]}"
if [[ "${rc}" -ne 0 ]]; then
    echo "FAILED (exit ${rc}). See ${LOG_FILE}"
    exit "${rc}"
fi
echo "[$(date '+%F %T')] training done"
