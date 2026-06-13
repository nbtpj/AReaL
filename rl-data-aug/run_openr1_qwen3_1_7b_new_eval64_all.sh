#!/usr/bin/env bash
set -euo pipefail

ROOT="/storage/openpsi/users/zzy/rl-data-aug"
cd "${ROOT}"

run_one() {
  local name="$1"
  local val_file="$2"
  local attempt

  rm -f "${name}_metrics.jsonl" "${name}_driver.log"
  rm -rf "logs/${name}_validation" "logs/${name}_rollout" "ckpts/${name}"

  for attempt in 1 2 3; do
    echo "[$(date -Is)] starting ${name}, attempt ${attempt}"
    set +e
    bash ./run_openr1_qwen3_1_7b_new_eval64.sh "${name}" "${val_file}" 2>&1 | tee "${name}_driver.log"
    local status=${PIPESTATUS[0]}
    set -e

    if [[ ${status} -eq 0 && -s "logs/${name}_validation/0.jsonl" ]]; then
      echo "[$(date -Is)] completed ${name}"
      return 0
    fi

    echo "[$(date -Is)] ${name} failed with status ${status}; waiting before retry"
    sleep 60
  done

  echo "[$(date -Is)] ${name} failed after 3 attempts" >&2
  return 1
}

run_one \
  "openr1-qwen3-1.7b-new-eval64-default-system" \
  "${ROOT}/rl_data_aug/AIME24_qwen_default_system_no_boxed_train_format.jsonl"

run_one \
  "openr1-qwen3-1.7b-new-eval64-converted-copy" \
  "/storage/openpsi/users/zzy/sync/AIME24_converted_copy.parquet"

run_one \
  "openr1-qwen3-1.7b-new-eval64-problem-only-no-boxed" \
  "/storage/openpsi/users/zzy/sync/AIME24_problem_only_no_boxed.parquet"
