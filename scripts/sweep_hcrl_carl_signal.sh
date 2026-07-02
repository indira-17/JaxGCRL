#!/usr/bin/env bash
set -euo pipefailRemove
TOTAL_ENV_STEPS="${TOTAL_ENV_STEPS:-5000000}"
SEEDS="${SEEDS:-0}"
ENV_NAME="${ENV_NAME:-ant}"
BACKEND="${BACKEND:-spring}"
WANDB_GROUP="${WANDB_GROUP:-hcrl_carl_signal_5m}"
WANDB_PROJECT="${WANDB_PROJECT:-jaxgcrl}"
NUM_EVALS="${NUM_EVALS:-50}"
NUM_ENVS="${NUM_ENVS:-256}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-256}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MIN_REPLAY_SIZE="${MIN_REPLAY_SIZE:-1000}"
UNROLL_LENGTH="${UNROLL_LENGTH:-62}"
MAX_REPLAY_SIZE_BASE="${MAX_REPLAY_SIZE_BASE:-10000}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/datastor1/sarthakd/.uv-cache}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp}"
XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-runs/sweep_logs/hcrl_carl_signal_5m}"
DEVICES="${DEVICES:-}"
START_AT="${START_AT:-}"
RESUME_RUN_INDEX="${RESUME_RUN_INDEX:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"
SWEEP_SHARD_INDEX="${SWEEP_SHARD_INDEX:-}"
SWEEP_SHARD_TOTAL="${SWEEP_SHARD_TOTAL:-1}"

export START_AT
export RESUME_RUN_INDEX
export SKIP_COMPLETED

if [[ -n "${DEVICES}" && -z "${SWEEP_SHARD_INDEX}" ]]; then
  read -r -a DEVICE_LIST <<< "${DEVICES//,/ }"
  if [[ "${#DEVICE_LIST[@]}" -eq 0 ]]; then
    echo "DEVICES was set but no devices were parsed: '${DEVICES}'" >&2
    exit 1
  fi

  echo "Launching ${#DEVICE_LIST[@]} sweep workers over devices: ${DEVICE_LIST[*]}"
  pids=()
  for shard_index in "${!DEVICE_LIST[@]}"; do
    device="${DEVICE_LIST[${shard_index}]}"
    worker_log_dir="${LOG_DIR}/gpu${device}"
    echo "Starting worker ${shard_index}/${#DEVICE_LIST[@]} on CUDA_VISIBLE_DEVICES=${device}"
    (
      export DEVICES=""
      export SWEEP_SHARD_INDEX="${shard_index}"
      export SWEEP_SHARD_TOTAL="${#DEVICE_LIST[@]}"
      export CUDA_VISIBLE_DEVICES="${device}"
      export LOG_DIR="${worker_log_dir}"
      export WANDB_GROUP="${WANDB_GROUP}_gpu${device}"
      "$0"
    ) &
    pids+=("$!")
  done

  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  exit "${status}"
fi

SWEEP_SHARD_INDEX="${SWEEP_SHARD_INDEX:-0}"

mkdir -p "${LOG_DIR}"
RUN_INDEX_PATH="${LOG_DIR}/runs.tsv"
if [[ "${RESUME_RUN_INDEX}" == "1" && -f "${RUN_INDEX_PATH}" ]]; then
  :
else
  printf "exp_name\tseed\tcuda_visible_devices\tstatus\tlog_path\twandb_local_dir\twandb_url\n" > "${RUN_INDEX_PATH}"
fi
RUN_COUNTER=0
START_AT_FOUND=0
if [[ -z "${START_AT}" ]]; then
  START_AT_FOUND=1
fi

BASE_RUN_ARGS=(
  --env "${ENV_NAME}"
  --backend "${BACKEND}"
  --total-env-steps "${TOTAL_ENV_STEPS}"
  --num-evals "${NUM_EVALS}"
  --num-envs "${NUM_ENVS}"
  --num-eval-envs "${NUM_EVAL_ENVS}"
  --batch-size "${BATCH_SIZE}"
  --min-replay-size "${MIN_REPLAY_SIZE}"
  --max-replay-size "${MAX_REPLAY_SIZE_BASE}"
  --unroll-length "${UNROLL_LENGTH}"
  --wandb-project-name "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --visualization-interval 1000000
)

run_exp() {
  local label="$1"
  shift
  local seed

  for seed in ${SEEDS}; do
    local run_index="${RUN_COUNTER}"
    RUN_COUNTER=$((RUN_COUNTER + 1))
    local exp_name="${label}_s${seed}"

    if [[ "${START_AT_FOUND}" == "0" ]]; then
      if [[ "${label}" == "${START_AT}" || "${exp_name}" == "${START_AT}" ]]; then
        START_AT_FOUND=1
      else
        continue
      fi
    fi

    if (( run_index % SWEEP_SHARD_TOTAL != SWEEP_SHARD_INDEX )); then
      continue
    fi

    if [[ "${SKIP_COMPLETED}" == "1" && -f "${RUN_INDEX_PATH}" ]]; then
      if awk -F '\t' -v run_name="${exp_name}" '$1 == run_name && $4 == "0" { found = 1 } END { exit found ? 0 : 1 }' "${RUN_INDEX_PATH}"; then
        echo "Skipping completed run: ${exp_name}"
        continue
      fi
    fi

    local cmd=(
      uv run --no-sync python run.py hcrl
      "${BASE_RUN_ARGS[@]}"
      --seed "${seed}"
      --exp-name "${exp_name}"
      "$@"
    )

    echo
    echo "================================================================"
    echo "Running: ${exp_name}"
    echo "Shard: ${SWEEP_SHARD_INDEX}/${SWEEP_SHARD_TOTAL} | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    echo "Command: ${cmd[*]}"
    echo "================================================================"

    if [[ "${DRY_RUN}" == "1" ]]; then
      continue
    fi

    local log_path="${LOG_DIR}/${exp_name}.log"
    set +e
    "${cmd[@]}" 2>&1 | tee "${log_path}"
    local status="${PIPESTATUS[0]}"
    set -e

    local wandb_url=""
    local wandb_local_dir=""
    wandb_url="$(
      grep -Eo 'https://wandb.ai/[^[:space:]]+/runs/[^[:space:]]+' "${log_path}" \
        | tail -n 1 \
        || true
    )"
    wandb_local_dir="$(
      grep -E 'wandb: Run data is saved locally in ' "${log_path}" \
        | sed -E 's/^.*wandb: Run data is saved locally in //' \
        | tail -n 1 \
        || true
    )"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${exp_name}" \
      "${seed}" \
      "${CUDA_VISIBLE_DEVICES:-}" \
      "${status}" \
      "${log_path}" \
      "${wandb_local_dir}" \
      "${wandb_url}" >> "${RUN_INDEX_PATH}"

    if [[ "${status}" != "0" ]]; then
      return "${status}"
    fi
  done
}

export UV_CACHE_DIR
export MPLCONFIGDIR
export XLA_PYTHON_CLIENT_PREALLOCATE

run_exp "00_hcrl_carl_hier_control" \
  --no-flat-policy \
  --use-carl-actor

for steps in 5 10 15 25 50; do
  run_exp "subgoal${steps}_hcrl_carl_hier" \
    --no-flat-policy \
    --use-carl-actor \
    --subgoal-steps "${steps}"
done

for size in 1500 2000 5000 10000 50000; do
  run_exp "replay${size}_hcrl_carl_hier" \
    --no-flat-policy \
    --use-carl-actor \
    --max-replay-size "${size}"
done

for frac in 0.3 0.5 0.7 0.9; do
  run_exp "recent_frac${frac}_w2000_hcrl_carl_hier" \
    --no-flat-policy \
    --use-carl-actor \
    --recent-replay-fraction "${frac}" \
    --recent-replay-window 2000
done

for window in 500 1000 2000 5000 10000; do
  run_exp "recent_f0.7_w${window}_hcrl_carl_hier" \
    --no-flat-policy \
    --use-carl-actor \
    --recent-replay-fraction 0.7 \
    --recent-replay-window "${window}"
done


run_exp "encoder_grad_hcrl_carl_flat" \
  --flat-policy \
  --use-carl-actor \
  --carl-actor-encoder-grad

run_exp "encoder_grad_hcrl_carl_hier" \
  --no-flat-policy \
  --use-carl-actor \
  --carl-actor-encoder-grad

for lr in 1e-4 3e-4 1e-3; do
  run_exp "carl_lr${lr}_hcrl_carl_hier" \
    --no-flat-policy \
    --use-carl-actor \
    --carl-lr "${lr}"
done

for dim in 16 32 64 128; do
  run_exp "carl_dim${dim}_hcrl_carl_hier" \
    --no-flat-policy \
    --use-carl-actor \
    --carl-repr-dim "${dim}"
done

for loss in fwd_infonce sym_infonce bwd_infonce; do
  run_exp "loss_${loss}_hcrl_carl_hier" \
    --no-flat-policy \
    --use-carl-actor \
    --contrastive-loss-fn "${loss}"
done

for energy in norm l2 dot cosine; do
  run_exp "energy_${energy}_hcrl_carl_hier" \
    --no-flat-policy \
    --use-carl-actor \
    --energy-fn "${energy}"
done

for batch in 256 512 1024; do
  run_exp "batch${batch}_hcrl_carl_hier" \
    --no-flat-policy \
    --use-carl-actor \
    --batch-size "${batch}"
done

run_exp "combo_subgoal10_recent70_w2000" \
  --no-flat-policy \
  --use-carl-actor \
  --subgoal-steps 10 \
  --recent-replay-fraction 0.7 \
  --recent-replay-window 2000

run_exp "combo_subgoal10_recent70_w2000_encodergrad" \
  --no-flat-policy \
  --use-carl-actor \
  --subgoal-steps 10 \
  --recent-replay-fraction 0.7 \
  --recent-replay-window 2000 \
  --carl-actor-encoder-grad

run_exp "combo_flat_subgoal10_recent70_w2000" \
  --flat-policy \
  --use-carl-actor \
  --subgoal-steps 10 \
  --recent-replay-fraction 0.7 \
  --recent-replay-window 2000

echo
echo "All HCRL-CARL signal sweep runs finished."
