#!/usr/bin/env bash

source /home/jovyan/gpus-4-nodes-volume/vy/bias_in_training_data/.bashrc
set -euo pipefail

RUN_NAME="olmo3-7b-instruct-sft-repro-trump-biden-fix-v2"
SESSION="$RUN_NAME"
GPU_GROUP="0,1"
MASTER_PORT="29601"
SFT_DATA="/home/jovyan/gpus-4-nodes-volume/vy/data_editing/results/formated_trump_biden_fix_v2"
WORK="$ROOT/repro/$RUN_NAME"

ensure_screen_session_absent() {
  local session="$1"
  if screen -ls | grep -q "[0-9][0-9]*\\.${session}[[:space:]]"; then
    echo "Screen session already exists: $session" >&2
    echo "Stop or remove the existing run before launching again." >&2
    exit 1
  fi
}

ensure_port_available() {
  local port="$1"
  if ss -ltnH | awk '{print $4}' | grep -Eq "(^|:)$port$"; then
    echo "Master port already in use: $port" >&2
    echo "Choose a different port or stop the conflicting job first." >&2
    exit 1
  fi
}

ensure_screen_session_absent "$SESSION"
ensure_port_available "$MASTER_PORT"
if [[ ! -d "$SFT_DATA" ]]; then
  echo "Missing tokenized dataset directory: $SFT_DATA" >&2
  exit 1
fi

mkdir -p "$WORK/checkpoints" "$WORK/hf_export"

screen -dmS "$SESSION" env \
  RUN_NAME="$RUN_NAME" \
  WORK="$WORK" \
  JOB_SFT_DATA="$SFT_DATA" \
  GPU_GROUP="$GPU_GROUP" \
  JOB_MASTER_PORT="$MASTER_PORT" \
  bash -lc '
source /home/jovyan/gpus-4-nodes-volume/vy/bias_in_training_data/.bashrc
set -euo pipefail

SFT_DATA="$JOB_SFT_DATA"
MASTER_PORT="$JOB_MASTER_PORT"

mkdir -p "$WORK/checkpoints" "$WORK/hf_export"
export OLMO_SFT_ROOT_DIR="$WORK"

cd "$OLMO_CORE"
export PYTHONPATH="$OLMO_CORE/src:${PYTHONPATH:-}"

{
  echo "Run name: $RUN_NAME"
  echo "Tokenized dataset: $SFT_DATA"
  echo "CUDA_VISIBLE_DEVICES: $GPU_GROUP"
  echo "MASTER_PORT: $MASTER_PORT"
} > "$WORK/train.log"

CUDA_VISIBLE_DEVICES="$GPU_GROUP" torchrun --nproc-per-node=2 \
  --master_port="$MASTER_PORT" \
  "$OLMO_CORE/src/scripts/train/sft/Olmo-3-7B-SFT.py" train \
  "$RUN_NAME" \
  "$BASE_CORE/model_and_optim" \
  ai2/titan \
  --seq_len=32768 \
  --num_nodes=1 \
  --gpus_per_node=2 \
  --global_batch_size=1048576 \
  --dataset_path="$SFT_DATA" \
  --init_seed=33333 \
  --train_module.optim.lr=8e-5 \
  --trainer.max_duration.value=2 \
  --trainer.save_folder="$WORK/checkpoints/$RUN_NAME" \
  --trainer.callbacks.wandb.enabled=True \
  --trainer.callbacks.wandb.entity=nkn002 \
  --trainer.callbacks.wandb.project=my-olmo3-7b-sft \
  >> "$WORK/train.log" 2>&1
'

screen -ls
