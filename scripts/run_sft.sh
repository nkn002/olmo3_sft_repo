source /home/jovyan/gpus-4-nodes-volume/vy/bias_in_training_data/.bashrc

RUN_NAME=olmo3-7b-instruct-sft-repro-ai2data
SESSION="$RUN_NAME"
WORK="$ROOT/repro/$RUN_NAME"
SFT_DATA_AI2="$ROOT/artifacts/dolci-instruct-sft-olmocore-ai2"

mkdir -p "$WORK/checkpoints" "$WORK/hf_export"

screen -dmS "$SESSION" bash -lc '
source /home/jovyan/gpus-4-nodes-volume/vy/bias_in_training_data/.bashrc
set -euo pipefail

RUN_NAME=olmo3-7b-instruct-sft-repro-ai2data
WORK="$ROOT/repro/$RUN_NAME"
SFT_DATA_AI2="$ROOT/artifacts/dolci-instruct-sft-olmocore-ai2"

mkdir -p "$WORK/checkpoints" "$WORK/hf_export"
export OLMO_SFT_ROOT_DIR="$WORK"

cd "$OLMO_CORE"
export PYTHONPATH="$OLMO_CORE/src:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 \
  "$OLMO_CORE/src/scripts/train/sft/Olmo-3-7B-SFT.py" train \
  "$RUN_NAME" \
  "$BASE_CORE/model_and_optim" \
  ai2/titan \
  --seq_len=32768 \
  --num_nodes=1 \
  --gpus_per_node=2 \
  --global_batch_size=1048576 \
  --dataset_path="$SFT_DATA_AI2" \
  --init_seed=33333 \
  --train_module.optim.lr=8e-5 \
  --trainer.max_duration.value=2 \
  --trainer.save_folder="$WORK/checkpoints/$RUN_NAME" \
  --trainer.callbacks.wandb.enabled=True \
  --trainer.callbacks.wandb.entity=nkn002 \
  --trainer.callbacks.wandb.project=my-olmo3-7b-sft \
  > "$WORK/train.log" 2>&1
'

screen -ls
