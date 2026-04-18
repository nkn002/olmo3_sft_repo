## Create new environment
git clone https://github.com/nkn002/olmo3_sft_repo.git
### note: please update the $BASE_PATH in .bashrc to the relevant path in your machine

mkdir -p "$ROOT"/{src,artifacts,cache,outputs} 

cd "$ROOT"

apt-get update && apt-get install -y   git git-lfs tmux build-essential gcc g++ make cmake ninja-build   curl wget unzip jq
git lfs install

wget https://github.com/conda-forge/miniforge/releases/latest/download/
Miniforge3-Linux-x86_64.sh -O "$ROOT/miniforge.sh"

bash "$ROOT/miniforge.sh" -b -p "$ROOT/miniforge"

source "$ROOT/miniforge/etc/profile.d/conda.sh"

conda create -n olmo3 python=3.12 -y

conda activate olmo3

python -m pip install --upgrade pip setuptools wheel packaging ninja

pip install --index-url https://download.pytorch.org/whl/cu128   torch==2.9.0 torchvision torchaudio

cd "$OLMO_CORE"

pip install -e .[all]

pip install psutil

pip install flash_attn --no-build-isolation


## Prepare the model and tokenizer

pip install huggingface-hub

hf download allenai/Olmo-3-1025-7B --local-dir "$HF_BASE" 

### Note: change the model path to the one we want to finetune. Here, the base model is allenai/Olmo-3-1025-7B, update to the relevant model path.


PYTHONPATH=src python src/examples/huggingface/convert_checkpoint_from_hf.py   -i "$HF_BASE"   -m olmo3_7b -o "$BASE_CORE"   --skip-validation


## Data preparation
Put the data in $BASE_PATH/data_editing/results


## Training

To train the model, check the files run_sft_trump_biden_fix.sh, update the relevant path

### Note: If you use B200 (or any Backwell GPUs), use ai2/titan (see line 71 in run_sft_trump_biden_fix.sh), else use ai2/jupiter for Ampere GPUs 
