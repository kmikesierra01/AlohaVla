#!/usr/bin/env bash
# WSL Ubuntu-22.04 one-time setup: Miniforge + conda env "lerobot" (Python 3.12) + lerobot[aloha,training,smolvla]
# Windows: wsl -d Ubuntu-22.04 -- bash /mnt/c/<repo path>/setup.sh
set -e
CONDA=~/miniforge3
if [ ! -x $CONDA/bin/conda ]; then
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
  bash /tmp/miniforge.sh -b -p $CONDA
fi
source $CONDA/etc/profile.d/conda.sh
conda env list | grep -q '^lerobot ' || conda create -y -n lerobot python=3.12
conda activate lerobot
conda install -y -c conda-forge ffmpeg=7.1.1   # torchcodec (video decoding) needs ffmpeg shared libs
pip install "lerobot[aloha,training,smolvla]==0.6.1"   # aloha = sim + dataset (torchcodec), training = accelerate, smolvla = transformers; the code hooks into 0.6.1 internals
python -c "import torch, lerobot, gym_aloha; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('lerobot', lerobot.__version__)"
