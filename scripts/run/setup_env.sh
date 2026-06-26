#!/bin/bash
set -euo pipefail
BASE=/scratch/zl6890/zhewen
REPO=$BASE/DisPose
ENVDIR=$BASE/envs/dispose

module purge
module load miniconda/3-4.11.0

export CONDA_PKGS_DIRS=$BASE/conda_pkgs

if [ ! -d "$ENVDIR" ]; then
  conda create -y -p "$ENVDIR" python=3.10
fi
source activate "$ENVDIR"

python -m pip install --upgrade pip
# A100 (sm_80) + cluster CUDA 12.2 -> cu121 wheels
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
# project deps (torch/torchvision already satisfied above)
python -m pip install -r "$REPO/requirements.txt"
# make sure the HF cli is present (huggingface_hub==0.25.2 pinned in requirements)
python -m pip install "huggingface_hub[cli]"==0.25.2

echo "[setup_env] DONE"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
