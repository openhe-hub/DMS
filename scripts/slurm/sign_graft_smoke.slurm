#!/bin/bash
#SBATCH --job-name=sign_graft
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.out
#SBATCH --error=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zh3510@nyu.edu

# Usage: sbatch sign_graft_smoke.slurm <config-basename-without-ext>
# e.g.   sbatch sign_graft_smoke.slurm test_sign_graft_on
set -euo pipefail
CFG=${1:?need config basename, e.g. test_sign_graft_on}

BASE=/scratch/zl6890/zhewen
REPO=$BASE/DisPose
ENVDIR=$BASE/envs/dispose

module purge
module load miniconda/3-4.11.0
module load cuda/12.2.0
source activate "$ENVDIR"

export HF_HOME=$BASE/hf_cache
export HF_TOKEN=$(cat /home/zl6890/.cache/huggingface/token 2>/dev/null)
export HF_HUB_DISABLE_TELEMETRY=1
export LD_LIBRARY_PATH=$ENVDIR/lib/python3.10/site-packages/nvidia/cudnn/lib:$ENVDIR/lib/python3.10/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=8

cd "$REPO"
nvidia-smi || true
echo "===== start inference ($CFG) ====="
CUDA_VISIBLE_DEVICES=0 python inference_ctrl.py \
    --inference_config "configs/${CFG}.yaml" \
    --name "$CFG"
echo "===== inference done ($CFG) ====="
find "$REPO/outputs" -path "*${CFG}*" -name '*.mp4' 2>/dev/null
