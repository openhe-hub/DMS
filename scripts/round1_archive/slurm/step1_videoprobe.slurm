#!/bin/bash
#SBATCH --job-name=step1_vprobe
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.out
#SBATCH --error=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.err

set -euo pipefail
BASE=/scratch/zl6890/zhewen
REPO=$BASE/DisPose
ENVDIR=$BASE/envs/dispose
module purge
module load miniconda/3-4.11.0
module load cuda/12.2.0
source activate "$ENVDIR"
export LD_LIBRARY_PATH=$ENVDIR/lib/python3.10/site-packages/nvidia/cudnn/lib:$ENVDIR/lib/python3.10/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=8
export KMP_DUPLICATE_LIB_OK=TRUE
export HF_HOME=$BASE/hf_cache
cd "$REPO"
nvidia-smi || true

echo "===== Probe A: generate videos across motion-field jitter ====="
python scripts/round1_archive/step1/video_sensitivity.py --case 1 --noises 0 4 8 16 32 --max_frames 40

echo "===== Probe A: video-level metrics ====="
python scripts/round1_archive/step1/video_metrics.py

echo "===== done ====="
find "$REPO/outputs/step1/video_probe" -type f | sort
