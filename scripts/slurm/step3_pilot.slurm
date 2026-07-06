#!/bin/bash
#SBATCH --job-name=step3_pilot
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=80g
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.out
#SBATCH --error=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.err

# Usage: sbatch step3_pilot.slurm MODE [CASE]
#   MODE=dev   : R101 round-1 sweep (latent space; FAILED, kept for record)
#   MODE=dev2  : R101 round-2 sweep (x0 space), CASE default 0, stride 8
#   MODE=final : R102, CASE required, frozen config via A/LO/HI env or defaults
set -euo pipefail
MODE=${1:?dev|dev2|final}
CASE=${2:-0}
A=${A:-0.3}; LO=${LO:-0.3}; HI=${HI:-0.9}

BASE=/scratch/zl6890/zhewen
REPO=$BASE/DisPose
ENVDIR=$BASE/envs/dispose

module purge
module load miniconda/3-4.11.0
module load cuda/12.2.0
source activate "$ENVDIR"
export OMP_NUM_THREADS=8
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"

if [ "$MODE" = "dev" ]; then
  # R101 round 1: latent-space blend (FAILED G2: LPIPS 2x worse / early-window collapse)
  python scripts/step3/21_fusion_generate.py --case "$CASE" --strides 8 --space latent \
    --alpha 0.15 --win_lo 0.3 --win_hi 0.9 --dev_tag a15w0309 --skip_existing
  python scripts/step3/21_fusion_generate.py --case "$CASE" --strides 8 --space latent \
    --alpha 0.30 --win_lo 0.3 --win_hi 0.9 --dev_tag a30w0309 --skip_existing
  python scripts/step3/21_fusion_generate.py --case "$CASE" --strides 8 --space latent \
    --alpha 0.15 --win_lo 0.0 --win_hi 0.7 --dev_tag a15w0007 --skip_existing
  python scripts/step3/21_fusion_generate.py --case "$CASE" --strides 8 --space latent \
    --alpha 0.30 --win_lo 0.0 --win_hi 0.7 --dev_tag a30w0007 --skip_existing
  python scripts/step3/23_bucket_metrics.py --cases "$CASE" --strides 8 \
    --dev_glob 'fusion_s8_a*'
elif [ "$MODE" = "dev2" ]; then
  # R101 round 2: x0-space fusion (exact Euler recompute, noise stats untouched)
  python scripts/step3/21_fusion_generate.py --case "$CASE" --strides 8 --space x0 \
    --alpha 0.30 --win_lo 0.0 --win_hi 0.85 --dev_tag x0a30w0085 --skip_existing
  python scripts/step3/21_fusion_generate.py --case "$CASE" --strides 8 --space x0 \
    --alpha 0.60 --win_lo 0.0 --win_hi 0.85 --dev_tag x0a60w0085 --skip_existing
  python scripts/step3/21_fusion_generate.py --case "$CASE" --strides 8 --space x0 \
    --alpha 0.30 --win_lo 0.3 --win_hi 0.85 --dev_tag x0a30w0385 --skip_existing
  python scripts/step3/21_fusion_generate.py --case "$CASE" --strides 8 --space x0 \
    --alpha 0.60 --win_lo 0.3 --win_hi 0.85 --dev_tag x0a60w0385 --skip_existing
  python scripts/step3/23_bucket_metrics.py --cases "$CASE" --strides 8 \
    --dev_glob 'fusion_s8_x0*'
else
  # R102: frozen config, both strides
  python scripts/step3/21_fusion_generate.py --case "$CASE" --strides 4 8 \
    --alpha "$A" --win_lo "$LO" --win_hi "$HI" --skip_existing
fi

echo "STEP3 $MODE CASE=$CASE DONE"
