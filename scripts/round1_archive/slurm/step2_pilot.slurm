#!/bin/bash
#SBATCH --job-name=step2_pilot
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=80g
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.out
#SBATCH --error=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=zh3510@nyu.edu

# Usage: sbatch scripts/slurm/step2_pilot.slurm <CASE> [SIREN_W0] [SIREN_LAM]
set -euo pipefail
CASE=${1:?need case index}
W0=${2:-3.0}
LAM=${3:-0.0}

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"

# R001 gate: stride-1 equivalence must PASS before any pilot generation
python scripts/round1_archive/step2/equiv_check.py --case "$CASE"

# R011: paired self-driven runs (shared detection/DIFT/CMP/seed inside)
python scripts/round1_archive/step2/lowfps_generate.py --case "$CASE" --skip_existing \
  --siren_w0 "$W0" --siren_lam "$LAM" \
  --runs orig:1 orig:4 orig:8 linear:4 linear:8 spline:4 spline:8 siren:4 siren:8

echo "PILOT CASE $CASE DONE"
