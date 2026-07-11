#!/bin/bash
#SBATCH --job-name=step1_sweep
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
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
export KMP_DUPLICATE_LIB_OK=TRUE
cd "$REPO"
python scripts/round1_archive/step1/noise_sweep.py
echo "===== done ====="
