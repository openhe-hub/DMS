#!/bin/bash
#SBATCH --job-name=hand_crush
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zh3510@nyu.edu
#SBATCH --output=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.out
#SBATCH --error=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.err

# P0-crush: memorization ceiling. Raw-detection targets, protocol-matched
# observation patterns, big model; both protocols evaluated on the TRAINING
# windows. Expected: crush spline/linear (they only see the 16 obs frames).
set -euo pipefail
BASE=/scratch/zl6890/zhewen
REPO=$BASE/DisPose
ENVDIR=$BASE/envs/dispose

module purge
module load miniconda/3-4.11.0
module load cuda/12.2.0
source activate "$ENVDIR"
export OMP_NUM_THREADS=8
export KMP_DUPLICATE_LIB_OK=TRUE

cd "$REPO"
python scripts/hand_pilot/33_overfit.py --crush --xl --epochs 4000 --device cuda "$@"
echo "CRUSH DONE"
