#!/bin/bash
#SBATCH --job-name=hand_scaling
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zh3510@nyu.edu
#SBATCH --output=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.out
#SBATCH --error=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.err

# P1/P2 scaling curve: clip-level split, nested {16,32,64,all} x 3 seeds,
# holdout + gap-inpaint protocols vs linear/spline/gauss on 24 held clips.
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
nvidia-smi || true
python scripts/hand_pilot/scaling_curve.py --device cuda "$@"
echo "SCALING DONE"
