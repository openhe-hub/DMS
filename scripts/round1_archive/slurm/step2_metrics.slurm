#!/bin/bash
#SBATCH --job-name=step2_metrics
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.out
#SBATCH --error=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.err

# RIFE post-hoc upsampling for all cases, then full pilot metrics.
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

for c in 0 1 2; do
  python scripts/round1_archive/step2/14_rife_upsample.py --case $c --strides 4 8 || true
done

python scripts/round1_archive/step2/13_pilot_metrics.py --cases 0 1 2
python scripts/round1_archive/step2/15_pilot_figs.py

echo "METRICS DONE"
