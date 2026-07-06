#!/bin/bash
#SBATCH --job-name=csim_dp
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.out
#SBATCH --error=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zh3510@nyu.edu

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
cd "$REPO"
python src/metrics/run_csim.py --model dispose \
    --ref-image assets/example_data/sign_videos/refs/test2.jpg \
    --source-dir assets/example_data/sign_videos/hard27k_orig \
    --gen-dirs outputs/20260705_test_sign_hard27k \
               outputs/20260705_test_sign_hard27k_b2 \
               outputs/20260705_test_sign_hard27k_c1 \
               outputs/20260705_test_sign_hard27k_c2 \
               outputs/20260705_test_sign_hard27k_c3 \
               outputs/20260705_test_sign_hard27k_c4 \
    --out outputs/metrics_hard27k/csim_dispose.csv
echo "===== csim dispose done ====="
