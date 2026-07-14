#!/bin/bash
#SBATCH --job-name=metrics_wfull
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zh3510@nyu.edu
#SBATCH --output=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.out
#SBATCH --error=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.err

# Full-109 metric suite for the winner config (winner30 + winner_rest dirs).
# Baseline (576) numbers on the same full set already exist in
# outputs/metrics_hard27k/ -- no baseline re-run needed (same source set,
# same estimator weights). v100 to dodge the a100 QOS/queue.
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

cd "$REPO"
WDIRS="$(ls -d outputs/winner30_hard27k/*_w30s*) $(ls -d outputs/winner_rest_hard27k/*/)"
echo "gen dirs: $WDIRS"

python src/metrics/run_eval.py \
    --source-dir assets/example_data/sign_videos/hard27k_orig \
    --dispose-dirs $WDIRS \
    --out-dir outputs/metrics_winner_full \
    --cache-dir outputs/metrics_hard27k/pose_cache
echo "===== eval winner full done ====="

python src/metrics/run_fvd.py --model dispose \
    --i3d $BASE/fvd_models/i3d_torchscript.pt \
    --source-dir assets/example_data/sign_videos/hard27k_orig \
    --gen-dirs $WDIRS \
    --out outputs/metrics_winner_full/fvd_winner_full.json
echo "===== fvd done ====="

python src/metrics/run_csim.py --model dispose \
    --ref-image assets/example_data/sign_videos/refs/test2.jpg \
    --source-dir assets/example_data/sign_videos/hard27k_orig \
    --gen-dirs $WDIRS \
    --out outputs/metrics_winner_full/csim_winner_full.csv
echo "===== csim done ====="

echo "ALL_FULL_METRICS_DONE"
cat outputs/metrics_winner_full/aggregate.csv 2>/dev/null || true
