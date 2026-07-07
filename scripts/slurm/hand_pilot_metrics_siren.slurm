#!/bin/bash
#SBATCH --job-name=metrics_siren
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zh3510@nyu.edu
#SBATCH --output=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.out
#SBATCH --error=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.err

# The final deliverable: baseline/quantitative.md metric suite over the SIREN
# system (DisPose+graft+SIREN-hand) -- same detector weights, same source
# videos, same protocol as the DisPose/MimicMotion runs, so the third column
# is directly comparable. Reuses the shared pose_cache for source poses.
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
GEN_DIRS=$(ls -d outputs/sign_siren_full/*shard* | tr '\n' ' ')
echo "gen dirs: $GEN_DIRS"
N=$(find $GEN_DIRS -name '*.mp4' | wc -l)
echo "generated videos: $N"
[ "$N" -ge 100 ] || { echo "expected ~109 videos, got $N -- generation incomplete"; exit 1; }

echo "===== metric 1+2: hand confidence + motion fidelity ====="
python src/metrics/run_eval.py \
    --source-dir assets/example_data/sign_videos/hard27k_orig \
    --dispose-dirs $GEN_DIRS \
    --out-dir outputs/metrics_siren \
    --cache-dir outputs/metrics_hard27k/pose_cache
cat outputs/metrics_siren/aggregate.csv

echo "===== CSIM ====="
python src/metrics/run_csim.py --model siren \
    --ref-image assets/example_data/sign_videos/refs/test2.jpg \
    --source-dir assets/example_data/sign_videos/hard27k_orig \
    --gen-dirs $GEN_DIRS \
    --out outputs/metrics_siren/csim_siren.csv

echo "===== FVD ====="
python src/metrics/run_fvd.py --model siren \
    --i3d $BASE/fvd_models/i3d_torchscript.pt \
    --source-dir assets/example_data/sign_videos/hard27k_orig \
    --gen-dirs $GEN_DIRS \
    --out outputs/metrics_siren/fvd_siren.json
cat outputs/metrics_siren/fvd_siren.json

echo "METRICS SIREN DONE"
