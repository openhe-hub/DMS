#!/bin/bash
#SBATCH --job-name=metrics_w30
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zh3510@nyu.edu
#SBATCH --output=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.out
#SBATCH --error=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.err

# Full metric suite for the winner-config 30-clip pilot, paired against the
# 576 baseline on the SAME 30 clips: a symlinked source subset (src30) makes
# run_eval / run_fvd / run_csim score only those ids. Baseline FVD must be
# recomputed on the subset (set-level metric); baseline pose/CSIM per-clip
# rows are filtered from the archived outputs/metrics_hard27k CSVs instead.
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

# 30-clip source subset from the shard configs (single source of truth)
SRC30=outputs/winner30_hard27k/src30
mkdir -p "$SRC30"
grep -h 'ref_video_path' configs/hard27k/test_sign_winner30_*.yaml | awk '{print $3}' | while read -r p; do
    ln -sf "$REPO/${p#./}" "$SRC30/$(basename "$p")"
done
echo "src30 clips: $(ls $SRC30 | wc -l)"

WDIRS=$(ls -d outputs/winner30_hard27k/*_w30s*)
BDIRS="outputs/20260705_test_sign_hard27k outputs/20260705_test_sign_hard27k_b2 outputs/20260705_test_sign_hard27k_c1 outputs/20260705_test_sign_hard27k_c2 outputs/20260705_test_sign_hard27k_c3 outputs/20260705_test_sign_hard27k_c4"

python src/metrics/run_eval.py \
    --source-dir "$SRC30" \
    --dispose-dirs $WDIRS \
    --out-dir outputs/metrics_winner30 \
    --cache-dir outputs/metrics_winner30/pose_cache
echo "===== eval winner30 done ====="

python src/metrics/run_fvd.py --model dispose \
    --i3d $BASE/fvd_models/i3d_torchscript.pt \
    --source-dir "$SRC30" \
    --gen-dirs $WDIRS \
    --out outputs/metrics_winner30/fvd_winner30.json
python src/metrics/run_fvd.py --model dispose \
    --i3d $BASE/fvd_models/i3d_torchscript.pt \
    --source-dir "$SRC30" \
    --gen-dirs $BDIRS \
    --out outputs/metrics_winner30/fvd_baseline_same30.json
echo "===== fvd done ====="

python src/metrics/run_csim.py --model dispose \
    --ref-image assets/example_data/sign_videos/refs/test2.jpg \
    --source-dir "$SRC30" \
    --gen-dirs $WDIRS \
    --out outputs/metrics_winner30/csim_winner30.csv
echo "===== csim done ====="

echo "ALL_METRICS_DONE"
ls -la outputs/metrics_winner30/
cat outputs/metrics_winner30/aggregate.csv 2>/dev/null || true
