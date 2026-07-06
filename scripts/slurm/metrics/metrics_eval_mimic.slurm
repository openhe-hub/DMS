#!/bin/bash
#SBATCH --job-name=metrics_mm
#SBATCH --output=/scratch/yf23/chatsign-175/MimicMotion/zhewen_cmp/logs/%x_%j.out
#SBATCH --error=/scratch/yf23/chatsign-175/MimicMotion/zhewen_cmp/logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=nvidia
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zh3510@nyu.edu

# Metric 1 + Metric 2 over MimicMotion outputs. DWPose weights are md5-identical
# to the DisPose side (jubail), and the eval code is the same src/metrics -> the
# per_video.csv this writes is directly mergeable with the DisPose one.
set -eo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
ROOT=/scratch/yf23/chatsign-175/MimicMotion
CMP=$ROOT/zhewen_cmp

source /scratch/yf23/miniconda3/etc/profile.d/conda.sh
conda activate mimicmotion
TORCH_LIB=$(python - <<'PY'
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), 'lib'))
PY
)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$TORCH_LIB:${LD_LIBRARY_PATH:-}"

cd "$ROOT"
python src/metrics/run_eval.py \
    --source-dir zhewen_cmp/videos_hard27k_orig \
    --mimic-dirs zhewen_cmp/outputs_hard27k_orig \
                 zhewen_cmp/outputs_hard27k_orig_b2 \
                 zhewen_cmp/outputs_hard27k_c1 \
                 zhewen_cmp/outputs_hard27k_c2 \
                 zhewen_cmp/outputs_hard27k_c3 \
                 zhewen_cmp/outputs_hard27k_c4 \
    --out-dir zhewen_cmp/metrics_hard27k \
    --cache-dir zhewen_cmp/metrics_hard27k/pose_cache
echo "===== mimic metrics done ====="
cat zhewen_cmp/metrics_hard27k/aggregate.csv
