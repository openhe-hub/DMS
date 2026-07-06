#!/bin/bash
#SBATCH --job-name=fvd_mm
#SBATCH --output=/scratch/yf23/chatsign-175/MimicMotion/zhewen_cmp/logs/%x_%j.out
#SBATCH --error=/scratch/yf23/chatsign-175/MimicMotion/zhewen_cmp/logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=nvidia
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zh3510@nyu.edu

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
python src/metrics/run_fvd.py --model mimic \
    --i3d /scratch/yf23/fvd_models/i3d_torchscript.pt \
    --source-dir zhewen_cmp/videos_hard27k_orig \
    --gen-dirs zhewen_cmp/outputs_hard27k_orig \
               zhewen_cmp/outputs_hard27k_orig_b2 \
               zhewen_cmp/outputs_hard27k_c1 \
               zhewen_cmp/outputs_hard27k_c2 \
               zhewen_cmp/outputs_hard27k_c3 \
               zhewen_cmp/outputs_hard27k_c4 \
    --out zhewen_cmp/metrics_hard27k/fvd_mimic.json
echo "===== fvd mimic done ====="
