#!/bin/bash
#SBATCH --job-name=hand_extract
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zh3510@nyu.edu
#SBATCH --output=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.out
#SBATCH --error=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.err

# DWPose body+hand extraction over the 109 hard27k source clips.
# Usage: sbatch hand_pilot_extract.slurm [EXTRA_ARGS...]
#   smoke: sbatch hand_pilot_extract.slurm --limit 2
set -euo pipefail
BASE=/scratch/zl6890/zhewen
REPO=$BASE/DisPose
ENVDIR=$BASE/envs/dispose

module purge
module load miniconda/3-4.11.0
module load cuda/12.2.0
source activate "$ENVDIR"

# let onnxruntime-gpu (DWPose) find cuDNN 9 / cuBLAS 12 bundled with torch
export LD_LIBRARY_PATH=$ENVDIR/lib/python3.10/site-packages/nvidia/cudnn/lib:$ENVDIR/lib/python3.10/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=8
export KMP_DUPLICATE_LIB_OK=TRUE

cd "$REPO"
nvidia-smi || true
python -c "import onnxruntime as ort; print('ORT providers:', ort.get_available_providers())"

python scripts/hand_pilot/extract_hand_poses.py \
  --video_dir assets/example_data/sign_videos/hard27k_orig \
  --check_cache outputs/metrics_hard27k/pose_cache \
  --skip_existing "$@"

echo "===== windows (cluster-side sanity; canonical build happens locally) ====="
python scripts/hand_pilot/build_windows.py

echo "done"
find "$REPO/outputs/hand_pilot/poses" -type f | wc -l
