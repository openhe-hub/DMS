#!/bin/bash
#SBATCH --job-name=face_preview
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zh3510@nyu.edu
#SBATCH --output=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.out
#SBATCH --error=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.err

# Skeleton-only preview of the head/face blend-ratio sweep (no SVD): DWPose
# once, 10 combos replayed + labeled grid mp4. v100 dodges the a100 queue.
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
python scripts/face_blend/render_pose_preview.py \
    --video assets/example_data/sign_videos/5ok8y3eheq8_7-1-rgb_front_8s.mp4 \
    --ref assets/example_data/sign_videos/refs/test2.jpg \
    --out outputs/face_blend/preview
echo "FACE_PREVIEW_DONE"
ls -la outputs/face_blend/preview/
