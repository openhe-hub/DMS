#!/bin/bash
#SBATCH --job-name=omnihand_fusion_prep
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zh3510@nyu.edu
#SBATCH --output=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.out
#SBATCH --error=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.err

# Prep for the OmniHands x DisPose fusion arms:
#  1. stage OmniHands kps npz for arm B (full override)
#  2. extract DWPose hand tracks for the sign videos (GPU, onnxruntime)
#  3. gated fusion -> arm C npz + stats + overlays
set -euo pipefail
BASE=/scratch/zl6890/zhewen
REPO=$BASE/DisPose
ENVDIR=$BASE/envs/dispose

module purge
module load miniconda/3-4.11.0
module load cuda/12.2.0
source activate "$ENVDIR"
# onnxruntime-gpu needs torch-bundled cuDNN9/cuBLAS12 (see run_inference.slurm)
export LD_LIBRARY_PATH=$ENVDIR/lib/python3.10/site-packages/nvidia/cudnn/lib:$ENVDIR/lib/python3.10/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=8
export KMP_DUPLICATE_LIB_OK=TRUE

cd "$REPO"

mkdir -p outputs/omnihand_kps
cp thirdparty/omnihand/kps_out/*.npz outputs/omnihand_kps/

python scripts/hand_pilot/extract_hand_poses.py \
    --video_dir assets/example_data/sign_videos \
    --out_dir outputs/hand_fusion/poses \
    --skip_existing

python scripts/hand_fusion/fuse_kps.py \
    --poses_dir outputs/hand_fusion/poses \
    --omnihand_dir outputs/omnihand_kps \
    --out_dir outputs/hand_fusion/fused \
    --videos_dir assets/example_data/sign_videos

echo "FUSION_PREP_DONE"
ls -la outputs/hand_fusion/fused/
