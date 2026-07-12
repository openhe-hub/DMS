#!/bin/bash
# Export DWPose-format 2D hand keypoints from the OmniHands smooth runs.
# CPU-only and light — runs on the jubail login node in the omhand env.
# Remote copies: /scratch/zl6890/zhewen/omnihand_kps.sh + omnihand_to_dwpose.py
set -ex
source /scratch/zl6890/miniconda/etc/profile.d/conda.sh
conda activate omhand

# login node caps user threads: BLAS/OpenMP pool creation segfaults without this
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

cd /scratch/zl6890/zhewen/DisPose/thirdparty/omnihand
EX=/scratch/zl6890/zhewen/DisPose/assets/example_data
VIDEOS=(
  "$EX/sign_videos/5ok8y3eheq8_7-1-rgb_front_8s.mp4"
  "$EX/sign_videos/1aRNY8wFqa0_32-8-rgb_front_8s.mp4"
  "$EX/sign_videos/DI6T6tbk3r0_15-5-rgb_front_8s.mp4"
  "$EX/videos/video1.mp4"
)

for VIDEO in "${VIDEOS[@]}"; do
    VNAME=$(basename "$VIDEO" .mp4)
    python /scratch/zl6890/zhewen/omnihand_to_dwpose.py \
        --traj "demo_out_smooth/$VNAME/traj.npz" \
        --video "$VIDEO" \
        --mano _DATA/data/mano \
        --out "kps_out/$VNAME.npz" \
        --overlay "kps_out/kps_$VNAME.mp4"
done

echo "ALL_KPS_DONE"
ls -la kps_out/
