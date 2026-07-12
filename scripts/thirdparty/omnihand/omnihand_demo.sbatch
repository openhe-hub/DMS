#!/bin/bash
#SBATCH --job-name=omnihand_demo
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zh3510@nyu.edu
#SBATCH --output=/scratch/zl6890/zhewen/DisPose/thirdparty/omnihand/slurm_demo_%j.out

set -x
source /scratch/zl6890/miniconda/etc/profile.d/conda.sh
conda activate omhand

cd /scratch/zl6890/zhewen/DisPose/thirdparty/omnihand
export PROJECT_ROOT=$PWD
export PYOPENGL_PLATFORM=egl

nvidia-smi
python -c "import torch; print('cuda avail:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

EX=/scratch/zl6890/zhewen/DisPose/assets/example_data
VIDEOS=(
  "$EX/sign_videos/5ok8y3eheq8_7-1-rgb_front_8s.mp4"
  "$EX/sign_videos/1aRNY8wFqa0_32-8-rgb_front_8s.mp4"
  "$EX/sign_videos/DI6T6tbk3r0_15-5-rgb_front_8s.mp4"
  "$EX/videos/video1.mp4"
)

for VIDEO in "${VIDEOS[@]}"; do
    VNAME=$(basename "$VIDEO" .mp4)
    if [ -s "./demo_out/video_${VNAME}.mp4" ]; then
        echo "SKIP $VNAME (already done)"
        continue
    fi
    # upstream bug: run_demo writes demo_out/<vname>/bbox.json without creating the dir
    mkdir -p "./demo_out/$VNAME"
    python run_demo.py \
        --checkpoint ./checkpoints/Demo_Video.pth \
        --cfg ./checkpoints/config_video.yaml \
        --video_dir "$VIDEO" \
        --out_dir ./demo_out \
        --gpu 0 \
        --mode video
    echo "DONE $VNAME EXIT=$?"
done

echo "ALL_VIDEOS_DONE"
ls -la ./demo_out/
