#!/bin/bash
#SBATCH --job-name=dispose_infer
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.out
#SBATCH --error=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zh3510@nyu.edu

set -euo pipefail
BASE=/scratch/zl6890/zhewen
REPO=$BASE/DisPose
ENVDIR=$BASE/envs/dispose

module purge
module load miniconda/3-4.11.0
module load cuda/12.2.0
source activate "$ENVDIR"

export HF_HOME=$BASE/hf_cache
export HF_TOKEN=$(cat /home/zl6890/.cache/huggingface/token 2>/dev/null)
export HF_HUB_DISABLE_TELEMETRY=1
# let onnxruntime-gpu (DWPose) find cuDNN 9 / cuBLAS 12 bundled with torch
export LD_LIBRARY_PATH=$ENVDIR/lib/python3.10/site-packages/nvidia/cudnn/lib:$ENVDIR/lib/python3.10/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}
# keep onnxruntime from spamming CPU-affinity warnings
export OMP_NUM_THREADS=8

cd "$REPO"
nvidia-smi || true
python -c "import onnxruntime as ort; print('ORT providers:', ort.get_available_providers())"
echo "===== start inference ====="
CUDA_VISIBLE_DEVICES=0 python inference_ctrl.py \
    --inference_config configs/test.yaml \
    --name test
echo "===== inference done ====="
echo "Outputs:"
find "$REPO/outputs" -name '*.mp4' 2>/dev/null
