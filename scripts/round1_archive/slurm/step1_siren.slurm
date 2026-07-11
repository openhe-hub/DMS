#!/bin/bash
#SBATCH --job-name=step1_siren
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.out
#SBATCH --error=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.err

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

echo "===== 1.1 extract real DWPose trajectories ====="
python scripts/round1_archive/step1/01_extract_trajectories.py --stride 1

echo "===== 1.2 train scale-invariant amortized FiLM-SIREN + synthetic sanity ====="
python scripts/round1_archive/step1/02_train_amortized.py --epochs 400 --ntr 2000

echo "===== 1.3 evaluate on real trajectories (both protocols) ====="
python scripts/round1_archive/step1/03_eval_real.py --span 48 --step 24

echo "===== done ====="
find "$REPO/outputs/step1" -type f | sort
