#!/bin/bash
#SBATCH --job-name=siren_full
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=80g
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zh3510@nyu.edu
#SBATCH --output=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.out
#SBATCH --error=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.err

# P2 third system: DisPose+graft+SIREN-hand on the full 109-case benchmark.
# Usage: sbatch hand_pilot_siren_full.slurm SHARD   (0|1|2)
# Requires outputs/hand_pilot/hands_recon (43, already built for all 109).
set -euo pipefail
SHARD=${1:?0|1|2}
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
CFG=configs/siren/test_sign_handflow_siren_full_${SHARD}.yaml
[ -f "$CFG" ] || { echo "missing $CFG"; exit 1; }
python inference_ctrl.py --inference_config "$CFG" \
  --output_dir outputs/sign_siren_full --name "shard${SHARD}"
echo "SIREN FULL SHARD ${SHARD} DONE"
find outputs/sign_siren_full -name '*.mp4' | wc -l
