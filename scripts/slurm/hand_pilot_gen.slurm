#!/bin/bash
#SBATCH --job-name=hand_gen
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

# Generic generation runner: sbatch hand_pilot_gen.slurm CFG OUTDIR NAME
# e.g. sbatch hand_pilot_gen.slurm configs/test_sign_handflow_reroll123.yaml \
#        outputs/sign_siren_reroll s123
set -euo pipefail
CFG=${1:?config path}
OUTDIR=${2:?output dir}
NAME=${3:?run name}
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
[ -f "$CFG" ] || { echo "missing $CFG"; exit 1; }
python inference_ctrl.py --inference_config "$CFG" \
  --output_dir "$OUTDIR" --name "$NAME"
echo "GEN $NAME DONE"
