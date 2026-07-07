#!/bin/bash
#SBATCH --job-name=hand_gate_a
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

# Gate A: hand_flow channel-causality probe.
# Usage: sbatch hand_pilot_gate_a.slurm MODE
#   MODE=check  : blocking regression (41 real + original 12_equiv_check).
#                 MUST PASS before submitting any generation arm.
#   MODE=off|raw|smooth : one paired generation arm (8 cases, ~2.5-3 h).
set -euo pipefail
MODE=${1:?check|off|raw|smooth}

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
nvidia-smi || true

if [ "$MODE" = "check" ]; then
  echo "===== 41 hand_flow regression (synthetic + real) ====="
  python scripts/hand_pilot/41_equiv_check_hands.py --mode real \
    --config configs/test_sign_hard27k.yaml --case 0
  echo "===== original step2 equivalence gate must still PASS ====="
  python scripts/step2/12_equiv_check.py --config configs/test.yaml --case 1
  echo "CHECK DONE"
else
  CFG=configs/test_sign_handflow_${MODE}.yaml
  [ -f "$CFG" ] || { echo "missing $CFG -- run 40_make_gate_a_configs.py"; exit 1; }
  python inference_ctrl.py --inference_config "$CFG" \
    --output_dir outputs/hand_pilot/gate_a --name "$MODE"
  echo "ARM $MODE DONE"
  find outputs/hand_pilot/gate_a -name '*.mp4' | sort | tail -20
fi
