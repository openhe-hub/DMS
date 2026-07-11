#!/bin/bash
#SBATCH --job-name=bestof_siren
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zh3510@nyu.edu
#SBATCH --output=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.out
#SBATCH --error=/scratch/zl6890/zhewen/DisPose/slurm/%x_%j.err

# Best-of-N pipeline: score rerolls (hand conf, GT-free) -> select argmax per
# clip -> full metric suite on the selected set.
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

cd "$REPO"
SRC=assets/example_data/sign_videos/hard27k_orig
CACHE=outputs/metrics_hard27k/pose_cache

for S in s123 s777; do
  python src/metrics/run_eval.py --source-dir $SRC \
    --dispose-dirs outputs/sign_siren_reroll/*_${S} \
    --out-dir outputs/metrics_reroll_${S} --cache-dir $CACHE
done

python scripts/hand_pilot/select_best.py \
  --reroll outputs/metrics_reroll_s123/per_video.csv "outputs/sign_siren_reroll/*_s123" \
  --reroll outputs/metrics_reroll_s777/per_video.csv "outputs/sign_siren_reroll/*_s777"

echo "===== final metrics on the selected set ====="
python src/metrics/run_eval.py --source-dir $SRC \
  --dispose-dirs outputs/sign_siren_best/best \
  --out-dir outputs/metrics_siren_best --cache-dir $CACHE
cat outputs/metrics_siren_best/aggregate.csv
python src/metrics/run_csim.py --model siren_best \
  --ref-image assets/example_data/sign_videos/refs/test2.jpg \
  --source-dir $SRC --gen-dirs outputs/sign_siren_best/best \
  --out outputs/metrics_siren_best/csim_siren_best.csv
python src/metrics/run_fvd.py --model siren_best \
  --i3d $BASE/fvd_models/i3d_torchscript.pt \
  --source-dir $SRC --gen-dirs outputs/sign_siren_best/best \
  --out outputs/metrics_siren_best/fvd_siren_best.json
cat outputs/metrics_siren_best/fvd_siren_best.json
echo "BESTOF DONE"
