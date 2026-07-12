#!/bin/bash
# OmniHands setup on Jubail — clone + conda env + weights + _DATA assembly
set -uo pipefail
set -x

SCRATCH=/scratch/zl6890/zhewen
REPO=$SCRATCH/DisPose/thirdparty/omnihand
DL=$SCRATCH/omnihand_downloads
export PIP_CACHE_DIR=$SCRATCH/pip_cache_omnihand
mkdir -p "$DL" "$PIP_CACHE_DIR"

step() { echo "===== [$(date +%H:%M:%S)] $1 ====="; }

fail() { echo "SETUP_FAILED: $1"; exit 1; }

# ---------- 1. clone ----------
step "clone"
mkdir -p "$SCRATCH/DisPose/thirdparty"
if [ ! -d "$REPO/.git" ]; then
  git clone https://github.com/LinDixuan/OmniHands "$REPO" || fail clone
fi
cd "$REPO"
git submodule update --init --recursive || fail submodule
mkdir -p _DATA/data/mano _DATA/vitpose_ckpts/vitpose+_huge checkpoints demo_out

# ---------- 2. background downloads ----------
step "start background downloads"
if ! gzip -t "$DL/hamer_demo_data.tar.gz" 2>/dev/null; then
  curl -sL --retry 3 -C - -o "$DL/hamer_demo_data.tar.gz" \
    https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz &
  HAMER_PID=$!
else
  HAMER_PID=""
fi

# ~/.torch -> scratch symlink so iopath cache lands on scratch, then prefetch ViTDet ckpt
if [ ! -e ~/.torch ]; then
  mkdir -p "$SCRATCH/torch_cache" && ln -sfn "$SCRATCH/torch_cache" ~/.torch
fi
VITDET_DIR=~/.torch/iopath_cache/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692
mkdir -p "$VITDET_DIR"
VITDET_URL=https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl
VITDET_WANT=$(curl -sIL "$VITDET_URL" | grep -i content-length | tail -1 | tr -dc 0-9)
VITDET_HAVE=$(stat -c%s "$VITDET_DIR/model_final_f05665.pkl" 2>/dev/null || echo 0)
echo "vitdet want=$VITDET_WANT have=$VITDET_HAVE"
if [ "$VITDET_HAVE" != "$VITDET_WANT" ]; then
  curl -sL --retry 3 -C - -o "$VITDET_DIR/model_final_f05665.pkl" "$VITDET_URL" &
  VITDET_PID=$!
else
  VITDET_PID=""
fi

# ---------- 3. conda env ----------
step "conda env"
source /scratch/zl6890/miniconda/etc/profile.d/conda.sh || fail conda-source
conda env list | grep -qE '^omhand\s' || conda create -y -n omhand python=3.10 || true
conda activate omhand || fail conda-activate
python -V

step "pip torch"
pip install --upgrade pip "setuptools<81" wheel ninja
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118 || fail torch

step "pip deps"
pip install numpy==1.23.5 scipy opencv-python-headless pyrender "pytorch-lightning>=2.0,<2.4" \
  scikit-image smplx==0.1.28 yacs timm einops pandas plyfile gdown hydra-core \
  hydra-submitit-launcher hydra-colorlog pyrootutils rich webdataset trimesh \
  xtcocotools json_tricks munkres || fail pip-deps
pip install --no-build-isolation mmcv==1.3.9 || fail mmcv
pip install --no-build-isolation "chumpy @ git+https://github.com/mattloper/chumpy" || fail chumpy

step "detectron2 (CUDA build)"
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/11.8.0 2>/dev/null || module load cuda/11.8 2>/dev/null || true
module load gcc/11.5.0 2>/dev/null || true
which nvcc || fail nvcc-missing
which g++ || fail gxx-missing
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.0;8.0" MAX_JOBS=8 \
  pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2" || fail detectron2

step "install omnihand + vitpose"
pip install -e . --no-deps || fail omnihand-install
pip install -e third-party/ViTPose --no-deps || fail vitpose-install
pip install numpy==1.23.5  # re-pin in case anything bumped it

# ---------- 4. wait downloads, assemble _DATA ----------
step "wait downloads"
[ -n "${HAMER_PID}" ] && { wait "$HAMER_PID" || fail hamer-download; }
[ -n "${VITDET_PID}" ] && { wait "$VITDET_PID" || fail vitdet-download; }
ls -la "$DL" "$VITDET_DIR"

step "assemble _DATA"
if [ ! -s _DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth ]; then
  tar -xzf "$DL/hamer_demo_data.tar.gz" -C "$DL" || fail hamer-extract
  find "$DL" -name "wholebody.pth" -o -name "mano_mean_params.npz" | head
  cp "$(find "$DL" -name wholebody.pth | head -1)" _DATA/vitpose_ckpts/vitpose+_huge/ || fail wholebody
  cp "$(find "$DL" -name mano_mean_params.npz | head -1)" _DATA/data/ || fail mean-params
fi
cp hands_4d/misc/mano/MANO_LEFT.pkl hands_4d/misc/mano/MANO_RIGHT.pkl _DATA/data/mano/ || fail mano

# ---------- 5. OmniHands checkpoints from Google Drive ----------
step "gdown checkpoints"
if [ ! -s checkpoints/Demo_Video.pth ]; then
  gdown 1ZoP4qmYE8MyXCfhGK5meWWfBOYpy1VZ7 -O checkpoints/Demo_Video.pth || echo "GDOWN_VIDEO_FAILED"
fi
if [ ! -s checkpoints/Demo_Image.pth ]; then
  gdown 1jLo7cFIWeDXep_hhvumWdm90QIwy_HwB -O checkpoints/Demo_Image.pth || echo "GDOWN_IMAGE_FAILED"
fi
ls -la checkpoints/

# ---------- 6. sanity imports ----------
step "sanity imports"
python - <<'EOF'
import torch, torchvision, detectron2, mmcv, pyrender, chumpy, smplx, trimesh
import hands_4d, hands_multiview
from mmpose.apis import init_pose_model
print("imports OK | torch", torch.__version__, "| cuda build", torch.version.cuda)
import pickle
m = pickle.load(open('_DATA/data/mano/MANO_RIGHT.pkl','rb'), encoding='latin1')
print("MANO pkl OK, faces:", m['f'].shape)
EOF
[ $? -eq 0 ] || fail sanity-imports

step "DONE"
echo "SETUP_OK"
