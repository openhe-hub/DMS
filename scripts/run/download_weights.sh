#!/bin/bash
set -euo pipefail
BASE=/scratch/zl6890/zhewen
REPO=$BASE/DisPose
ENVDIR=$BASE/envs/dispose

module purge
module load miniconda/3-4.11.0
source activate "$ENVDIR"
export HF_TOKEN=$(cat /home/zl6890/.cache/huggingface/token 2>/dev/null)

export HF_HOME=$BASE/hf_cache
export HF_HUB_DISABLE_TELEMETRY=1

cd "$REPO"
PW="$REPO/pretrained_weights"
mkdir -p "$PW/DWPose"
CMP_DIR="$REPO/mimicmotion/modules/cmp/experiments/semiauto_annot/resnet50_vip+mpii_liteflow/checkpoints"
mkdir -p "$CMP_DIR"

echo "[dl] DisPose.pth"
huggingface-cli download lihxxx/DisPose DisPose.pth --local-dir "$PW"

echo "[dl] MimicMotion_1-1.pth"
huggingface-cli download tencent/MimicMotion MimicMotion_1-1.pth --local-dir "$PW"

echo "[dl] DWPose onnx"
huggingface-cli download yzd-v/DWPose dw-ll_ucoco_384.onnx yolox_l.onnx --local-dir "$PW/DWPose"

echo "[dl] SVD img2vid xt 1.1 (needed subfolders only)"
huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt-1-1 \
  --local-dir "$PW/stable-video-diffusion-img2vid-xt-1-1" \
  --include "model_index.json" "unet/config.json" "vae/*" "image_encoder/*" "scheduler/*" "feature_extractor/*"

echo "[dl] stable-diffusion-v1-5 (diffusers components)"
huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --local-dir "$PW/stable-diffusion-v1-5" \
  --include "model_index.json" "unet/config.json" "unet/diffusion_pytorch_model.safetensors" \
            "vae/config.json" "vae/diffusion_pytorch_model.safetensors" \
            "text_encoder/config.json" "text_encoder/model.safetensors" \
            "tokenizer/*" "scheduler/*" "feature_extractor/*" "unet/diffusion_pytorch_model.fp16.safetensors" "vae/diffusion_pytorch_model.fp16.safetensors" "text_encoder/model.fp16.safetensors"

echo "[dl] CMP ckpt_iter_42000.pth.tar"
wget -q --show-progress -O "$CMP_DIR/ckpt_iter_42000.pth.tar" \
  "https://huggingface.co/MyNiuuu/MOFA-Video-Hybrid/resolve/main/models/cmp/experiments/semiauto_annot/resnet50_vip%2Bmpii_liteflow/checkpoints/ckpt_iter_42000.pth.tar"

echo "[dl] DONE. Tree:"
ls -R "$PW" | head -80
