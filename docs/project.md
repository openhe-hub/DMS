# DisPose 推理复现记录（Jubail HPC）

> 目标：在 NYU Abu Dhabi 的 **Jubail** 集群上复现 [DisPose](https://github.com/lihxxx/DisPose)（ICLR 2025 可控人物图像动画）的**推理**流程。仅复现到推理为止，不复现训练。
>
> 复现日期：2026-06-26 ｜ 状态：✅ 完成（3 个示例全部跑通）

---

## 1. 环境与路径

| 项目 | 值 |
|---|---|
| 集群 | Jubail（NYU AD HPC），SSH 走 `jubail-ts`（`-ts` 备用路由） |
| 工作根目录 | `/scratch/zl6890/zhewen/` |
| 仓库 | `/scratch/zl6890/zhewen/DisPose` |
| Conda 环境 | `/scratch/zl6890/zhewen/envs/dispose`（Python 3.10，torch 2.4.1+cu121） |
| HF 缓存 | `/scratch/zl6890/zhewen/hf_cache` |
| GPU | partition `nvidia`，A100 80GB，`--gres=gpu:a100:1` |
| Module | `miniconda/3-4.11.0`、`cuda/12.2.0` |

> ⚠️ 所有大文件放在 `scratch`（home 配额很小）。登录节点上 `module` / `sbatch` 需要在 `bash -lc` 下使用。

复现脚本（已固化进仓库，可一键重跑）：

```
scripts/run/setup_env.sh          # 建 conda 环境 + 装依赖
scripts/run/download_weights.sh   # 下载全部权重
scripts/slurm/run_inference.slurm # GPU 推理作业（含邮件通知）
```

---

## 2. 复现步骤

```bash
# 0) 登录并进入工作目录
ssh jubail-ts
cd /scratch/zl6890/zhewen

# 1) clone
git clone https://github.com/lihxxx/DisPose.git
cd DisPose

# 2) 建环境（后台跑，约几分钟）
bash scripts/run/setup_env.sh

# 3) 下载权重（约 14GB；SVD 为 gated 仓库，需 HF token）
bash scripts/run/download_weights.sh

# 4) 提交 GPU 推理作业
sbatch scripts/slurm/run_inference.slurm
```

权重清单（落在 `pretrained_weights/`）：

| 权重 | 来源 |
|---|---|
| `DisPose.pth` | `lihxxx/DisPose` |
| `MimicMotion_1-1.pth` | `tencent/MimicMotion` |
| `DWPose/{dw-ll_ucoco_384.onnx, yolox_l.onnx}` | `yzd-v/DWPose` |
| `stable-video-diffusion-img2vid-xt-1-1/` | `stabilityai/...`（**gated**） |
| `stable-diffusion-v1-5/`（含 fp16 变体） | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| `.../cmp/.../checkpoints/ckpt_iter_42000.pth.tar` | `MyNiuuu/MOFA-Video-Hybrid` |

---

## 3. 踩坑与修复（关键）

复现过程中遇到 5 个会直接导致作业失败/极慢的问题，均已修复并固化进脚本：

1. **OpenCV `libGL.so.1` 缺失** — 计算节点无 libGL。
   → 把 `opencv_contrib_python` 换成 `opencv-contrib-python-headless`（代码只用 `cv2.filter2D` 等，无 GUI）。

2. **DWPose 目录名大小写** — 代码默认路径是 `./pretrained_weights/DWPose/`（大写），不是 `dwpose`。
   → 目录命名为 `DWPose`。

3. **SVD 为 gated 仓库** — 匿名访问 `stable-video-diffusion-img2vid-xt-1-1` 报 `401 GatedRepoError`。
   → 下载时传 `HF_TOKEN=$(cat ~/.cache/huggingface/token)`（账号已有访问权限）。

4. **SD1.5 缺 fp16 变体** — DIFT 用 `variant="fp16"` 加载 SD1.5 的 unet/vae/text_encoder。
   → 必须额外下载 `*.fp16.safetensors`，仅下默认权重会报 `no file named diffusion_pytorch_model.fp16.bin`。

5. **onnxruntime 是 CPU 版** — DWPose 在 CPU 上约 **18 s/帧**（一个视频 ~34 分钟）。
   → 装 `onnxruntime-gpu==1.19.2`，并在作业里 `export LD_LIBRARY_PATH` 指向环境内 `nvidia/cudnn/lib` + `nvidia/cublas/lib`（torch 自带 cuDNN9 / cuBLAS12）。提速到 **113 帧 ~3.5 秒**。

---

## 4. Slurm 作业要点

```bash
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zh3510@nyu.edu
```

- 作业开始 / 结束都会发邮件到 `zh3510@nyu.edu`（`--mail-type=ALL`）。
- 推理脚本：`bash scripts/test.sh`（即 `inference_ctrl.py --inference_config configs/test.yaml`）。

---

## 5. 结果

作业 `16433397` **COMPLETED**（exit 0:0，约 75 分钟），生成 3 个视频，分辨率 576 宽（9:16）：

| 输出 | 帧数 × 分辨率 |
|---|---|
| `ref1_to_video1_CFG2.0_16_15.mp4` | 113 × 1024×576 |
| `ref2_to_video2_CFG2.0_16_15.mp4` | 148 × 1024×576 |
| `ref3_to_video3_CFG2.0_16_15.mp4` | 141 × 1024×576 |

集群路径：`/scratch/zl6890/zhewen/DisPose/outputs/20260626_test/`
本地（已 rsync 回）：`outputs/20260626_test/`，并在 `inputs/` 放了对应的参考图 `refN.png` 与驱动视频 `videoN_driving.mp4` 供对照。

> 每组 = 参考图（template）+ 驱动动作视频 → DisPose 生成的动画：参考图中的人物按驱动视频的姿态动起来。

---

## 6. 重跑 / 调参

```bash
sbatch scripts/slurm/run_inference.slurm   # 直接重跑
```

调参改 `configs/test.yaml`：`resolution`、`num_inference_steps`、`guidance_scale`、`sample_stride`、`decode_chunk_size`（显存紧张时设为 1）等。
