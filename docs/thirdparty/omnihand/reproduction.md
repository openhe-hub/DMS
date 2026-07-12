# OmniHands 推理复现(Jubail)

> [OmniHands](https://github.com/LinDixuan/OmniHands)(ACM TOG 2026,[arXiv:2405.20330](https://arxiv.org/abs/2405.20330),[项目主页](https://omnihand.github.io/))单目视频双手 4D 重建。
> 复现日期 2026-07-12,Slurm job `16662229`,A100 单卡,demo 全程 4min16s,exit 0。

## 结论

推理 demo 在 Jubail 上端到端跑通:手语视频 + 舞蹈视频共 4 段,输出渲染视频中
左手(紫)/右手(青)MANO 网格与手势贴合准确。

> 渲染颜色改动:上游左手默认粉色,在肤色/粉色衣物上不可辨,已把
> `run_demo.py` 顶部 `LIGHT_RED` 改为 `(0.58, 0.24, 1.0)`(亮紫)。
> `thirdparty/` 不进 git,重新 clone 后需重打这一行补丁(本地和 jubail 均已改)。

**结果统一放在本地 `outputs/omnihand/`(gitignore,不进 git)**:各视频的
`video_<vname>.mp4` 双手网格 overlay 渲染 + `verify_frame.png` 验证帧;
远程原件在 jubail `thirdparty/omnihand/demo_out/`。

## 位置

| 项 | 路径 |
| --- | --- |
| 本地代码(仅作代码/文档库,不跑) | `thirdparty/omnihand/`(git clone + `git submodule update --init`) |
| 远程代码(jubail,`zl6890`) | `/scratch/zl6890/zhewen/DisPose/thirdparty/omnihand/` |
| conda 环境 | `omhand`(`/scratch/zl6890/miniconda`,python 3.10,torch 2.0.1+cu118) |
| setup 脚本(幂等,可整体重跑) | 远程 `/scratch/zl6890/zhewen/omnihand_setup.sh`,存档 [`scripts/thirdparty/omnihand/omnihand_setup.sh`](../../../scripts/thirdparty/omnihand/omnihand_setup.sh) |
| Slurm 作业脚本 | 远程 `/scratch/zl6890/zhewen/omnihand_demo.sbatch`,存档 [`scripts/thirdparty/omnihand/omnihand_demo.sbatch`](../../../scripts/thirdparty/omnihand/omnihand_demo.sbatch) |
| 输出(远程原件) | `thirdparty/omnihand/demo_out/video_<vname>.mp4` + `<vname>/bbox.json` |
| 输出(本地副本) | `outputs/omnihand/`(gitignore) |

## 权重与数据(全部已就位)

| 文件 | 来源 | 位置(相对 omnihand 仓库) |
| --- | --- | --- |
| `Demo_Video.pth` / `Demo_Image.pth`(各 3.1GB) | Google Drive,登录节点 `gdown` 直下 | `checkpoints/` |
| ViTPose+ huge `wholebody.pth`(3.8GB) | HaMeR demo tarball(6GB,cs.utexas.edu) | `_DATA/vitpose_ckpts/vitpose+_huge/` |
| `mano_mean_params.npz` | 同上 tarball | `_DATA/data/` |
| `MANO_LEFT/RIGHT.pkl` | **无需注册**,仓库自带 `hands_4d/misc/mano/` | `_DATA/data/mano/` |
| ViTDet `model_final_f05665.pkl`(2.6GB) | dl.fbaipublicfiles.com,预下到 iopath 缓存 | `~/.torch/iopath_cache/detectron2/ViTDet/...`(`~/.torch` 已软链到 scratch) |

## 再跑一次 / 换视频

```bash
# 改 omnihand_demo.sbatch 里的 VIDEO= 后:
ssh jubail 'bash -lc "sbatch /scratch/zl6890/zhewen/omnihand_demo.sbatch"'
```

作业参数:`-p nvidia --gres=gpu:a100:1 -c 8 --mem=64G`,`--mail-type=ALL --mail-user=zh3510@nyu.edu`。
图片模式:`--checkpoint ./checkpoints/Demo_Image.pth --cfg ./checkpoints/config_image.yaml --image_dir <dir> --mode image`。

## 踩坑记录(按遇到顺序)

1. **mmcv 1.3.9 构建失败** `No module named 'pkg_resources'`:新 setuptools(≥81)删除了 pkg_resources → 环境固定 `setuptools<81`,mmcv 用 `--no-build-isolation` 安装。
2. **detectron2 编译失败** `which g++ 非零`:Jubail 登录节点默认无编译器 → `module load gcc/11.5.0 cuda/11.8.0`,`FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.0;8.0"`。**因此作业必须限定 a100/v100**(H100 需要 sm_90,当时未编)。
3. **`third-party/ViTPose` 是空目录**:它是 git submodule(ViTAE-Transformer/ViTPose @ d5216452)→ clone 后必须 `git submodule update --init --recursive`。
4. **HaMeR tarball 解压 Unexpected EOF**:上一轮被中断的 curl 留下半截文件,而脚本只查"非空" → 改为 `gzip -t` 校验 + `curl -C -` 断点续传(ViTDet pkl 同理按 Content-Length 比对)。
5. **上游代码 bug**:`run_demo.py` 视频模式写 `demo_out/<vname>/bbox.json` 前不建目录(`main()` 里只 makedirs 了 `out_dir` 本身)→ sbatch 中先 `mkdir -p demo_out/$VNAME`,不改上游代码。

另:登录节点 `import pyrender` 必报 `Unable to load OpenGL library`——属预期(无 GL),计算节点上 `PYOPENGL_PLATFORM=egl` 正常渲染;`ckpt 加载时大量 `unexpected key ... mlp.experts.*` 警告是 ViTPose+ MoE 权重载入普通 ViT 的正常现象,可忽略。

## 去抖实验(2026-07-12,job 16675946)

上游视频模式抖动严重(已提 [issue #9](https://github.com/LinDixuan/OmniHands/issues/9)):
bbox 中心不平滑、时序序列 gap=10、`seq_smooth` 只对顶点做 70/30 弱混合且**完全不平滑相机平移**。

修复(`run_demo.py` 补丁,`OMNIHAND_SMOOTH=savgol` 环境变量开关,默认行为不变):
bbox 中心 SavGol(窗9, poly2)+ 顶点/相机平移 SavGol(窗11, poly3)替代 seq_smooth,
并 dump 平滑前后轨迹到 `<vname>/traj.npz`。作业脚本
[`omnihand_demo_smooth.sbatch`](../../../scripts/thirdparty/omnihand/omnihand_demo_smooth.sbatch)
(复用 baseline 的 bbox 缓存,4 视频 5min43s),指标脚本
[`jitter_metric.py`](../../../scripts/thirdparty/omnihand/jitter_metric.py)。

结果(逐帧二阶差分加速度均值,4 视频):**顶点抖动 ↓74–84%,相机平移抖动 ↓80–83%**;
raw cam_t 加速度达米级/帧²,证实相机平移是主要抖源。目检无滞后错位。
平滑版输出在本地 `outputs/omnihand/smooth/`。

## 2D 关键点导出——回接 DisPose 链路(2026-07-12)

动机:DisPose 用 DWPose 提 2D 手部关键点,检测不准时用 OmniHands 的 3D 恢复
再**投影回 2D** 送回原链路。转换脚本
[`omnihand_to_dwpose.py`](../../../scripts/thirdparty/omnihand/omnihand_to_dwpose.py)
(runner [`omnihand_kps.sh`](../../../scripts/thirdparty/omnihand/omnihand_kps.sh),
omhand 环境登录节点 CPU 即可,4 视频约 1 分钟):

- 关节回归与 `hands_4d/models/mano_wrapper.py` 完全一致:MANO `J_regressor`
  16 关节 + 5 指尖顶点,按 `mano_to_openpose` 重排——**即 DWPose 的
  COCO-WholeBody 手部顺序,一一对应**。右手 J_regressor 对左手同样成立
  (左手顶点是镜像的右手网格,回归是线性的)。
- 投影用渲染同款针孔相机:`f = 5000/256 × max(W,H)`,主点图像中心——mesh
  overlay 目检贴合即保证投影像素级正确。
- 输出与 hand_pilot SIREN arm 的 `hands_recon` npz 同格式:
  `{hands[T,2,21,2] 归一化(0=左,1=右), hands_score=0.61, covered[T,2]}`,
  可直接走 `mimicmotion/dwpose/preprocess.py` 的 `hand_override` /
  `hand_recon_dir` 注入口,DisPose 侧零改动。

结果(4 视频,`outputs/omnihand/kps/`,叠加视频 `kps_<vname>.mp4` + npz):
投影点在画面内 95.8–100%;慢动作帧手指级对齐;快速挥动+运动模糊帧骨架滞后于
真手——**raw 与 smoothed 投影偏移相同**(`kps_video1_raw.mp4` 对照),说明滞后
来自 OmniHands 时序融合(gap=10)本身而非 SavGol。后续融合策略:按 DWPose
置信度门控,高置信帧保留 DWPose,低置信/缺失帧用 OmniHands 投影补。

坑:**登录节点有线程数上限**,torch/BLAS 默认开满线程会
`libgomp: Thread creation failed` 甚至段错误(报成 `numpy._core.multiarray
failed to import` 的假 ABI 错)→ runner 里
`export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`。

## Jubail SSH 操作铁律(本次血泪)

- `jubail` / `jubail-ts` 两条路由都会间歇抽风(banner 超时 / exec request failed),所有操作用双路由 for 循环重试 + `-o ControlPath=none`(复用 socket 会坏)。
- **上传(`cat file | ssh 'cat > remote'`)与后台启动(`setsid nohup ... &`)必须分开两次 ssh**:混在一条带 `&` 的命令里,后台化的 job stdin 变 `/dev/null`,`cat > file` 会把远程文件**写空**(本次两次中招)。
- `pkill/pgrep -f <脚本全路径>` 会匹配到 wrapper `bash -c` 自身命令行——pkill 直接自杀,pgrep 数出假阳性。判断后台脚本存活用**日志 mtime/大小增长**。
