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

## Jubail SSH 操作铁律(本次血泪)

- `jubail` / `jubail-ts` 两条路由都会间歇抽风(banner 超时 / exec request failed),所有操作用双路由 for 循环重试 + `-o ControlPath=none`(复用 socket 会坏)。
- **上传(`cat file | ssh 'cat > remote'`)与后台启动(`setsid nohup ... &`)必须分开两次 ssh**:混在一条带 `&` 的命令里,后台化的 job stdin 变 `/dev/null`,`cat > file` 会把远程文件**写空**(本次两次中招)。
- `pkill/pgrep -f <脚本全路径>` 会匹配到 wrapper `bash -c` 自身命令行——pkill 直接自杀,pgrep 数出假阳性。判断后台脚本存活用**日志 mtime/大小增长**。
