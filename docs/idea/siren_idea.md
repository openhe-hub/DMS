# SIREN × DisPose — idea 与定位

> 把 NIAF / PA-HiRes 的"连续可微运动表示（SIREN）"思路嫁接到 DisPose 的 pose 控制信号
> （pose 关键点轨迹 / motion field）。本文档只记 **idea / 动机 / 查新 / 定位 / 结论**。
> **架构图 + 实验数据**见 [`../experiments/siren_module.md`](../experiments/siren_module.md)。
> 日期：2026-06（基于本地复现的 DisPose 仓库）。

---

## 0. TL;DR

> **⚰️ 最终结论（2026-07-02,step2 video-level pilot 后）:idea 已按预注册 gate 判死。**
> 完整证据链:① 去噪打法死于 step1(真实 DWPose 抖动仅 2–3px,learned-INR 输给线性插值,
> 见 `../experiments/step1_real_validation.md`);② 时域超分/低fps打法死于 step2 pilot
> (RIFE 后处理 mid-PSNR 6/6 全胜控制侧连续化;linear≈spline≈siren,INR 垫底,
> 见 `../experiments/step2_video_level.md`)。
> 根本机制:**扩散的控制跟随误差 >> 控制信号的时间/噪声精度误差**——
> pose 控制信号侧的任何精化在 video-level 都是二阶小量。此结论对"更好的 motion field"
> 这一整类 idea 都成立,不止 SIREN。

- **简单版（per-clip 测试时拟合 SIREN，无先验）打不过 baseline** —— 只是"光滑地插值噪声"。fd+高斯是又便宜又强的 baseline（这正是 DisPose 用它的原因）。
- **学习版（摊销式 FiLM-调制 SIREN，跨序列训练 + 速度监督）在合成数据上决定性跑赢** —— 4 档噪声 velMSE 领先最强 baseline **约 2.2–2.7×**。~~核心假设成立~~ → **后被 step1 证明是 train/test 同分布的产物,真实数据上不成立**。
- ~~路线确定：走学习版~~ → step1 真实轨迹上完败(domain gap + 数据墙),转打"时域超分"(step2)→ 亦死于 RIFE anti-claim。
- 表示层面 novelty 已被 **PA-HiRes**（2512.21183）和 **NIAF**（2603.01766）双重抢注 —— 且最终连"装法"的下游收益也被证伪。

---

## 1. 两篇源论文的关系

两个**不同领域**，唯一精神共鸣是"少用过度离散/稠密的硬控制信号，转向更连续、更解耦的表示"。

| | DisPose (ICLR 2025) | NIAF (2026 preprint) |
|---|---|---|
| 领域 | 可控人体**图像/视频动画** | **机器人 VLA** 动作生成 |
| 输入→输出 | 参考图+驱动视频 → 动画视频 | 图像+指令 → 机器人动作 |
| 核心动作 | 解耦稀疏姿态 → motion field + keypoint correspondence | 离散航点 → 连续 SIREN 函数 𝒜(τ)=Φ(τ;θ) |
| 关键模块 | CMP 稀疏光流传播 + DIFT 特征 + Hybrid ControlNet（冻结主干） | SIREN 解码器 + MLLM 当谱调制器(hypernetwork) + 速度/jerk 监督 |

### SIREN 是什么（NIAF 的核心积木）
- SIREN = 把 MLP 激活换成 `sin` 的隐式神经表示（Sitzmann 2020）。
- 关键性质：`d/dx sin = cos`，所以**任意阶导数都行为良好、C^∞ 光滑、可解析计算**。
- NIAF 用它当动作解码器：位置=𝒜(τ)、速度=𝒜'(τ)、jerk=𝒜'''(τ) 全部解析可得，不需数值差分。

### 嫁接的"命门对应"（最初动机）
DisPose 算稀疏 motion field 的两步，恰好是 NIAF 点名批判的：
1. `P_s = 相邻帧 pose 坐标差`（`mimicmotion/utils/utils.py:173` `points_to_flows`）= **数值差分** → NIAF 说放大噪声。
2. `高斯滤波稳定化`（`inference_ctrl.py:105-108`，kernel=199）= 手工平滑 hack。

→ 直觉：用连续可微表示替差分+高斯，还能任意 fps。**动机干净，但需实测** —— 实测结论见实验文档（简单版不行、学习版赢）。

---

## 2. 查新结论：积木不新，"装法"未被占

跑了 10 组 arXiv 检索，核了 5 篇最像的先验（用了 `literature-search-arxiv` 技能）。

### 最危险的先验

| 论文 | 做了什么 | 区别（我们的护城河） |
|---|---|---|
| **PA-HiRes** / arXiv:2512.21183（2025-12，最危险）| "首次"用 INR + Fourier 参数化周期激活 + 速度一致性损失，做 **3D mocap 运动序列**的任意帧率插值/补间/外推 | 纯 mocap 关节空间，**不碰 RGB 视频/扩散/pose 控制信号/2D 关键点**；评测是关节 PSNR/SSIM |
| **NeMF** / arXiv:2206.03287（He et al. 2022）| INR 把 3D 运动表示成时间连续函数，VAE 生成先验 | 运动生成/编辑，非动画控制信号 |
| **INR Variable-Length Motion** / 2203.13694 | INR+时间嵌入做变长运动生成 | 纯运动生成 |
| **CPAB** / 2401.09146 | 从关键点构造连续(微分同胚)变换做 image animation | GAN 时代的**空间 warp 场**连续化，非**时间轨迹 INR**、非扩散 |

### 没人做的事 = 真空地带
1. 把 2D pose 关键点轨迹用 SIREN/INR 连续化，作为**注入 pose-guided 扩散动画模型的控制信号** —— 查不到。
2. 用解析速度场替换 DisPose 的 finite-diff+高斯，换取**生成视频**的时序平滑 —— 查不到。
3. **任意 fps 的 RGB 视频**生成（PA-HiRes 只做 motion，不出像素）—— 查不到。

### 定位建议
- ❌ 别写"我们提出连续/SIREN 运动表示"——会被 PA-HiRes/NeMF/NIAF 秒。
- ✅ 写"首次把连续可微 pose 轨迹**接入扩散式人体动画控制分支**，换来 (a) 可量化的 VBench 时序平滑/抗闪烁、(b) 任意帧率视频生成新能力"。
- related work 必须同时正面引 PA-HiRes 和 NIAF 并切割。

**检索/核查论文 URL**：2512.21183、2206.03287、2203.13694、2401.09146、2508.04049（其余命中：2407.09012 TCAN、2311.16498 MagicAnimate、2507.15064 StableAnimator++、2506.03119 等，均为扩散动画/插值但未用 INR 表示控制信号）。

---

## 3. 结论 & 方向

**简单版（替换差分+高斯的 per-clip 拟合）在精度上死了**——非调参问题，是无先验拟合的本质局限（实测见实验文档 §3）。

**学习版（Option B）已在合成数据上验证可行且决定性跑赢**（实验文档 §4），不再是"赌"。关键风险从"方法能不能赢"转移到"**能不能在真实 DWPose 轨迹 + 真实生成视频指标上复现这个赢**"。

下一步明确：
1. **真实轨迹验证**：集群上从真实 DWPose `P_traj` 提轨迹，同协议测 learned-INR vs fd+高斯。
2. **接入 DisPose + 视频指标**：learned-INR 接到 `controlnet` motion 分支，比 VBench Motion Smoothness / Temporal Flickering + FID-FVD。
3. **任意 fps 能力 demo**：连续 pose → 高帧率视频生成（baseline 缺席的能力）。
4. related work 硬切 PA-HiRes、NIAF；贡献口径 = "接入扩散动画控制 + 下游视频收益 + 任意帧率能力"。

> 模块架构图、整体 pipeline 图、完整实验数据 → [`../experiments/siren_module.md`](../experiments/siren_module.md)
