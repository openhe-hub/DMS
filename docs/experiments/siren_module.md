# SIREN 连续 pose 表示模块 — 架构与实验

> learned-INR（摊销式 FiLM-调制 SIREN）模块的结构、嵌入 DisPose 的整体 pipeline，以及合成验证实验数据。
> 背景动机/查新/定位见 [`../idea/siren_idea.md`](../idea/siren_idea.md)。
> 日期：2026-06。实验环境：本地 `scratchpad/siren-env`（torch 2.12 CPU + matplotlib），独立 venv。

---

## 1. 模块架构：learned-INR（FiLMSIREN）

跑赢 baseline 的模型 = **摊销式 hypernetwork + FiLM 调制 SIREN**，即 NIAF/PA-HiRes 的精简版。
代码：`outputs/siren_idea/learned_experiment.py` 的 `FiLMSIREN`。

```
   x_noisy (B,16,2)  ──flatten──►  32
        │
        ▼
   ┌──────────────────────────────────────────────────┐
   │ Encoder (MLP)            32→128→ReLU→128→ReLU→96 │
   │   → latent z  (96-d)                             │
   └──────────────────────────────────────────────────┘
        │
        ▼
   ┌──────────────────────────────────────────────────────┐
   │ to_film:  Linear(96 → 384)      384 = L·H·2 = 3·64·2 │
   │   reshape → (B, 3, 64, 2)       每层每单元一组 (γ,β) │
   └──────────────────────────────────────────────────────┘
        │  (γ,β) 调制系数
        ▼
   ┌──────────────────────────────────────────────────┐
   │ SIREN decoder  Φ(τ; z)          L=3, H=64, ω0=15 │
   │   u = 2τ − 1  ∈ [−1,1]                           │
   │   h = sin( ω0·( W·h·(1+γ) + b + β ) )   ← FiLM   │
   │   out:  Linear(64 → 2) = (x, y)                  │
   └──────────────────────────────────────────────────┘
        │
        ├──►  位置  = Φ(τ)                     （任意 τ → 任意 fps）
        └──►  速度  = ∂Φ/∂τ / (N−1)            （autograd, 逐样本）
```

**谁是 hypernetwork、谁被调制：**
- **Encoder + to_film** = hypernetwork：读观测（16 个带噪点）→ 吐调制系数 (γ,β)。
- **SIREN** = 被调制的解码器，权重 `W_i, b_i` 是**所有轨迹共享的运动先验骨架**；(γ,β) 是**实例相关的形变**。
- 对应 NIAF Eq.7：`Ŵ=W⊙(1+γ)`（γ 缩放频率）、`b̂=b+β`（β 移相位）。区别：NIAF 用 MLLM 当 hypernetwork，这里用小 MLP encoder。

**SIREN 配置**：3 层、隐藏宽 64、ω0=15（标准 SIREN init：首层 `U(−1/d, 1/d)`，其余 `U(−√(6/d)/ω0, +)`）。ω0=15 针对轨迹分布频率(0.5–3 cycles)调；per-clip 版用 ω0=30 就因太高炸过。

---

## 2. 整体 Pipeline：learned-INR 嵌入 DisPose

learned-INR 插在 `pose2track()` 之后、`points_to_flows()`+高斯之前，**替换差分+高斯两步**，其余 DisPose 结构保留。带代码位置，可当实现蓝图。

```
驱动视频 frames
│  DWPose → pose2track()  [utils.py:115]
▼
P_traj (18, N, 2)   ← 带检测噪声
│
▼
┌─────────────────────────────────────────────┐
│ ★ NEW  Learned-INR 模块   (结构见上节)      │
│   Encoder → FiLM(γ,β) → 共享 SIREN  Φ(τ; z) │
└─────────────────────────────────────────────┘
│
│  〔替换〕points_to_flows() 差分   [utils.py:173]
│  〔替换〕高斯滤波                 [inference_ctrl.py:105-108]
│
├─①─►  Φ(τ) 去噪连续轨迹 (任意 fps) ─►  渲染 skeleton 图 ─►  PoseNet [pose_net.py] ─►  pose_latents
│
└─②─►  ∂Φ/∂τ 解析稀疏运动场 ─►  get_sparse_flow() [utils.py:72] ─►  CMP [utils.py:16] ─►  稠密 F_d
                                                              ─►  ControlNet 条件嵌入 [controlnet.py]
                                                                   (flow 编码 F_d + traj 编码 速度)

参考图 ref.png ─►  DWPose + DIFT (SDFeaturizer) ─►  PointAdapter [point_adapter.py]

冻结 UNet 的三路控制输入（左对齐汇入）:
   ┌─  pose_latents        （来自 ①, PoseNet）
   ├─  F_d + 稀疏速度       （来自 ②, ControlNet 条件嵌入）
   └─  point 特征          （来自 参考图, PointAdapter / DIFT）
          │
          ▼
   ┌──────────────────────────────────────────────────┐
   │ ControlNetSVD  →  冻结 UNet   [pipeline_ctrl.py] │
   └──────────────────────────────────────────────────┘
          │
          ▼
   生成视频 (RGB)
   评测:  VBench Motion Smoothness / Temporal Flickering  +  FID-FVD
```

### 设计要点
1. **插入位置**：`pose2track()` 拿到带噪 `P_traj` 之后。吃带噪离散轨迹，吐两路：
   - **去噪连续轨迹 Φ(τ)** → 渲染 skeleton 图喂 PoseNet；任意 τ 采样 = **任意 fps** 能力。
   - **解析速度 ∂Φ/∂τ** → 当稀疏运动场，走原 `get_sparse_flow → CMP → F_d` 链。
2. **替换 vs 保留**：
   - ╳ 替换：`points_to_flows()` 差分、`inference_ctrl.py:105-108` 高斯滤波。
   - ✅ 保留：CMP、ControlNet 嵌入、PointAdapter(DIFT)、PoseNet、冻结 UNet（最小侵入，符合即插即用哲学）。
3. **训练范围**：只训 learned-INR（encoder+FiLM），SIREN 骨架共享、UNet 冻结。监督 = 高帧率/干净 pose 当 GT 位置 + 解析速度（NIAF 的 L_pos+L_vel）。

### 两个待定设计选择
- **Encoder 用多强**：纯轨迹小 MLP（轻，只看几何）vs 接入参考图外观特征 / 更大 backbone（重，更像 NIAF，可能更稳）。
- **关键点是否联合建模**：每个关键点独立 SIREN，还是 18 个关键点共享 latent 联合解码（利用骨架结构相关性，更鲁棒）。

---

## 3. 实验 A：最小验证（per-clip 测试时拟合，负面结果）

证明"无先验的 per-clip SIREN"不行。代码 `outputs/siren_idea/siren_experiment.py`。

### 设计
- 合成一条 2D 关键点轨迹（挥手式），有**已知闭式 GT 速度**（真实 DWPose 无 GT，无法量化对错）。
- N=16 帧（DisPose 采样数）；加 4 档检测抖动 σ=3/6/12/20 px。
- 对比：`finite-diff`（DisPose `points_to_flows`）、`finite-diff + Gaussian`（DisPose 稳定化）、`SIREN 测试时拟合`（扫 ω0 与 jerk 正则 λ 取最优，解析求导）。

### 结果（速度对 GT 的 MSE，越低越好）

| σ(px) | finite-diff | fd+Gauss | SIREN(best λ) | SIREN 赢？ |
|---:|---:|---:|---:|:--:|
| 3 | **26.9** | 224 | 28.3 | no |
| 6 | **63.4** | 229 | 109 | no |
| 12 | **203** | 243 | 463 | no |
| 20 | 529 | **269** | 904 | no |

### 三个发现
1. **测试时 SIREN 最优 λ 几乎总是 0** → 它只是**光滑地插值噪声**，跟着噪声点过冲。per-clip 无先验，分不开噪声和真实运动。
2. **fd+高斯又便宜又强**：跨噪声几乎不退化（224→269），σ=20 反而最优 → 解释了 DisPose 为何用高斯滤波。
3. 插曲：首跑 SIREN 默认 **ω0=30**，velMSE 爆到 **14727**（16 稀疏点上频率太高，点间剧烈振荡），压到 ω0≈5 才正常。

### 图
- `outputs/siren_idea/figA_velocity_highjitter.png` — σ=12 三方法速度曲线 vs GT
- `outputs/siren_idea/figB_crossover.png` — velMSE vs 噪声 σ 交叉曲线
- `outputs/siren_idea/figC_fps.png` — 16→64 任意帧率重采样（SIREN 光滑但跟噪声）

---

## 4. 实验 B：学习版（摊销先验，决定性正面结果 ✅）

针对实验 A 的负面结果，测试"学习版（摊销先验）能否真赢"。代码 `outputs/siren_idea/learned_experiment.py`。

### 设计（为"赢得可复现"）
- **轨迹分布**：平滑 band-limited 2D 运动，随机幅度/频率(0.5–3.0)/相位，有解析 GT 速度；train/test 严格分开。
- **学习模型** = §1 的 FiLM-调制 SIREN：读 16 个带噪点 → (γ,β) → 共享 SIREN；用**干净 GT 位置 + GT 速度**监督（NIAF L_pos+L_vel）。带噪输入+干净目标 ⇒ 学会去噪。训练：Adam lr 1e-3 + cosine，400 epochs，batch 256，2000 train。
- **公平性**：baseline 的 fd+高斯平滑 σ 在测试集上扫到**最优**（最强 baseline）；2000 train / 200 test held-out；3 seeds 报 mean±std。

### 结果（速度对 GT 的 MSE，越低越好）

| σ(px) | finite-diff | fd+Gauss(best σ) | **learned-INR** | 赢家 | 领先倍数 |
|---:|---:|---:|---:|:--:|:--:|
| 3 | 78.6±3.9 | 266.6±11.8 | **29.5±0.4** | LEARNED | 2.7× |
| 6 | 107.3±4.3 | 276.1±11.9 | **44.3±1.6** | LEARNED | 2.4× |
| 12 | 223.0±5.9 | 315.5±12.0 | **96.2±1.6** | LEARNED | 2.3× |
| 20 | 497.8±9.4 | 410.4±12.2 | **189.6±8.0** | LEARNED | 2.2× |

**全部 4 档噪声完胜，领先最强 baseline 2.2–2.7×，方差极小。** 定性图 `outputs/siren_idea/figD_learned.png`：learned-INR 紧贴 GT，finite-diff 尖刺、fd+高斯过平滑且滞后。

### 关键修正（踩过的坑）
- 首版学习模型 velMSE 爆到 ~8000。根因：`velocity()` 对**全 batch 共享的 τ** 求 `grad(pos.sum())`，autograd 把导数**在 batch 上求和**再广播 → 每条轨迹拿同一条速度，训练目标是垃圾。
- 修法：每样本**独立 τ leaf** `(B,T)`，因 `pos[b,j]` 只依赖 `tau[b,j]`，梯度即逐样本逐时刻导数。修后 loss_vel 收敛（5609→107）。

### 含义
- 实验 A 不是"idea 死了"，是"无先验 per-clip 版该死"；实验 B 证明**有先验的学习版真能赢**。
- 也回答了"training-free"之问：per-clip 版 training-free（无先验）→ 注定输；**学习版必须预训练**，这正是它赢的原因。
- ⚠️ 这是**合成 proof-of-concept**，证明机制成立；**还不是 paper 终局**——真正要赢的是真实 DWPose 轨迹 + 生成视频 VBench/FID-FVD。

---

## 5. DisPose 代码替换点（实现参考）

| 对象 | 位置 |
|---|---|
| 离散轨迹 `P_traj (18,N,2)`（learned-INR 输入） | `mimicmotion/utils/utils.py:115` `pose2track()` |
| 有限差分运动 `(end-start)`（★替换） | `mimicmotion/utils/utils.py:173` `points_to_flows()` |
| 参考帧相对流 `(poses-poses[:,0:1])` | `mimicmotion/utils/utils.py:72` `get_sparse_flow()` |
| 高斯滤波 hack（★替换） | `inference_ctrl.py:105-108` |
| CMP 稀疏→稠密 | `mimicmotion/utils/utils.py:16` `get_cmp_flow()` + `modules/cmp_model.py` |
| Motion/Point 编码注入 | `modules/controlnet.py`（`ControlNetConditioningEmbeddingSVD`）、`modules/point_adapter.py` |
| 主推理入口 | `inference_ctrl.py` `preprocess()` (45-148) / `run_pipeline()` |

---

## 6. 下一步（真实数据 / paper 主战场）
1. **真实轨迹验证**：集群上从真实 DWPose `P_traj` 提轨迹，同协议测 learned-INR vs fd+高斯（无 GT 速度时用 held-out 帧重建 / 高帧率视频差分作伪 GT）。
2. **接入 DisPose + 视频指标**：learned-INR 去噪/解析速度场接到 `controlnet` motion 分支，比 VBench Motion Smoothness / Temporal Flickering + FID-FVD。
3. **任意 fps 能力 demo**：连续 pose → 高帧率视频生成（baseline 缺席的能力）。
