# Step3 查新 — 采样时 latent 帧间融合 × pose 驱动低 fps 生成(R100)

> 日期 2026-07-03。约 20 组 arXiv 检索,覆盖:diffusion 插帧、zero-shot 时序一致性、
> pose×帧率、latent 插值、光流引导采样、TokenFlow/FRESCO 系、生成式 inbetweening、
> noise-warping、time-reversal、稀疏控制、human-centric interpolation。

## 结论(先说)

**真空地带仍存在,但很窄。** 无人同时做到:pose-guided 人体动画模型 + 低帧率 pose 输入
+ 单次去噪生成全帧率 + training-free flow-warp latent 软约束。但三个组成部分各有强先验,
贡献口径必须精确切割(见文末)。

## 危险先验(按威胁度)

### T1 — 威胁 "pose-guided 人体时域超分"
1. **PoseFuse3D-KI** (2506.03119) ⚠️ 最危险:人体关键帧插值 + SMPL-X/2D pose 控制编码进
   diffusion,自带 CHKI-Video 基准。**但**:training-based、输入真实 RGB 关键帧对;
   我们 training-free、关键帧亦由同一模型生成、低 fps pose 单独驱动。
   → 杀死 "pose-guided human inbetweening" 表述;其 benchmark 需引用/讨论。

### T2 — 威胁 "低帧率条件 → 全帧率生成"
2. **SparseCtrl** (2311.16933):时域稀疏条件(sketch/depth/RGB)+ 训练附加编码器。
   → 杀死 "首个时域稀疏控制"。
3. **KeyFace** (2503.01715):音频驱动人脸,低fps关键帧→训练式插值扩散。pipeline 模式先例。
4. **VHOI** (2512.09646):稀疏轨迹→稠密 HOI 控制,微调式。叙事拥挤提示。

### T3 — 威胁 "training-free 采样干预式插帧"
5. **TRF / Explorative Inbetweening** (2403.14611):SVD training-free 双向去噪融合。
6. **ViBiDSampler** (2410.05651) + **MPD** (2602.12679):双向顺序采样,流派仍活跃(2026.02)。
7. **DiffuseSlide** (2506.01454) ⚠️ 概念最接近:低fps RGB 视频→training-free noise
   re-injection + 滑窗 latent 去噪→高fps。**但输入是 RGB 视频而非控制信号,无 pose。**
8. **ZeroSmooth** (2406.00908):training-free 自级联 + hidden state correction 升帧。
   → 5–8 合计杀死 "首个 training-free 生成式升帧"。

### T4 — 机制先验(flow-warp latent blend 不新)
9. **DiffIR2VR-Zero** (2407.01519)、Rerender-A-Video (2306.07954)、FLATTEN (2310.05922)、
   TokenFlow (2307.10373)、FRESCO (2403.12962)、FGDVI (2311.15368):去噪中 flow-warp 邻帧
   latent/特征再融合 = 编辑/修复流派标准工具箱。
   → 机制只能作为"针对本任务的适配设计"(warp 来源=控制关键点、调度、软约束强度)。
10. **LumosFlow** (2506.02497):关键帧 + latent 光流扩散 + warp 精修(训练式)。必引必比。

其他确认非致命:Generative Inbetweening (2408.15239)、MultiCOIN (2510.08561)、
FC-VFI (2603.04899) —— 均训练式。
**检索面 "pose-guided × frame rate / human animation × temporal super-resolution"
在 arXiv 零命中** —— 确切组合无人占位。

## 贡献口径切割(写作红线)

1. 任务定位:**"控制信号时域稀疏下的条件视频生成"**,不是"插帧/升帧"。卖点 = 中间帧
   无控制信号时,冻结的条件模型如何保持可控与连贯 —— SparseCtrl(要训练)与全部
   inbetweening 工作(无条件模型)都不覆盖的组合。
2. 绝不声称:首个 training-free 升帧 / 首个稀疏控制 / 首个 pose 人体插值 /
   flow-warp latent blend 机制本身。
3. **隐藏 baseline 已内置**:审稿人第一反应"插值 pose 再喂稠密控制"= R011 的 linear:s,
   我们已有其完整数据(且已知被 RIFE 小胜)→ 若 fusion 同时赢 linear 和 RIFE,
   正好构成三方完整论证。
4. 必比/必讨论:RIFE(已有)、pose 插值(已有)、TRF/ViBiDSampler(套关键帧输出,
   候补)、PoseFuse3D-KI 的 CHKI-Video(至少讨论)。主打大运动子集指标差。
