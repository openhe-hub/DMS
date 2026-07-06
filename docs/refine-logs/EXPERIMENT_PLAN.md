# Experiment Plan — SIREN×DisPose Video-Level 真实验

**Problem**: DisPose 的 pose 控制信号(motion field)由离散 DWPose 关键点经有限差分 + 高斯滤波构造,被锁死在驱动视频的帧网格上;低 fps / 大 sample_stride 驱动下差分粗糙、控制退化,且无法生成高于驱动帧率的视频。
**Method Thesis**: 用连续可微轨迹表示(INR/SIREN,或退化为 spline)替换差分+高斯,把 pose 控制从帧网格解放:任意时刻可采位置 Φ(τ) 与解析速度 ∂Φ/∂τ,实现 **pose 控制时域超分**(低 fps 驱动 → 全 fps 视频)并提升输出时序平滑。
**Date**: 2026-07-02

---

## 前情摘要(为何主战场在 video-level)

- pose-level 去噪已被自己的实验否掉:真实 DWPose 抖动仅 2–3px(relative≈0.12),learned-INR 输给线性插值 2.7×(domain gap),真实改训撞数据墙(`docs/experiments/step1_real_validation.md`)。
- Video probe(job 结果 `step1_video_probe.json`):motion-field 分支**传导到输出视频**(warp_error 随注入抖动单调 +10.6% @ σ=32),但敏感度低(rel_div 2.4%),线性外推到真实噪声 σ≈3 只有 ~1% 变化 → **"去噪"claim 在 video-level 也没有头寸,必须换打法**。
- 新打法:打 baseline **结构性缺席/结构性退化**的场景 —— 低 fps 驱动与任意帧率生成。stride 越大,有限差分越粗、离散 skeleton 越稀,连续表示的优势才是结构性的,不靠噪声头寸。

---

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|-----------------|-----------------------------|---------------|
| **C1(主)**: 连续轨迹控制使 DisPose 在低 fps/大 stride 驱动下不退化,并能从低 fps pose 生成全 fps 视频(时域超分) | baseline 结构性缺席的能力;卖点不依赖噪声头寸;自驱动协议有 GT,证据硬 | 自驱动重建:stride∈{4,8} 时连续控制 vs 离散 baseline 在 PSNR/LPIPS/FVD + VBench Motion Smoothness/Temporal Flickering 上显著赢,**且赢过后处理插帧(RIFE)** | B1, B4 |
| **C2(辅)**: 解析速度场替换 fd+Gauss 在正常 fps 下也带来时序平滑收益 | 支撑"控制信号质量→视频质量"叙事;probe 已证通路存在 | stride=1 时 VBench MS/TF 小幅但一致的改善(≥3 视频、固定 seed 配对比较) | B1(stride=1 列) |

**Anti-claims(必须排除)**:
1. "收益 = 后处理插帧就能拿到" → RIFE/IFRNet 后处理 baseline(B4)。**这是最危险的一条,pilot 就测。**
2. "收益 = 任意平滑插值就能拿到,INR 不必要" → cubic spline / linear 关键点插值 baseline(B2)。若 spline≈INR,贡献口径退为"连续时间 pose 控制",INR 只是实现选择——口径两手准备。
3. "收益 = 更强平滑" → 调优高斯 σ 的离散 baseline(B2 内)。

---

## Paper Storyline

- **Main paper 必须证明**:低 fps 驱动下离散管线退化曲线 vs 连续控制平坦曲线(主图);时域超分主表(vs GT 全 fps);打赢后处理插帧;stride=1 无损(即插即用不伤原性能)。
- **Appendix 支撑**:参数化消融(INR vs spline vs linear)、per-clip vs amortized、运动速度分层分析、seeds 方差。
- **有意砍掉**:pose-level 去噪叙事(已证无头寸,只在 motivation 提一句 probe 结论);跨模型泛化(AnimateAnyone 等,时间不够);用户研究。

---

## Experiment Blocks

### Block 1: 主结果 — 低 fps 驱动时域超分(自驱动重建)
- **Claim tested**: C1(+C2 的 stride=1 列)
- **Why**: 这是 baseline 结构性缺席的战场,且自驱动协议有像素级 GT,证据最硬。
- **Dataset / split / task**: TikTok test set(本领域标准,DisPose 原文评测集,~10 段)。自驱动协议:参考图 = 首帧,驱动 = 同视频以 stride s∈{1,2,4,8} 抽帧的 pose;各方法生成**全 fps** 视频(离散 baseline 只能生成 N/s 帧,以 (a) 原样低 fps、(b)+RIFE 上采样两种形态参赛);与 GT 全 fps 视频逐帧比。
- **Compared systems**(≤3 个 baseline family):
  1. **DisPose 原版**(fd+Gauss,stride-s 网格上跑,含调优高斯 σ)± RIFE 后处理
  2. **连续控制-简单参数化**:linear / cubic spline 关键点插值 → 渲 skeleton + 差分速度
  3. **连续控制-INR**(本方法):Φ(τ) 渲 skeleton + ∂Φ/∂τ 解析速度 → CMP
- **Metrics**: 决定性 = PSNR/SSIM/LPIPS(vs GT 逐帧)、FVD;次要 = VBench Motion Smoothness、Temporal Flickering、warp_error(已有实现 `07_video_metrics.py`)。
- **Setup**: Jubail A100;UNet/ControlNet/CMP/PointAdapter 全冻结(只动控制信号构造);固定 diffusion seed 逐视频配对比较;分辨率/步数沿用 `configs/test.yaml`。
- **Success criterion**: stride≥4 时 INR(或 spline)在 LPIPS/FVD/MS 上显著优于「原版+RIFE」;stride=1 时不差于原版。
- **Failure interpretation**: 若连续控制在 stride≥4 无优势 → 扩散模型对控制信号时间分辨率不敏感,方向终止;若只有 spline≈INR 都赢 → 贡献口径改为"连续时间控制信号"。
- **Table / figure target**: 主表(Tab.1)+ 退化曲线主图(Fig.3:质量 vs stride,离散陡降、连续平坦)。
- **Priority**: MUST-RUN

### Block 2: Novelty isolation — 连续参数化消融
- **Claim tested**: C1 中"INR 是否必要"(anti-claim 2、3)
- **Why**: 审稿人第一问就是"为什么不用 spline"。pose-level 结果已暗示简单插值很强,必须正面回答。
- **Dataset**: B1 的子集(3–5 段,stride 4/8)。
- **Compared systems**: linear / cubic spline / per-clip SIREN(jerk 正则)/ amortized FiLM-SIREN;速度来源消融:解析 ∂Φ/∂τ vs 对插值后位置再差分。
- **Metrics**: 同 B1。
- **Success criterion**: INR 在快运动/大位移窗口上相对 spline 有可测优势(哪怕整体接近);或解析速度相对差分速度有一致收益。
- **Failure interpretation**: spline 全面持平 → 论文改口径:方法 = 连续时间 pose 控制框架,INR 作为其中一种实现(同时给 spline 数字,诚实);SIREN 卖点退为单模块同时给出任意阶导数。
- **Table / figure target**: 消融表(Tab.2)。
- **Priority**: MUST-RUN

### Block 3: Simplicity check — training-free per-clip vs 学习版
- **Claim tested**: 方法最简形态是否够用(避免为已死的 amortized 先验背数据墙)
- **Why**: Step1.5 证明 amortized 先验撞数据墙;若 per-clip 拟合(training-free)在 video-level 够用,方法变成**零训练、即插即用**——巨大简化,直接消解数据墙风险。插值任务(点间补)≠ 去噪任务(点上纠),per-clip 在前者并无 Experiment A 的本质缺陷(真实检测足够干净,平滑穿过观测点正是所需)。
- **Dataset**: 先 pose-level 快筛(held-out 中间帧插值精度,CPU 即可),赢家进 B1/B2 的 video 子集。
- **Compared systems**: per-clip SIREN(ω0、jerk-λ 在 dev 视频上调)vs amortized FiLM-SIREN(现有 ckpt)。
- **Success criterion**: per-clip ≥ amortized → 主方法定为 training-free。
- **Failure interpretation**: 两者都不如 spline → 与 B2 失败分支合并处理。
- **Table / figure target**: appendix 表。
- **Priority**: MUST-RUN(便宜,先行)

### Block 4: Necessity check — 打后处理插帧
- **Claim tested**: anti-claim 1("直接对输出视频插帧就行,不需要你")
- **Why**: 这是本方向的生死线:若 baseline 低 fps 输出 + RIFE 追平连续控制的全 fps 输出,整个"控制侧连续化"就是不必要的。README 自己都推荐 IFRNet 后处理,审稿人必问。
- **Dataset**: 与 B1 同(RIFE 系统直接并入 B1 主表);pilot 阶段即测,并刻意包含快速运动片段(大位移是插帧法的已知弱点:鬼影/断肢)。
- **Compared systems**: 原版@stride-s + RIFE×s vs 连续控制@全fps。
- **Metrics**: 同 B1 + 快运动子集单列。
- **Success criterion**: 连续控制在快运动子集上明确赢(整体至少持平)。
- **Failure interpretation**: RIFE 全面追平 → **方向终止**,不要再投入;剩余可挖的只有"驱动视频本身低fps时 DWPose 检测都在低fps网格上"这类极端场景,价值有限。
- **Table / figure target**: 并入主表 + 快运动对比图(Fig.4,定性鬼影对比)。
- **Priority**: MUST-RUN(pilot 阶段前置)

### Block 5: Failure analysis / 定性诊断
- **Claim tested**: 无(诊断)
- **Why**: 明确方法边界:收益随运动速度/stride 的分层;失败样例(遮挡、检测缺点时 INR 外推会不会飞)。
- **Dataset**: B1 全部输出的事后分析 + 挑选片段。
- **Metrics**: 按窗口运动幅度分桶的 LPIPS/MS;定性网格图;60fps demo 视频(任意帧率能力展示,τ 连续采样)。
- **Priority**: NICE-TO-HAVE(appendix + demo 页)

---

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| **M0 Sanity** | 集成 hook:stride 抽帧 + 连续控制注入(复用 06 的 preprocess 改造);stride=1 下"插值版控制"与原版逐像素等价性;VBench + RIFE 环境装通 | 2–3 个短视频生成 | stride=1 差异 ≲ seed 间噪声,否则先修集成 | ~4 GPU-h | 集成 bug 污染全部下游结论 |
| **M1 Pilot(生死局)** | 3 段自带示例视频 × stride{1,4,8} × {原版, 原版+RIFE, linear, spline, per-clip INR} | ~45 个生成 | ① stride≥4 连续 vs 离散有明显差距?无 → **方向终止**。② 连续 vs RIFE 谁赢?③ INR vs spline 定贡献口径。④ B3 快筛定 per-clip/amortized | ~12 GPU-h | 最大信息量/最小成本;三个 anti-claim 在此全部初判 |
| **M2 Data** | TikTok test set 下载/预处理/DWPose 抽轨迹到集群 | CPU/传输 | 拿不到 → 换 UBC-Fashion 或自采 10–20 段 | ~0 GPU-h,1–2 天 | 数据可得性 |
| **M3 Main table** | 10 段 × stride{1,2,4,8} × 胜出的 3–4 系统 | ~120–160 个生成 | 主表成立?→ 写作启动 | ~50 GPU-h | 墙钟时间(排队+逐段生成) |
| **M4 Ablations** | B2 全套 + 3 seeds 方差子集 + B5 分桶分析 | ~40 个生成 | — | ~15 GPU-h | — |
| **M5 Polish** | 60fps demo、定性图、appendix | ~10 个生成 | — | ~5 GPU-h | — |

**Must-run** = M0–M4;**Nice-to-have** = M5 及 B5。

## Compute and Data Budget

- **总 GPU-hours(A100)**:约 85–95(单段全管线 ~20–25 min,含 DWPose/CMP/扩散;40 帧 pilot 短跑更便宜)
- **数据准备**:TikTok test set(~10 段)获取 + rsync 上集群;RIFE/IFRNet 权重;VBench 安装(依赖较重,装在独立 env,防污染 dispose env)
- **人评**:不做(用 VBench + 像素 GT 指标替代)
- **最大瓶颈**:M3 的墙钟时间(Slurm 排队 × ~150 段生成)→ 用 job array 并行 + `max_frames` 截断到统一长度

## Risks and Mitigations

- **[R1] spline ≈ INR(pose-level 已有此征兆)** → pilot 先判;口径两手准备:赢则"INR 控制信号",平则"连续时间 pose 控制框架 + INR 单模块解析导数";论文诚实报 spline 数字。
- **[R2] RIFE 后处理追平(生死线)** → pilot 前置到 M1;刻意纳入快运动/大位移片段;若仍追平,方向终止,及时止损(此时总投入仅 ~16 GPU-h)。
- **[R3] 扩散对控制信号时间分辨率不敏感(probe 的低敏感度可能预示)** → M1 的 stride 退化曲线直接回答;若原版在 stride=8 都不怎么退化,C1 无从谈起 → 终止。probe 的对立面证据:warp_error 单调性说明控制确实影响输出时序质量,存活概率中等偏上。
- **[R4] TikTok 数据拿不到** → UBC-Fashion 备选;再不行自采(YouTube 舞蹈,注意版权只做指标不放 demo)。
- **[R5] CMP 在全 fps 时间网格上算稀疏→稠密流,帧数 ×s 倍,预处理变慢** → CMP 是轻量前向,预算里已含;必要时对 flow 分支半分辨率。

## Final Checklist
- [ ] Main paper tables are covered(B1 主表 + 退化曲线)
- [ ] Novelty is isolated(B2:INR vs spline vs linear;解析速度 vs 差分)
- [ ] Simplicity is defended(B3:training-free per-clip 优先)
- [ ] Frontier contribution is justified or explicitly not claimed(INR 必要性由 B2 裁决,口径两手准备)
- [ ] Nice-to-have runs are separated from must-run runs(M5/B5 与 M0–M4 分离)

---

# Step 3 — 采样时 latent 帧间融合(打 RIFE 软肋)【预注册 2026-07-03】

> 前提:step2 已判死"控制侧连续化",机制结论 = 瓶颈在扩散控制跟随;RIFE 唯一可见弱点 =
> 快运动处插值涂抹(R012 定性检查,case2 手臂)。本 step 打法换到**采样侧**。

**Method Thesis**: 低 fps 驱动下,中间帧不再事后插值(RIFE),而是由扩散模型生成
(控制 = linear 插值 pose,R012 已证方法无关),并在每步去噪后用
"关键帧 latent 按关键点自建后向流 warp 到中间帧" 做软约束(warp-and-blend):
`lat[m] ← (1-α)·lat[m] + α·(w0·warp(lat[k0]) + w1·warp(lat[k1]))`,α 只在中段
去噪窗口生效。中间帧是"画出来的"(扩散先验补全快运动),邻帧约束只负责外观/时序锚定。

**Claim C3**: 生成时融合 > 生成后插帧,差距集中在**大运动 mid 帧**(RIFE 结构性弱点)。

**实现要点**(零侵入):`pipeline_ctrl.py:683` 的 `callback_on_step_end` 允许改写
latents(`[1,F,4,h/8,w/8]`,帧 0 = ref,输出时丢弃 → mid/obs 索引整体 +1);
后向流 m→k0/k1 从插值关键点自建(可见 kp 位移在 m 位置撒点 + 高斯核 Nadaraya-Watson
归一化 → 稠密场,latent 分辨率,无 kp 支撑处 →0 = 背景静止),确定性、不碰 CMP 竞态。

**系统矩阵(pilot)**:复用 R011 全部产物(orig:1 上限、linear:4/8、RIFE:4/8),
新增 fusion:4/8 × 3 case = 6 个生成 + case0/stride8 上 4 个 dev 配置
(α∈{0.15,0.3} × 窗口{[0.3,0.9],[0.0,0.7]}),dev 选定后冻结。

**评测**:23_bucket_metrics —— mid 帧逐帧 PSNR/LPIPS,按 GT 关键点运动幅度
(相邻 GT 帧平均可见 kp 位移)三分桶,报告 slow/mid/fast 桶均值;obs 帧不受融合影响
(callback 不碰 obs)= 内置 sanity。

**预注册 kill-gate(动手前钉死)**:
- **G1(生死)**: stride=8 的 fast 桶上,fusion 的 mid-PSNR **和** LPIPS 均需在 ≥2/3
  case 上优于 RIFE;否则方向终止。
- **G2(伪装检查)**: fusion 的全局 mid 指标不得显著差于 linear(若 α 把画面拉糊,
  全局指标会先暴露)→ 差于 linear 即配置失败,回 dev;dev 两轮内无合格配置 → 终止。
- **G3(定性)**: fast 桶抽 3 帧目检,fusion 不得出现 RIFE 式涂抹或新型伪影(鬼影/断肢)。
- 预算:~10 个生成 ≈ 4–5 GPU-h;一周内判决。
- 依赖查新:采样时 flow-warp latent 约束 × pose 驱动低 fps 是否被占
  (TokenFlow/FRESCO/生成式插帧/噪声 warp 系),查新结果否决则改口径或终止。
