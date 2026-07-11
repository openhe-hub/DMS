# Step 2 — Video-level 真实验:低 fps 驱动下的连续 pose 控制

> 主 claim 从"去噪"(已被 step1 + video probe 判死)切换为**"pose 控制时域超分"**:
> 低帧率驱动 pose → 连续轨迹表示重建全帧率控制信号 → 全帧率视频,vs 离散管线结构性退化。
> 计划见 [`../../refine-logs/EXPERIMENT_PLAN.md`](../../refine-logs/EXPERIMENT_PLAN.md),
> 运行记录见 [`../../refine-logs/EXPERIMENT_TRACKER.md`](../../refine-logs/EXPERIMENT_TRACKER.md)。
> 日期:2026-07。环境:Jubail A100。

---

## 1. R010 — pose-level 插值快筛(任务换成插值后,谁是最好的连续表示?)

**协议**:真实 DWPose 轨迹(video1/2/3),窗口 span=15·s+1,观测=16 个 stride-s 检测(含两端),
held-out=全部中间真实检测;比 held-out 位置 MSE(px²)。检测噪声地板对所有方法相同 → 排序公平。
perclip 配置(ω0, λ)只在 video1(dev)上调,video2+3(test)冻结;gauss σ 在测试集上调到最优(baseline 最优待遇)。

**结果(test = video2+3)**:

| stride | linear | spline | gauss+lin | perclip-SIREN | amortized |
|---|---|---|---|---|---|
| 2 | **37.78** | 38.82 (×1.03) | 47.20 | 48.60 (×1.29) | 59.34 (×1.57) |
| 4 | 71.99 | **54.68 (×0.76)** | 138.70 | 68.42 (×0.95) | 232.19 (×3.23) |
| 8 | 298.56 | **279.55 (×0.94)** | 500.42 | 353.93 (×1.19) | 654.45 (×2.19) |

**判读**:
1. **spline 是 pose-level 插值赢家**(s4 领先 linear 24%,s8 领先 6%);s2 时 linear 已近最优(运动局部线性)。
2. **per-clip SIREN 全程输给 spline**(最优 ω0=3、λ=0 ≈ 低频先验,行为像"糙版 spline")。
   "INR 必要性" anti-claim 在 pose-level 已经成立一半——video-level 是最后战场。
3. **amortized(合成先验)再次惨败**,与 step1 结论一致,正式出局。
4. 插值任务 ≠ 去噪任务:step1 里"per-clip 必死"的结论不适用于插值(观测干净,穿过观测点正是所需),
   所以 per-clip 方法在此复活参赛——但没赢过 spline。

## 2. R001 — 集成等价性 gate(stride=1 逐位一致)

把 `inference_ctrl.preprocess` 拆出 `build_control()`(DIFT→traj_flow→CMP 顺序保持),
step2 路径 = `detect_video_full`(单次 DWPose)→ 抽帧 → 插值(`interp_pose_dicts`)→ 重画 skeleton → `build_control`。

- **GATE A(阻塞,PASS ✅)**:同一份检测,direct 构建 vs interp@stride1 构建,
  candidate / 画出的 skeleton / traj_flow / sparse_flow / mask 全部 **bit-exact**(video1、video2)。
- **GATE B(信息)**:独立重跑检测器,可见关键点 maxdiff ~2e-3(归一化),不可见关键点最高 0.28,
  可见性翻转 0/2034 → **ONNX-GPU 检测非确定性**,与集成无关;pilot 内所有系统共享单次检测,配对性不受影响。
- 设计决定:观测帧**原样保留原始检测坐标**(含不可见关键点的垃圾值,因 DisPose 的 sparse-flow 分支不看 subset)
  → 系统间唯一差异 = 帧间重建方式;video3 出现双人检测帧 → 统一 person-0 截断(与 `pose2track` 的行为一致)。

**副产物发现(DisPose 上游的两个可复现性问题,gate 逼出来的)**:
1. **DWPose ONNX-GPU 非确定性**:同一视频独立重检测,可见关键点坐标漂移 ~2e-3(归一化),
   不可见关键点垃圾坐标最高漂移 0.28。→ 任何跨 run 的"配对"比较都必须共享同一次检测。
2. **sparse-flow scatter 竞态**:`pose2track` 把不可见关键点放在 (0,0),`sample_optical_flow`
   的花式索引赋值允许重复索引(多个不可见点写同一像素,写入顺序未定义)→ **相同输入、
   不可复现输出**(video3 上实测同输入 maxdiff 达 676 px-flow)。gate 因此改为比较
   `get_sparse_flow` 的确定性输入(`pose2track` 输出)+ 竞态输出仅报告。生成实验不受影响
   (每个系统单独一次抽签,原版 DisPose 自身也带这个噪声)。

## 3. R011 — Pilot:自驱动低 fps 生成(video-level 生死局)

**协议**:ref 图 = 驱动视频首帧(自驱动)→ 原视频帧 = 像素级 GT。
系统(共享单次检测、单次 DIFT、同 diffusion seed,完全配对):

| 系统 | 控制信号 | 输出 |
|---|---|---|
| orig:1 | 全帧检测,原版管线 | 全帧率(上界/原版 DisPose) |
| orig:4 / orig:8 | 每 4/8 帧检测,原版管线 | 低帧率(短视频) |
| + RIFE | orig:4/8 输出 ×4/×8 插帧(Practical-RIFE v4.8) | 全帧率(后处理 anti-claim) |
| linear / spline / siren :4/:8 | 抽帧 → 连续重建全帧率关键点 → 重画 skeleton + 重建 motion field | 全帧率 |

指标:PSNR(all / obs / **mid**)、LPIPS、warp_error(输出时序平滑)、div_ub(vs orig:1)。
**obs/mid 拆分是关键诊断**:插值质量只体现在 mid 帧;obs 帧衡量共同的扩散上限。

**结果**(jobs 16485807/16485808/16485927 + metrics 16486552;PSNR↑ / LPIPS↓ / warp↓):

case0 = video1(orig:1 上限 17.10dB)/ case1 = video2(20.00)/ case2 = video3(13.60)。
mid = 只看插值出的中间帧(判决帧)。

| 系统 | c0 mid s4/s8 | c1 mid s4/s8 | c2 mid s4/s8 | LPIPS c1 s4/s8 |
|---|---|---|---|---|
| linear | 16.07 / 15.32 | 19.18 / 18.26 | 13.22 / 12.89 | 0.176 / 0.200 |
| spline | 16.04 / 15.28 | 19.24 / 18.24 | 13.24 / 12.91 | 0.176 / 0.201 |
| siren | 15.97 / 15.23 | 19.10 / 18.15 | 13.21 / 12.88 | 0.180 / 0.202 |
| **RIFE 后处理** | **16.08 / 15.60** | **19.26 / 18.43** | **13.40 / 12.94** | 0.180 / 0.200 |

Sanity(实验自洽性):连续系统 obs 帧 ≈ orig:1 上限(±0.05dB);RIFE 的 obs 帧=粗生成原帧,
PSNR 与 orig_s4/s8 逐位一致(17.13=17.13 等)→ 管线无 bug。粗网格 orig:4/8 作为视频 warp 爆炸
(case1: 693→1449;case2: 2049→5281)→ 低fps输出确实时序崩坏,但赢家是后处理修复而非控制侧。

## 4. R012 — 生死判定:**方向终止 ❌**

按预注册 gate 逐条:
1. **离散管线在低fps下退化?** 是(warp 爆炸)——但见 gate 2,修复它的最优解不在控制侧。
2. **连续控制 vs RIFE 后处理(生死线)?** **RIFE 胜**:mid-PSNR 六组对比(3 case × 2 stride)
   RIFE 全部 ≥ 最好的连续控制(+0.00~+0.31dB,方向完全一致);LPIPS 打平(各有 ~0.003 微弱胜负);
   warp RIFE 大幅占优(该指标结构性偏爱插帧,不单独作数——但没有任何一个指标显示连续控制占优)。
   预注册规则"RIFE 追平即终止",实际 RIFE 小胜。
3. **spline vs siren vs linear?** 全部打平(<0.1dB),**siren 三者一致垫底**。
   pose-level 上 spline 24% 的领先在 video-level 完全蒸发;INR novelty 死,连"连续 vs 线性"都无差异。

**机制结论(为什么控制侧改进注定是二阶小量)**:扩散生成即使在控制信号完美的 obs 帧,
重建也只有 13.6–20.0dB——**扩散自身的控制跟随/外观误差 >> 控制信号的插值误差**;
控制侧的任何精化(去噪、插值、解析速度)都被这层误差淹没。RIFE 则直接在扩散自己的输出帧
之间插值,贴着生成分布走。这与 video probe 的低敏感度(σ=32 抖动只挪动像素方差 2.4%)
和 step1 的"真实抖动只有 2–3px"共同构成闭环证据:
**pose-guided 扩散动画的质量瓶颈在扩散侧,不在 pose 控制信号的时间/噪声精度。**

定性核验(防指标盲区,已完成):case2(运动最大)最快 3 个 mid 帧的 GT|linear|RIFE 三联图
(`outputs/step2/fig/qc_case2_t{67,68,115}.png`)目检结论:GT 快动作帧自带真实运动模糊;
linear(连续控制)输出清晰但姿态偏离 GT;RIFE 在快速手臂上有轻微插值涂抹(其已知弱点,可见但不致命,
无断肢/鬼影),且这种涂抹反而更接近 GT 的真实运动模糊。两者各有偏差、无一方占优——
**与指标结论一致,判决无盲区**。

## 5. 残值与转向(诚实评估)

**这个 idea 不再值得投入。** 按预注册 gate 干净利落地死掉,总花费约 20 GPU-h(止损设计生效:
没有跑 M2/M3 的 50+ GPU-h 主表,没有为改训先验去扩数据)。

剩下的可用资产:
1. **负结果洞察(有系统性证据)**:pose-guided 扩散动画中,控制信号的时间/噪声精度是二阶因素,
   瓶颈在扩散的控制跟随。证据链:真实抖动 2–3px + 注噪 σ=32 仅 2.4% 输出方差 + 插值控制
   obs 帧无损但 mid 帧被 RIFE 反超。可作 workshop/短文或下一个 idea 的 motivation,
   独立成顶会正文不够。
2. **DisPose 上游两个可复现性问题**(可提 issue/PR):DWPose ONNX-GPU 非确定性影响配对评测;
   `sample_optical_flow` 重复索引 scatter 竞态(不可见关键点堆在 (0,0))导致相同输入不同控制信号。
3. **可复用基建**:自驱动低fps评测协议 + 配对生成管线(`scripts/round1_archive/step2/`)、批量 per-clip SIREN/样条库
   (`src/dispose_siren/interp.py`)——任何后续"控制信号 vs 后处理"类比较可直接复用。

**若要继续在此方向找 idea,应打扩散侧而非控制侧**(机制结论指向的真瓶颈):
例如提升 ControlNet 的控制跟随保真度、或把插帧先验搬进扩散采样过程(生成时保证时序一致,
而不是生成后修)。这是新 idea,需要重新走 refine/查新流程。

## 文件
- 库:`src/dispose_siren/interp.py`(自然三次样条 + 批量 per-clip SIREN)、`scripts/round1_archive/step2/lib_lowfps.py`
- 脚本:`scripts/round1_archive/step2/`、`scripts/round1_archive/slurm/`
- 产物(集群):`outputs/step2/{fig,pilot}/`
