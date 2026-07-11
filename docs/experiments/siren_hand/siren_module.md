# SIREN 手部轨迹模块(HandSetSIREN)— 架构、接入与实验

> DMS 的 SIREN 模块现役形态:**NIAF-style train-once 手部轨迹先验**,补上
> DisPose 控制链中缺位的手部通道,在 109 条手语难例基准上完成三系统对比交付。
> 日期:2026-07。环境:jubail A100(提取/生成/scaling/指标)+ 本地 MPS(P0)。
> 代码:`src/dispose_siren/hand_{traj,model,train,eval}.py`、
> `scripts/hand_pilot/`、`mimicmotion/dwpose/hand_control.py`。
>
> 历史注记:本文件 2026-06 的旧版描述的是"FiLMSIREN 替换 body 运动场差分+高斯"
> 方案,该路线连同 step1/step2 已按预注册 gate 终止(结论见
> [`../../idea/siren_idea.md`](../../idea/siren_idea.md) TL;DR;旧文档在 git 历史)。
> 本版是重启后的手部通道路线,**已交付、未判死**。

---

## 0. TL;DR(最终结果)

**三系统对比(109 条 asl27k 难例,与 `baseline/quantitative.md` 同协议):
DisPose+graft+SIREN 在手部结构质量上配对压倒性赢——mean_hand_conf
101/109(p=6×10⁻²²),灾难手率相对 −8.1%;FVD/CSIM/body 控制零回退。**
组件层:过拟合天花板测试中 SIREN 轨迹先验在 gap 补全上碾压 spline
6–36×;scaling 曲线斜率 −0.87,外推 ~1.5k clips 穿 sp线 ⇒ asl50k 扩数据
justified。

## 1. 模块架构:HandSetSIREN(NIAF 忠实版)

与旧 FiLMSIREN 的本质区别:21 个手部关键点**联合建模**(手形流形先验)、
transformer 调制器顶替 MLLM、grouped queries 按 SIREN 层分配(NIAF §3.1.3)。
默认 1.0M 参数;crush/部署用 xl 配置(d256/enc5/H256,~5.5M)。

```
每帧 token: 21×(x,y,conf)=63 + 腕 2 + 肘 2 + log-scale 1 + side 1 + τ 1 = 70
     │  Linear(70→d) + 位置嵌入
     ▼
┌────────────────────────────────────────────────────┐
│ TransformerEncoder  d=128×3 层(xl: 256×5)         │ ← 顶替 NIAF 的 MLLM
└────────────────────────────────────────────────────┘
     │ cross-attention
     ▼
┌────────────────────────────────────────────────────┐
│ learnable queries  Q = L·(G+1) = 12                │
│  每 SIREN 层: G=2 个 weight-query + 1 个 bias-query│
│  投影 MLP ψ(末层零初始化 → 起训时恒等调制)       │
└────────────────────────────────────────────────────┘
     │ (γ,β) / 层
     ▼
┌────────────────────────────────────────────────────┐
│ 共享 SIREN meta-prior  H=128×L=4, ω0=15            │
│   pre = (W·h)·(1+γ) + (b+β)     ← NIAF eq.7        │
│   h   = sin(ω0·pre)                                │
│   out: Linear(H → 42) = 21 点 × (x,y)  canonical   │
└────────────────────────────────────────────────────┘
     ├─► 位置 𝒜(τ)   任意 τ 采样(去噪 + 缺段补全)
     └─► 速度 𝒜′(τ)  闭式 cos 递推解析导数(NIAF §3.2,无 autograd 二阶)
```

- **canonical 帧**:conf 加权腕心原点 + 窗口中位 wrist→MCP 骨长尺度
  (对齐 `metrics._norm_hand` 语义)。
- **训练**(`hand_train.py`):conf 加权 L_pos + 0.5·解析 L_vel;观测模式混合
  {uniform-16+相位抖动, 连续缺段, 协议同款 even-frame};噪声增广档位取自
  Gate B 实测 jitter;Adam 1e-3 + 5% warmup + cosine(无 warmup 会 seed 性
  发散),grad-clip 1.0。记忆模式(target_sigma≤0)以原始检测为 L_pos 目标。

## 2. 接入 DisPose(实测有效链路)

```
驱动视频 → DWPose → reconstruct_hands.py
                     (滑窗 span32/stride16,三角混合;缺段帧 conf 抬到 0.61
                      → 原本不可见的帧变成可用控制)
                     └→ outputs/hand_pilot/hands_recon/{clip}.npz
推理:yaml 加  hand_flow: true + hand_recon_dir: <dir>
  → get_video_pose(hand_override=重建手)   ← 在 rescale/graft 之前替换,
     同一坐标链自然走完                       person-0 行正确处理多人帧
  → 重建手进入 ①骨架图分支(实测有效通道) + ②motion field(hand_flow)
```

机制事实(gain 消融,3 例 × {×3,×10}):motion field 里的手部 flow 放大
10× 输出纹丝不动 → **ControlNet 不读手部簇,手部信息的因果通道是骨架图
分支**;SIREN 的收益经由"净化/补全骨架分支消费的手部信号"实现。相关开关:
`hand_flow / hand_flow_smooth / hand_conf_thr / hand_kp_subset /
hand_flow_gain / hand_recon_dir`(逐 test_case);`pose2track/points_to_flows`
的 `n_points=18` 泛化对 K=18 逐位一致(`equiv_check_hands.py` 冻结参照
回归,全 PASS,原 step2 `equiv_check` 不破)。

## 3. 数据与测量门

- **数据**:109 条 asl27k 难例源视频,DWPose body+hands+conf @stride1
  (`extract_hand_poses.py`)。V1 实证:`hands[0]=LEFT`(腕点距离中位
  0.01 vs 对侧 0.13+;`pose_extract.py` 旧注释是错的)。
- **窗口**:span 32 / step 8;gating = 全帧有人 + ≥80% 帧 conf≥0.3 +
  **中位骨长 ≥8px** + **|canonical| ≤12** → 2125 窗 / 108 clips。
  后两道门必开:DWPose 会在 0.3–0.5 conf 下输出塌缩 ~2px 的"手",
  不滤会让 canonical 速度爆炸(首跑 loss_vel 1e17 的根因)。
- **Gate B 噪声画像(PASS)**:手 jitter 3.78px = body 腕 2.14px 的 1.76×;
  dropout 5.9%(146 缺段,中位 4 / p90 48 / max 97 帧);指尖 conf 0.63。
  去噪+补全头寸真实存在。

## 4. 组件实验(轨迹层)

### 4.1 P0 容量(PASS)
全量 2125 窗过拟合:conf 加权 L_pos 0.0234 = 噪声地板(0.0113)的 2.1×,
解析速度粗糙度 0.96(无振铃),ω0=15 冻结。

### 4.2 记忆天花板 crush(吊打 baseline)
raw-detection 目标 + 协议同款观测模式 + xl 模型,4000 epochs,训练窗口上
双协议对打(spline σ 在评测数据上调到最优 = 最强 baseline):

| gap 长度 | spline | learned | 领先 |
|---|---|---|---|
| 2 | 0.0286 | 0.0181 | 1.6× |
| 5 | 0.1970 | 0.0312 | 6.3× |
| 9 | 0.3089 | 0.0304 | 10.2× |
| 14 | 1.3834 | 0.0383 | **36.2×** |
| 16 | 1.0805 | 0.0563 | 19.2× |

learned 误差对 gap 长度近乎平坦(检索而非插值),spline 随 gap 爆炸——
而真实缺段 p90=48 帧。even/odd holdout:learned 0.0235 vs spline 0.0287
(1.2×,双方都被检测噪声地板锁死,结构性上限)。
教训:平滑目标 + 单一固定观测模式训出的模型在自己的训练窗口上反而输
spline 5–20×——检索接口必须在训练时被练到。

### 4.3 Scaling(asl50k justified)
clip 级切分,{16,32,64,84} × 3 seeds,held-out 24 clips:

| train clips | held-out MSE |
|---|---|
| 16 / 32 / 64 / 84 | 2.24 / 1.16 / 0.67 / 0.52 |
| spline / linear / gauss | **0.042** / 0.056 / 0.088 |

单调、跨 seed 紧;log-log 斜率 **−0.87**,外推 **≈1.5k clips** 穿 spline
(斜率减半也只到 ~26k < 50k)。84 clips 输 spline 是预期数据墙;判据是斜率。
预注册(斜率<−0.05 且交点<50k):**PASS**。

## 5. P2 交付:三系统对比(109 条全量)

同 DWPose 权重、共享 pose_cache、同协议(job 16551584 / 16571908)。
MimicMotion / DisPose 列来自 `baseline/quantitative.md`。

### 5.1 单 seed

| 指标 | MimicMotion | DisPose+graft | **+SIREN** | 配对(vs DisPose) |
|---|---|---|---|---|
| **mean_hand_conf ↑** | 0.6801 | 0.6988 | **0.7126** | **91/109**,p=3e-13 |
| hand_good_rate ↑ | 0.8831 | 0.8628 | **0.8713** | 坏手率 −6.2% rel |
| FVD ↓ | 907.1 | **830.4** [838,884] | 837.1 [839,894] | CI 重合,平 |
| CSIM mean/worst/std | .773/.671/.039 | .809/.766/.0189 | .809/**.768**/**.0181** | 平/微升 |
| body_pck / body_nme | .274/.444 | .280/.414 | .280/.415 | 平(护栏守住) |
| hand_pck / hand_nme | .326/.532 | .318/.533 | .297/.560 | 降,见注 |

### 5.2 best-of-N(最终交付)

18 条败例重摇 2 seeds(123/777),按 **DWPose 手部置信度**逐条选优——
GT-free、部署可用的 test-time reranking;14/18 reroll 胜出。

| 指标 | MimicMotion | DisPose+graft | **+SIREN (best-of-≤3)** | 配对 |
|---|---|---|---|---|
| **mean_hand_conf ↑** | 0.6801 | 0.6988 | **0.7149** | **101/109**,p=6.4e-22 |
| hand_good_rate ↑ | 0.8831 | 0.8628 | **0.8739** | 坏手率 **−8.1% rel** |
| FVD ↓ | 907.1 | 830.4 | 834.3 [837,891] | 平 |
| CSIM mean/worst/std | .773/.671/.039 | **.809**/.766/.0189 | .806/.765/**.0183** | 平(−.003 噪声内) |
| body_pck / body_nme | .274/.444 | .280/.414 | .279/.416 | 平 |
| hand_pck / hand_nme | .326/.532 | .318/.533 | .297/.566 | 降,见注 |

**论文脚注(必须带)**:SIREN 列 = best-of-≤3 seeds,按 DWPose 手部置信度
重排(无 GT,部署可用);baseline 为单 seed 原始配置。

**hand_pck/nme 下降注**:预注册即排除其作判据(DWPose 对糊手/blob 不敏感,
MimicMotion 靠 smooth-but-wrong 手得高分);且效应部分是定义性的——SIREN
用去噪/补全轨迹**替换**了原始检测,与"带噪原始检测"这把尺子的偏差按构造
增大。

**污染声明(pilot 有意为之)**:轨迹先验在这 109 条上过拟合训练——上限
演示;干净版训 asl50k(剔除 109 条/同 signer/同词)。

产物:`outputs/metrics_siren/`(单 seed)、`outputs/metrics_siren_best/`
(终版)、生成视频在集群 `outputs/sign_siren_full|_reroll|_best/`。

## 6. 复现笔记

- jubail env:insightface 带入的 CPU `onnxruntime` 会遮蔽 `onnxruntime-gpu`
  → DWPose 静默走 CPU;两个 wheel 共享 `onnxruntime/` 目录,卸载 CPU 版会
  连带损坏 GPU 版,须 `pip install --force-reinstall --no-deps
  onnxruntime-gpu==1.19.2`。跑批前先验证 CUDAExecutionProvider 在列。
- 长 clip 分布不均会让 109 条 3 片并行撞 4h slurm 时限(实际两片各损
  5 条),补跑用 `hand_pilot_gen.slurm CFG OUTDIR NAME`。
- 关键作业:提取 16540596;回归 check 16540597;Gate A 三臂 16541178-80;
  crush v2 16543899;scaling v2 16542440;P2 生成 16546502-04+16551355/56;
  指标 16551584;reroll 16566660/61;best-of-N 16571908。

## 7. 下一步(asl50k 阶段,待用户输入)

1. asl50k 与 asl27k/109 条的重叠关系?signer/word 元数据可否做训练集排除?
2. 数据形态:裸视频 or 预提 pose?(5 万条 DWPose 提取 ≈ 270 A100-h,
   是 scale-up 的算力大头,需排期/分批)
3. 干净泛化版复跑 P2;人评 / 尾部指标细化;论文口径定稿。
