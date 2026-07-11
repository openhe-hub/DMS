# Step 1 — Learned-INR vs fd/fd+Gaussian 在真实 DWPose 轨迹上的验证

**结论（诚实）：在 pose-level、真实 DWPose 轨迹上，learned-INR 完败于简单 baseline（linear interp / finite-diff），差距约 2.5–3×。而且这不是"噪声不够大"，而是 synthetic 先验与真实运动之间的 domain gap——注入噪声扫到 σ=32px 仍然没有任何 crossover。** Experiment B 在 synthetic 上 2.2–2.7× 的胜势，无法迁移到真实数据。

代码：`src/dispose_siren/`（库层）+ `scripts/round1_archive/step1/`（脚本层）。运行：Jubail A100，job 16445712（主）/ 16445726（噪声扫）。

---

## 协议（真实数据没有 GT 速度，用两个诚实代理）

- **数据**：DisPose 自带 3 段 driving video（`assets/example_data/videos/video{1,2,3}.mp4`）经 DisPose 自己的 DWPose + `pose2track` 抽出 18×T 关键点轨迹（T=282–295）。滑窗 span=48、step=24，只取窗口内全程可见的关键点 → 共 **416 windows**。
- **Protocol A（held-out 帧重建，中立）**：从 48 帧取 32 个等距真实检测，奇偶拆成 observed/held-out；各方法只用 observed 重建 held-out 位置，比 px² MSE。held-out 噪声方差对所有方法是同一常数（预测只依赖 observed，与 held-out 噪声独立），所以排序公平。
- **Protocol B（high-fps 伪 GT 速度，偏向 fd 类）**：48 帧轻平滑后做差分当伪 GT 速度，observed 取其中 16 帧。**伪 GT 本身就是带噪检测的差分，结构上偏袒 fd/fd+Gaussian**，所以 A 是主判据。
- baseline 全给最优待遇：fd+Gaussian / Gaussian+linear 的平滑 σ 在该数据上**调到最优**。
- learned 模型：scale-invariant 版本（per-trajectory z-score，单一 mixed-noise σ∈{2,4,8,14,20} 模型，便于迁移到未知真实噪声）。

## Synthetic sanity（full 400 epochs，通过）

scale-invariant 变体在 synthetic held-out 上**所有噪声级别都赢**（甚至比原 absolute-coord 版更强）：

| σ | finite-diff | fd+Gauss(best) | learned-INR | win |
|---|---|---|---|---|
| 3  | 78.6 | 266.6 | **60.5** | LEARNED |
| 6  | 107.3 | 276.1 | **68.7** | LEARNED |
| 12 | 223.0 | 315.5 | **102.2** | LEARNED |
| 20 | 497.8 | 410.4 | **182.7** | LEARNED |

→ 模型/velocity/normalize 代码正确，机制在 in-distribution 数据上成立。

## 真实数据结果（416 windows）

真实 DWPose 抖动很低：**abs ≈ 2.2–2.9px，relative ≈ 0.10–0.14**（抖动 std / 运动幅度 std）。

| | linint / fd | gauss+lin / fd+gauss | learned-INR | 判定 |
|---|---|---|---|---|
| **A** held-out pos-MSE (px²) | **105.87** | 144.12 | 284.79 | baseline（best/learned=0.37×）|
| **B** pseudoGT vel-MSE | **15708** | 23860 | 48654 | baseline（0.32×）|

learned-INR 比最强 baseline **差约 2.7×（A）/ 3.1×（B）**。

## 决定性诊断：domain gap，不是噪声级别

在**真实轨迹上注入**合成抖动 σ 并扫描，看 learned 是否随噪声增大反超 baseline（`scripts/round1_archive/step1/noise_sweep.py`，job 16445726）：

| 注入 σ(px) | base_A | learn_A | winA | base_B | learn_B | winB |
|---|---|---|---|---|---|---|
| 0  | 105.9 | 284.8 | base | 15708 | 48654 | base |
| 2  | 108.1 | 286.2 | base | 16802 | 49258 | base |
| 4  | 115.0 | 290.7 | base | 19809 | 50088 | base |
| 8  | 142.7 | 310.2 | base | 29438 | 53630 | base |
| 16 | 245.1 | 395.9 | base | 45908 | 67623 | base |
| 32 | 513.5 | 760.1 | base | 80168 | 126030 | base |

**从未 crossover。** 对比：synthetic σ=20（relative≈0.3–0.4）时 learned 以 2.7× 领先；真实 σ=32（relative>1）时 learned 仍落后 ~1.5×。

→ 否定"真实 DWPose 太干净所以先验用不上"的假设。真正原因是 **synthetic 运动先验（3 个正弦之和、freq 0.5–3/窗）≠ 真实人体关键点运动**。模型学到的是"把轨迹去噪成 synthetic 形状"，这个偏置在所有噪声级别都系统性地扭曲真实运动，因而恒输。

## 对 idea 的影响

1. **Experiment B 的胜利是 train/test 同分布的产物**，不构成对 paper claim 的支持。pose-level + synthetic → 真实 DWPose 的迁移，按当前设计**失败**。
2. 真实 DWPose 在 pose-level 已足够干净（relative≈0.12），简单 baseline（线性插值/差分）近最优，**去噪头寸很小**——这本身就削弱了"更好的 motion field"这条主线在 pose-level 的意义。
3. 要救这个方向，唯一诚实的修法是**让先验匹配真实运动统计**：在真实 DWPose 轨迹上自监督训练（high-fps 当 pseudo-clean，降采样+加噪当输入），而不是合成正弦。这是下一个该做的实验。
4. 或者把战场移出 pose-level：直接做 **video-level（Step 2，VBench/FID-FVD）** 或 **任意帧率**——胜负可能只在那里显现，pose-level MSE 不是 win 所在。

## 补充：改训真实运动先验（Step 1.5，leave-one-video-out）—— 撞数据墙

为修 domain gap，在真实 DWPose 轨迹上**自监督改训** INR（high-fps dense 轻平滑当 pseudo-clean target，降采样+加噪当输入，L_pos+0.5·L_vel，scale-invariant）。用 **leave-one-video-out**（2 段训、1 段测）做真正的泛化测试。`scripts/round1_archive/step1/real_finetune_lovo.py` + `src/dispose_siren/real_train.py`。

踩坑：伪 GT 速度 = `diff(real_dense)·(span-1)` 会把检测噪声放大 ×47，`loss_vel` 巨大且噪声化 → 必须 target_sigma=2.0 多平滑 + vel_w=0.5 压权重 + weight_decay。

结果（250 epochs，train loss 收敛到 loss_pos≈0.03）：

| | baseline | synthetic-INR | **real-INR (LOVO)** |
|---|---|---|---|
| A held-out pos-MSE | **105.9** | 284.8 | 724.8（6.85× 输）|
| B pseudoGT vel-MSE | **15708** | 48654 | 99269（6.32× 输）|

**关键观察：real-INR 在 held-out 视频上比 synthetic-INR 还差。** train loss 收敛但 held-out 崩 = 典型过拟合。2 段训练视频运动模式太少，~280 windows 高度相关。**广而错的 synthetic 先验（284）反而比窄而对的 real 先验（724）泛化更好 → 绑定约束是运动多样性，不是 domain-match。**

→ **现有 3 段视频根本撑不起"改训真实先验"的验证。** 要继续这条路，前提是先扩到几十–上百段多样人体视频。在扩数据之前，更划算的是先做 **video-level 敏感度探针**（training-free）确认 pose-level 改进到底能不能传导到生成视频，否则改训先验即便成功也可能白做。

## 文件
- 库：`src/dispose_siren/{synth,models,baselines,normalize,trajectory,train,eval_protocols}.py`
- 脚本：`scripts/round1_archive/step1/`、`scripts/round1_archive/slurm/`
- 产物（gitignored，local）：`outputs/step1/{traj,ckpt,fig}/`，图 `step1_real_bars.png`、`step1_noise_sweep.png`
