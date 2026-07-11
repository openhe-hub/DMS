# Step 3 — 采样时 latent 帧间融合(R100–R101):**方向终止 ❌**

> 动机:step2 判死控制侧后,攻扩散采样侧——RIFE 唯一可见弱点是快运动插值涂抹;
> 让中间帧"被生成"而非"被插值",每步去噪用 flow-warp 的关键帧信息软约束。
> 预注册计划见 `../../refine-logs/EXPERIMENT_PLAN.md` Step3 节;查新见
> `../../idea/step3_sampling_fusion_novelty.md`。日期 2026-07-03,总花费 <1.5 GPU-h。

## 方法(training-free,零侵入)

- 干预点:`pipeline_ctrl.py` 去噪循环(callback / scheduler 包装),latent
  `[1,F,4,h/8,w/8]`,帧 0 = ref(输出丢弃,索引 +1)。
- 后向流 m→k0/k1 从**插值关键点自建**(可见 kp 位移 Nadaraya-Watson 高斯撒点,
  无支撑处→0),确定性、避开 CMP scatter 竞态。
- 控制信号 = linear 插值 pose(R012:插值方法在 video-level 无差异)。
- 代码:`scripts/step3/`(lib_fusion.py:FusionCallback latent 空间 /
  X0FusionScheduler x0 空间;21 生成;23 分桶评测)。

## R101 dev(case0=video1,stride 8,与 R011 完全配对;两轮,预注册上限)

**Round 1 — latent 空间 blend(job 16498220)❌**

| 配置 | mid-PSNR | mid-LPIPS |
|---|---|---|
| linear(基准) | 15.32 | 0.343 |
| RIFE(基准) | 15.60 | 0.344 |
| α.15 win[.3,.9] | 15.26 | **0.606** |
| α.30 win[.3,.9] | 11.39 | 0.801 |
| α.15 win[0,.7] | 8.87 | 0.867 |
| α.30 win[0,.7] | 7.68 | 0.904 |

失败模式:早窗口 = 高噪声阶段跨帧 blend 破坏噪声统计 → 采样离流形,输出崩坏;
温和配置 = 平均化模糊,PSNR 撑住但 LPIPS 翻倍。教训与 TRF/ViBiDSampler 流派一致:
**不能碰带噪样本,只能碰去噪方向。**

**Round 2 — x0 空间融合(job 16498624)❌(按 G2)**

X0FusionScheduler:每步取 `pred_original_sample`,在预测干净视频上 warp-blend,
再用融合后 x0 精确重算 Euler 步(α=0 与原生逐位一致,mock 验证)。

| 配置 | mid-PSNR | fast桶 | mid-LPIPS |
|---|---|---|---|
| linear | 15.32 | 14.29 | **0.343** |
| RIFE | 15.60 | **14.61** | 0.344 |
| x0 α.3(两窗口一致) | 15.59 | **14.63** | 0.378 |
| x0 α.6 | 15.50 | 14.48 | 0.402 |

判读:
1. **结构对齐修复成功**:PSNR 从 linear 15.32 → 15.59,追平 RIFE;fast 桶 14.63
   还略胜 RIFE 14.61 —— 机制本身按设计工作了。
2. **但纹理代价系统性存在**:LPIPS 差 10%(0.378 vs 0.343),α 0.6→0.3 单调改善
   说明这是"结构收益 vs 纹理损失"的连续 trade-off,α→0 时两者同时回到 linear。
3. 两窗口结果几乎相同 → 早期融合无效(早期 x0 本来就糊、会被覆写),全部作用
   在后期 —— 而后期融合正是纹理损伤来源。**没有免费午餐的参数区间。**

## 判决

预注册 G2:"差于 linear 即配置失败;两轮 dev 无合格配置 → 终止"。
Round1 全灭,Round2 LPIPS 全部显著差于 linear → **step3 终止,R102(3-case 终审)不跑。**

## 机制结论(与 step2 合并后的完整图景)

- RIFE 的优势不是"没人在生成时做融合",而是它用**专门训练的运动补偿网络**在
  **成品帧**上插值;我们 training-free 的 18-kp 姿态流 warp 精度不足,x0 平均的
  纹理错位需要靠剩余去噪步重绘,而在收益可见的后期窗口里剩余步数不够。
- 更根本的量级问题:stride8 时 linear 与 RIFE 只差 0.28 dB,linear 与 orig:1 上限
  只差 ~1.8 dB 且大头是扩散自身误差(step2 结论)——**这个操作点上留给任何
  插帧改进的总空间本来就 <0.3 dB**,training-free 采样干预吃不下来。
- 若要继续,只剩训练路线(更好的 flow、遮挡处理、ViBiDSampler 式双向重采样、
  或直接训插值扩散)——但那要和 PoseFuse3D-KI/LumosFlow 等训练式工作正面竞争,
  且预期收益上限仍受上一条压制。**不建议。**

## 残值

1. 负结果 ×2(latent 空间不可 blend 的定量失败模式;x0 空间的结构/纹理 trade-off
   曲线)—— 可并入 step2 的负结果叙事:"控制侧与采样侧的 training-free 干预
   在该操作点均无头寸"。
2. `X0FusionScheduler`(零侵入 x0 空间干预框架)与关键点自建流场:可复用于
   任何后续采样时干预实验。
3. 分桶评测脚本(23_bucket_metrics.py):运动幅度分桶的逐帧 PSNR/LPIPS,通用。
