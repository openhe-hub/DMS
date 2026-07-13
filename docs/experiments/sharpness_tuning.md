# 生成清晰度调优(工程前置,2026-07-13)

> 背景:OmniHands 融合第一轮(见 [fusion.md](../thirdparty/omnihand/fusion.md))
> 发现生成视频**手部发糊是手部特有问题**(同帧脸部清晰):576 分辨率下一只手
> 只占 ~60–90px,过 VAE 8× 下采样后仅 ~8–11 latent token,装不下手指结构。
> 在回到 OmniHands/SIREN 创新点之前,先摸清管线的工程清晰度上限。

## 管线暴露的旋钮

yaml → `inference_ctrl.py run_pipeline`(L273-281):`resolution`(经参考图
长宽比定 H/W)、`num_inference_steps`、`guidance_scale`(管线把 SVD 的
min/max guidance 钉成同一常数)、`num_frames`(=tile_size,时间滑窗长)、
`frames_overlap`(=tile_overlap,窗间混合帧数)、`noise_aug_strength`、
`decode_chunk_size`(纯显存)。**未暴露**:SVD 的 fps 微条件(硬编码 7)、
motion_bucket_id。

## 实验(单变量 → 组合,同 clip 5ok8y / 同 seed 42 / 原生 DWPose)

指标:同帧(f100)手部区域 Laplacian 方差(锐度)、帧间差尖峰数(时序,
tile 接缝检测)、单段耗时(A100)。作业 16696812 / 16697108 / 16698304。

| 配置 | 锐度 | 时序尖峰 | 耗时 | 结论 |
| --- | --- | --- | --- | --- |
| 576 基线(25步 cfg2 tile16/6) | 6.2 | – | ~10min | 手是蜡质团块 |
| 768 基线 | 14.0 | 24 | ~50min | **分辨率是第一杠杆(+2.3×)** |
| 768 + 40步 | 13.8 | – | ~70min | 步数无效,不值 1.6× 时间 |
| 768 + cfg3.0 | 14.9 | 21 | ~50min | 有效,无过饱和/伪影 |
| 768 + 40步 + cfg3.0 | 15.2 | – | ~70min | 与 cfg3 单独差 2%,步数仍无效 |
| 768 + tile32/ovl8 | 14.7 | 19 | ~35min | 有效且更快(重叠计算少) |
| **768 + cfg3.0 + tile32/ovl8(推荐)** | **15.8** | **16** | **~34min** | 三项全胜,增益可叠加 |
| 1024(2 案例作业) | – | – | >4h 超时 | 注意力平方开销,单段 2h+,不经济 |

参考:源视频手部区域锐度 11.4(有压缩)——768 档生成已超过源画质,
继续上探主要是微观纹理。

帧间差尖峰与 tile 边界数对不上(tile32 只有 8 个边界但有 19 个尖峰),
尖峰来自真实快动作,**所有配置均无接缝病灶**。

## 推荐配置

[`configs/test_sign_sharpness_winner.yaml`](../../configs/test_sign_sharpness_winner.yaml):
`resolution 768 / guidance_scale 3.0 / num_frames 32 / frames_overlap 8 /
num_inference_steps 25`。相对 576 基线:手部锐度 +155%,耗时 ~3.4×(34min/8s段)。

素材(本地已精简,只留固定脸终版;网格全量原件在 jubail
`outputs/omnihand_fusion/{res,sharp}/`):`outputs/omnihand/fusion/`——
`fair_{576,winner,winner_omnihand}.mp4` 三件套 + `cmp/`(同帧手部放大
`hands_fair_*.png`、源手部参照、并排视频)。网格配置
[`test_sign_sharpness_grid.yaml`](../../configs/test_sign_sharpness_grid.yaml)、
[`test_sign_omnihand_res.yaml`](../../configs/test_sign_omnihand_res.yaml)。

## 公平复核(固定表情,作业 16698855)

发现 head 姿态默认按 15% 从驱动视频混入(`head_blend_ratio`,原先硬编码),
表情带随机性、混淆分辨率对比 → 已把该参数暴露到 yaml
(`inference_ctrl.py`),`head_blend_ratio: 0` 时表情完全钉在参考图。
固定表情重跑([`test_sign_fair_576_vs_winner.yaml`](../../configs/test_sign_fair_576_vs_winner.yaml)):

| 档 | 手部锐度 |
| --- | --- |
| 576 基线 | 7.2 |
| winner | 15.5(与首跑 15.8 一致,可复现) |
| winner + OmniHands 全替换手 | 13.9 |

winner+OmniHands 略低(-10%),疑似 hands_score=0.61 让骨架图手部偏暗、
条件略弱所致(见 fusion.md 风险 2)——**分数 0.9 消融是下一个待办**。
并排视频:`cmp/sbs_fair_576_vs_winner.mp4`、
`sbs_fair_winner_vs_omnihand.mp4`。

## 泛化验证(3 段,方法定稿,作业 16715583/84)

固定表情三件套扩到全部 3 段手语视频(整段手部区均值锐度,帧抽样 1/4):

| clip | 576 | winner | winner+OmniHands | winner/576 |
| --- | --- | --- | --- | --- |
| 5ok8y | 101.7 | 191.6 | 166.5 | 1.88× |
| 1arny | 93.7 | 183.2 | 169.5 | 1.95× |
| di6t6 | 94.4 | 169.9 | 161.7 | 1.80× |

三段一致:winner 稳定 ~1.9×;OmniHands 档一致地略低 5–13%(0.61 调暗假设
成立的旁证,分数消融仍待做),抽查无解剖伪影。**清晰度工程调优就此定稿**:
推荐配置 = `test_sign_sharpness_winner.yaml` 的
768/cfg3.0/tile32/ovl8/25步(+ 需要固定表情时 `head_blend_ratio: 0`)。
每段 grid 图:`cmp/grid_{,1arny_,di6t6_}orig_vs_winner.png`、
`grid_orig_vs_winner_omnihand.png`(量化选帧:锐度比 top-9、帧距 ≥15,
5ok8y 的 f10-35 因增强侧四指伪影被排除换补)。

## 若还要更清晰(未做)

1. 改代码级旋钮:恢复 SVD 原版 guidance 首末帧 1.0→3.0 线性爬升;调
   motion_bucket_id。
2. 手部区域二次精修:用 OmniHands 的逐帧 MANO 网格渲染 depth/normal,
   对手部做 HandRefiner 式 inpaint——已属 OmniHands 创新点范畴。
3. 时序一致的视频超分(BasicVSR++ 类;RealESRGAN 逐帧会闪)。
