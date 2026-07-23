# 表情/头部放开(face_blend_ratio,2026-07-23)

> 背景:graft_pose_v2 从参考姿态深拷贝出发,脸部 68 landmark 全程 100%
> 参考图(静态模板脸)——手语驱动视频的口型/眉眼完全不进入生成,
> mouthing 语言信息丢失;头部仅 5 个 body 点按 `head_blend_ratio`(0.15)
> 轻微跟随。本轮加表情旋钮并扫档定值。

## 机制(commit abe279d)

`mimicmotion/dwpose/graft.py::blend_face_expression`:驱动脸 landmark
以自身中心归一、按瞳距比缩放成"形状",按 `face_blend_ratio` 与参考脸
形状线性混合,锚点 = 参考脸中心 + head_blend 后的鼻尖偏移(脸跟头摆,
骨架自洽)。直接坐标插值会把脸拉向驱动者的画面位置,故必须先归一。
`face_blend_ratio: 0`(代码默认)= 旧行为逐位一致;yaml 与
`head_blend_ratio` 并列暴露。

## 实验

两步走,先零成本后真生成:

1. **骨架预览扫描**(job 16794613,v100,DWPose 一次 + 纯 numpy 重放):
   head {0.15, 0.3} × face {0, 0.2, 0.4, 0.6, 0.8} 共 10 档渲染骨架
   视频目检,确认无脸部畸变/漂移,砍掉 f0.2(与 f0 无差)。
2. **真生成 13 段**(jobs 16795144/153/154/155,winner 配置,seed 42):
   5ok8y 全 7 档 + 1arny/di6t6 各 3 档(对照 f0 / f0.4 / h0.3+f0.6)。

## 结论(目检,产物 outputs/face_blend/,见其 README)

| 档 | 观察 |
| --- | --- |
| f0(旧默认) | 参考图笑容 8 秒钉死,模板脸 |
| f0.2 | 与 f0 肉眼无差,无意义 |
| f0.4 | 有生气、变化温和,无伪影 |
| f0.6 | 驱动者口型开始接管(不笑时刻嘴收起),无身份漂移 |
| f0.8 | 口型跟随最强,嘴唇形态开始有"外来感"(身份泄漏上限) |
| head 0.3 | 头部朝向跟随更明显,静帧无伪影(旧 fork 0.20+ 出伪影未复现) |

三段 clip 泛化一致。**定档(用户选定):`head_blend_ratio 0.15 +
face_blend_ratio 0.4`**,已写入
[`test_sign_sharpness_winner.yaml`](../../configs/test_sign_sharpness_winner.yaml);
代码默认仍为 face 0(不影响历史配置复现)。

## 待办

- hard27k 小样本(~15 段)新档 vs 钉死档 CSIM 对比:表情放开预期回吐
  一部分 CSIM(109/109 全胜有模板脸成分),量化代价。
- 若 CSIM 回吐过大,可退 f0.3 或只混嘴部 landmark(48-67)。
