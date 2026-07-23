# configs 目录说明

| 位置 | 是什么 |
| --- | --- |
| `test_sign_sharpness_winner.yaml` | **当前正主配置**:winner(768/cfg3/tile32/ovl8/25步)+ head0.15/face0.4 表情放开。新实验从这份抄 |
| `test.yaml` | 上游 DisPose 原版 demo 配置,不动 |
| `face_blend/` | 表情/头部放开扫描(2026-07-23,docs/experiments/face_blend.md) |
| `sharpness/` | 清晰度调优:网格 + fair 三件套(docs/experiments/sharpness_tuning.md) |
| `hard27k/` | hard27k 定量批量:576 基线 6 份 + winner30 试点 5 shard(metrics_winner30.slurm 依赖 winner30 的 glob) |
| `omnihand/` | OmniHands 融合三 arm + 分辨率消融(docs/thirdparty/omnihand/fusion.md) |
| `siren/` | SIREN/hand_flow 时代全部配置(gate A/B、reroll、siren_full shard) |
| `early/` | 最早期:对齐验证 + graft on/off 冒烟 |

一次性 shard 配置(winner_rest_0..13、winner_fix_0..4,已消费)已删,
需要时从 git 历史找(commit c427034 之前)。
