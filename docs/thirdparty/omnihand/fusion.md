# OmniHands × DisPose 手部融合实验(第一轮:easy 片段)

> 日期 2026-07-12。前置:[2D 关键点导出](kps_export.md)。
> 作业:prep `16684364`,生成 off/full/fused = `16684560/61/62`(A100,各约 30 分钟,exit 全 0)。

## 设计

原生 DisPose(不上 SIREN),3 段 8s 手语视频 × `refs/test2.jpg` 跨身份驱动,seed 42 配对,三个 arm 仅 `hand_recon_dir` 不同:

| arm | 手部来源 | 配置 |
| --- | --- | --- |
| off | 原生 DWPose | [`test_sign_omnihand_off.yaml`](../../../configs/test_sign_omnihand_off.yaml) |
| full | OmniHands 全替换(SavGol 防抖) | [`test_sign_omnihand_full.yaml`](../../../configs/test_sign_omnihand_full.yaml) |
| fused | 置信度门控融合(0.3 整手均值) | [`test_sign_omnihand_fused.yaml`](../../../configs/test_sign_omnihand_fused.yaml) |

融合脚本 [`scripts/hand_fusion/fuse_kps.py`](../../../scripts/hand_fusion/fuse_kps.py):逐帧逐手,DWPose 均值置信度 ≥0.3 保留原检测,否则整手换 OmniHands 投影(分数 0.61)。prep 作业 [`omnihand_fusion_prep.slurm`](../../../scripts/slurm/omnihand_fusion_prep.slurm) 完成 DWPose 提取(复用 `extract_hand_poses.py`)+ 融合 + arm B npz 暂存;生成复用 `hand_pilot_gen.slurm`。

## 结果(本地 `outputs/omnihand_fusion/`,gitignore)

1. **门控在这批视频上零触发**:3 段绿幕手语视频 DWPose 逐手均值置信度 0.80–0.90,全部 1152 手×帧仅 1 帧 <0.5 → fused 的控制信号与 off 完全一致。**fused arm 因此转为注入链路保真度对照**。
2. **保真度验证通过**:fused vs off 逐段 PSNR 平均 39.6–44.7 dB(仅 GPU 非确定性微差)——`hand_override` 注入路径不引入任何副作用。
3. **off vs full 手部质量同级**(逐帧并排 `outputs/omnihand_fusion/cmp/`):easy 内容上 OmniHands 投影驱动的生成不劣于 DWPose 原生。**推论:门控偏灵敏是安全的**——即使误替换了好帧,质量也不掉。
4. 修复价值的真正战场在 hard27k:用缓存 DWPose 轨迹测算,8 段 gate-A 片段上 0.3 门控触发率 **28.2%**(单段最高 55%,`0ddpfhlmff` 左手 139/142 帧低置信)。

## 下一步(待定)

- 在 hard27k gate-A 片段上重跑三 arm(需先给这些片段跑 OmniHands 推理 + kps 导出)。
- 门控粒度讨论(用户提出):整手均值会被 19 个好点稀释,漏掉"单指乱飞";逐点替换又太灵敏且一手混两模型。候选:min/分位数触发、按指替换、时序滞回。鉴于结果 3(过度触发无害),倾向灵敏档(如 `frac(kp<0.3)≥25%` 整手替换)。

## 运维备注

- **jubail 仓库已接通 hub(2026-07-13)**:此前 `origin` 只指上游 `lihxxx/DisPose`、与私有 hub(openhe-hub/DMS)不通,靠散文件传输。现已配置:jubail 端 `~/.ssh/id_ed25519_github` 只读 deploy key(hub 仓库 key id 157118769),仓库级 `core.sshCommand` 指定该钥匙,`hub` remote + `main` 跟踪 `hub/main`,工作树已 reset 对齐(160 个 untracked 碰撞文件仅 3 个文档有差异且均为旧稿,备份在 `/scratch/zl6890/zhewen/dispose_untracked_diff_backup_20260713.tgz`)。**同步流程:本地 commit + push origin main → jubail `git pull`**。
- 生成作业输出 192 帧(不含 ref 帧),fps 24 与源一致。
