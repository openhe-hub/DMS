# hand_pilot — 现役 SIREN 手部通道流水线

实验记录:`docs/experiments/siren_hand/siren_module.md`。编号 = 执行顺序;
产出全部落在 `outputs/hand_pilot/`(gitignored,大件在集群)。

## 流水线 DAG

```
30_extract_hand_poses.py   [jubail GPU]  109 条源视频 DWPose → poses/{clip}.npz
        │                                (+V1 左右手核验, pose_cache 交叉核对)
        ├─► 31_gate_b_noise.py   [本地]  噪声画像 → gate_b/{summary,per_clip,
        │                                gap_lengths,case_ranking}
        └─► 32_build_windows.py  [本地]  建窗+gating → windows/windows_span32.npz
                │
                ├─► 33_overfit.py        P0 容量 / --crush 记忆天花板(出 ckpt)
                ├─► 34_scaling_curve.py  clip 级切分 + {16,32,64,all}×seeds
                └─► 35_report_figs.py    scaling/gap 决策图 + decision.json
40_make_gate_a_configs.py  生成各臂 yaml(--arms/--clips/--all/--shards/--seed)
41_equiv_check_hands.py    ★阻塞回归:K=18 逐位一致 + dead-hands 惰性
                           (--mode real 须在生成前 PASS)
42_gate_a_inspect.py       生成视频 DWPose 提取(--stage extract, 集群)
                           + 多臂裁块对比图/配对诊断(--stage report, --arms/--tag)
43_reconstruct_hands.py    SIREN 滑窗重建手轨迹 → hands_recon/(喂 hand_recon_dir)
44_select_best.py          best-of-N 按手部 conf 选优 → sign_siren_best/
```

## slurm 对应(scripts/slurm/)

| slurm | 干什么 |
|---|---|
| hand_pilot_extract | 30(+32 集群侧 sanity) |
| hand_pilot_gate_a MODE | check=41+旧等价gate / off\|raw\|smooth\|siren\|gain*=生成臂 / inspect / report |
| hand_pilot_crush | 33 --crush --xl |
| hand_pilot_scaling | 34 |
| hand_pilot_siren_full N / hand_pilot_gen CFG OUT NAME | P2 全量分片 / 通用生成 |
| hand_pilot_metrics_siren / hand_pilot_bestof | 指标套件 / best-of-N 全链 |
