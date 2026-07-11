# round1_archive — 第一轮(body 信号精化)脚本存档

冻结存档,对应 `docs/experiments/round1_archive/` 三份判死记录:
`step1/` 去噪验证、`step2/` 时域超分 + 等价性 gate、`step3/` 采样期融合;
`slurm/` 为当时的作业脚本。归档时已修正移动带来的路径/import
(`dispose_siren.round1.*`),保持可复跑。

例外:`step2/lib_lowfps.py` 与 `step2/12_equiv_check.py` 仍被现役
`hand_pilot` 引用(共享检测/等价性基建),勿删。
