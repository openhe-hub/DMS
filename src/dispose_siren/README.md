# dispose_siren — SIREN 轨迹先验包

| 模块 | 状态 | 作用 |
|---|---|---|
| `hand_traj.py` | **现役** | 手部轨迹提取包装/建窗 gating/canonical 归一化/V1 左右手核验 |
| `hand_model.py` | **现役** | HandSetSIREN(transformer 调制 + grouped queries + 共享 SIREN,解析速度) |
| `hand_train.py` | **现役** | conf 加权训练(伪 clean/raw 目标、混合观测模式、warmup) |
| `hand_eval.py` | **现役** | holdout / gap-inpaint 协议 + 非等距样条 + baseline 最优待遇 |
| `baselines.py` | 共享 | 高斯平滑/线性插值/差分等数值工具(新旧两代共用) |
| `round1/` | 归档 | 第一轮判死路线(FiLMSIREN 逐点建模、合成训练、body 轨迹提取);见 `docs/experiments/round1_archive/` |

现役实验文档:`docs/experiments/siren_hand/siren_module.md`。
