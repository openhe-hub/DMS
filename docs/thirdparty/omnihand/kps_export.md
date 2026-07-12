# OmniHands 3D → 2D 手部关键点导出(回接 DisPose 链路)

> 日期 2026-07-12。前置:[OmniHands 推理复现](reproduction.md)(环境、路径、
> 去抖实验均见该文)。

## 动机

DisPose 用 DWPose 提 2D 手部关键点,手部检测不准时用 OmniHands 的 3D MANO
恢复做 recovery,再**投影回 2D** 送回原来的 keypoint 链路——DisPose 侧零改动。

## 方法

转换脚本
[`omnihand_to_dwpose.py`](../../../scripts/thirdparty/omnihand/omnihand_to_dwpose.py)
(runner [`omnihand_kps.sh`](../../../scripts/thirdparty/omnihand/omnihand_kps.sh),
omhand 环境登录节点 CPU 即可,4 视频约 1 分钟),输入为去抖实验 dump 的
`demo_out_smooth/<vname>/traj.npz`(顶点 + 相机平移,平滑前后各一份):

- 关节回归与 `hands_4d/models/mano_wrapper.py` 完全一致:MANO `J_regressor`
  16 关节 + 5 指尖顶点,按 `mano_to_openpose` 重排——**即 DWPose 的
  COCO-WholeBody 手部顺序,一一对应,无需任何手工映射**。右手 J_regressor
  对左手同样成立(左手顶点是镜像的右手网格,回归是线性的)。
- 投影用渲染同款针孔相机:`f = 5000/256 × max(W,H)`,主点图像中心——mesh
  overlay 目检贴合即保证投影像素级正确。
- 输出与 hand_pilot SIREN arm 的 `hands_recon` npz 同格式:
  `{hands[T,2,21,2] 归一化(0=左,1=右), hands_score=0.61, covered[T,2]}`,
  可直接走 `mimicmotion/dwpose/preprocess.py` 的 `hand_override` /
  `hand_recon_dir` 注入口。

## 结果

4 视频,本地 `outputs/omnihand/kps/`(gitignore):叠加视频 `kps_<vname>.mp4`
+ 每视频一个 npz;远程原件在 jubail `thirdparty/omnihand/kps_out/`。

- 投影点在画面内 95.8–100%;慢动作帧手指级对齐。
- 快速挥动 + 运动模糊帧骨架滞后于真手——**raw 与 smoothed 投影偏移相同**
  (`kps_video1_raw.mp4` 对照),说明滞后来自 OmniHands 时序融合(gap=10)
  本身而非我们的 SavGol 平滑。

## 后续:融合策略(未做)

按 DWPose 置信度门控:高置信帧保留 DWPose 原检测(直接拟合图像证据更准),
低置信/缺失帧用 OmniHands 投影补,再在真实生成任务上对比。

## 踩坑

**登录节点有线程数上限**,torch/BLAS 默认开满线程会
`libgomp: Thread creation failed` 甚至段错误(报成 `numpy._core.multiarray
failed to import` 的假 ABI 错)→ runner 里
`export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`。
