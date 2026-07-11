# Experiment Tracker — SIREN×DisPose Video-Level

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | 集成 sanity:stride=1 插值控制 ≡ 原版 | detect 一次 → direct vs interp@1 双路构建 | video1/2 全帧 | 控制张量逐位比较 | MUST | **DONE ✅** | GATE A 全 PASS(job 16485280/281)。独立重检测存在 ONNX-GPU 非确定性(可见 kp ~2e-3、不可见 0.28、0 次可见性翻转)→ pilot 内共享单次检测,不受影响 |
| R002 | M0 | 评测指标就位(pilot 用 PSNR/LPIPS/warp;VBench 推后) | lpips(alex) | — | — | MUST | DONE | VBench 留到主表阶段,非阻塞 |
| R003 | M0 | RIFE 装通 | Practical-RIFE + v4.8 train_log(HF 镜像 codingggasdfasf/video-animator) | — | — | MUST | DONE | /scratch/.../tools/Practical-RIFE;支持任意 timestep → ×s 直接按 j/s 插 |
| R010 | M1 | B3 快筛:插值精度(任务=插值,非去噪) | linear/spline/gauss+lin/perclip/amortized | 真实 DWPose 轨迹,span=15s+1、16 obs;dev=video1,test=video2+3 | held-out pos-MSE(px²) | MUST | **DONE ✅** | **spline 赢**:s4 = 0.76×linear(54.7 vs 72.0),s8 = 0.94×;perclip(w0=3,λ=0)0.95×/1.19×;amortized 惨败 2.2–3.2×;s2 linear 最优(运动局部线性)。→ pilot 三连续方法全上,video-level 终审 |
| R011 | M1 | Pilot 主跑:stride 退化曲线 | orig:{1,4,8} + {linear,spline,siren}:{4,8} + RIFE 后处理 | video1/2/3 自驱动(ref=首帧,GT=原视频帧) | PSNR(all/obs/mid)/LPIPS/warp | MUST | **DONE ✅** | jobs 16485807/808/927 + metrics 16486552(中途踩坑:40G 卡 OOM→80g 约束+显存清理+检测缓存;video3 双人→person-0;sparse-flow scatter 竞态→gate 改比确定性输入)。27 个生成全部完成,sanity 全过 |
| R012 | M1 | 生死判定:三个 anti-claim 初判 | (R011 分析) | — | — | MUST | **DONE — 方向终止 ❌** | ① 低fps下离散管线 warp 崩坏(是);② **RIFE 后处理 mid-PSNR 6/6 全胜连续控制**(+0.00~0.31dB),LPIPS 平,预注册规则触发终止;③ linear≈spline≈siren(<0.1dB),siren 垫底,INR novelty 死。机制:扩散控制跟随误差(obs 帧仅 13.6–20dB)>> 控制信号插值误差 → 控制侧改进 = 二阶小量。详见 docs/experiments/round1_archive/step2_video_level.md §4 |
| R020 | M2 | TikTok test set 上集群 + DWPose 轨迹预抽 | — | 10 段 | — | MUST | TODO | 备选 UBC-Fashion |
| R030 | M3 | 主表全矩阵 | 胜出 3–4 系统 × stride{1,2,4,8} | TikTok 10 段 | PSNR/SSIM/LPIPS/FVD + VBench | MUST | TODO | job array 并行 |
| R040 | M4 | B2 消融:参数化 + 速度来源 | linear/spline/perclip/amortized;解析 vs 差分速度 | 3–5 段 × stride{4,8} | 同主表 | MUST | TODO | |
| R041 | M4 | seeds 方差 | 主方法 + 最强 baseline × 3 seeds | 3 段 × stride 4 | 同主表 | MUST | TODO | appendix |
| R050 | M5 | 60fps 任意帧率 demo | INR 连续采样 | 挑 2–3 段 | 定性 | NICE | ~~TODO~~ | R012 终止,连同 R020–R041 一并关闭 |
| R051 | M5 | 运动幅度分桶分析 + 失败样例 | (R030 事后分析) | — | 分桶 LPIPS/MS | NICE | ~~TODO~~ | 同上 |
| **R100** | S3 | step3 查新:采样时 latent warp 融合 × pose 低fps | — | — | — | MUST | in-progress | 致命先验 = (a)低fps控制→全fps生成 (b)采样中 flow-warp latent 约束中间帧 (c)pose 动画时域超分 |
| **R101** | S3 | fusion dev:两轮(latent→x0 空间) | R1: latent×{α.15/.3}×{[.3,.9]/[0,.7]};R2: x0×{α.3/.6}×{[0,.85]/[.3,.85]} | video1 自驱动 | mid-PSNR/LPIPS 全局+fast桶 | MUST | **DONE — 方向终止 ❌** | R1(16498220)全灭:早窗口破坏噪声统计崩坏,温和配置 LPIPS 翻倍。R2(16498624)x0 融合:PSNR 15.59 追平 RIFE、fast 桶 14.63 略胜,**但 LPIPS 0.378 vs linear 0.343 全配置差 10%**(结构/纹理 trade-off 无免费区间)。预注册 G2 两轮未过 → 终止。详见 docs/experiments/round1_archive/step3_sampling_fusion.md |
| ~~R102~~ | S3 | ~~fusion pilot 3-case 终审~~ | — | — | — | — | 关闭 | R101 未过 G2,按预注册不跑 |
