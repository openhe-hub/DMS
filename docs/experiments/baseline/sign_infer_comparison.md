# Baseline — 手语驱动的 MimicMotion vs DisPose 推理定性对比

> 目标:在最难的内容(手语,手部精细动作)上确立后续工作的 baseline 生成配置。
> 对比对象:手语版 MimicMotion(jubail2 fork,graft 内建) vs DisPose(graft 开关移植版)。
> 日期:2026-07-04 ~ 07-05。环境:Jubail A100(zl6890 = DisPose / yf23 = MimicMotion)。

---

## 1. 动机与背景

- 定性实验的源头输入选**手语视频**(how2sign):手指精细动作是 pose 驱动生成公认的最难内容,最能区分控制方案优劣。
- jubail2 上的手语版 MimicMotion 是上游改版,其 pose 预处理含 `graft_pose_v2`(OpenHE 自研):
  以**参考图 pose 为底**(身体/脸/腿冻结),按左右肩颈距离比例只移植驱动视频的**手臂(肘/腕)+手部**关键点,
  另加 15% 头部混合(nose/eyes/ears,>0.20 会出 artifact)。这是跨身份手语生成的身份保持手段。
- 为了公平对比,把 `graft_pose_v2` 移植进 DisPose 做成**开关**,两边共享同一 pose 条件构建方式。

### 关键代码事实(对判读重要)

- DisPose 的运动场增强分支(traj_flow / CMP 稠密流 / point embedding)**只用 body 18 点**,
  DWPose 检测出的手指关键点被丢弃(`preprocess.py` 只收集 `bodies`/`faces`);
  手指信息仅存在于两模型共用的骨架图分支。→ 任何手部差异都是运动场的**间接**效应。
- graft 开关(`graft_pose: true`)同时作用于骨架图与运动场输入的 point list,控制链自洽。

## 2. 实现与素材

| 项 | 内容 |
|---|---|
| graft 移植 | `mimicmotion/dwpose/graft.py`(逐行搬运 jubail2 fork);`get_video_pose(graft=)` 接入;config 字段 `graft_pose`(默认 false,完全保持原行为) |
| 驱动视频 | how2sign 3 条 × 8s 剪辑:`1aRNY8wFqa0_32-8`、`5ok8y3eheq8_7-1`、`DI6T6tbk3r0_15-5`(均 192 帧;前两条 24fps,第三条 23.976fps,误差 0.1% 忽略)。存放 `assets/example_data/sign_videos/` |
| 参考图 | `test2.jpg`(jubail2 批量任务同款,跨身份);备用 `ref_01~05.png`(identity 实验用,已拷入 refs/) |
| 共同参数 | 25 步 / CFG 2.0 / tile 16 / overlap 6 / seed 42 / square 576 / noise_aug 0 |

注:how2sign 源里 `CanYlZX_uyE_6-5` 文件本身损坏(moov atom 缺失,md5 与远端一致证实源头即坏),弃用。

## 3. 实验一:graft 开关冒烟测试(15fps / stride 2)

**协议**:DisPose graft on vs off,3 case,其余参数全同(输出 96/97 帧 @15fps)。
作业:jubail 16502243(on)/ 16502244(off),两张 A100 并行,单 case 去噪约 4-5 min。

**结果**(`outputs/sign_graft_smoke/graft_on|graft_off/`):

| 维度 | graft on | graft off |
|---|---|---|
| 身份保持 | 脸部/头部牢固锁定参考图 | **明显身份漂移**(头随驱动歪,t≈5s 脸型已不像参考) |
| 动作传递 | 手臂+手正确跟随 | 全身跟随(含不需要的躯干/头部运动) |
| 手形 | 略干净 | 正常 |

**判读**:开关行为正确;graft 是跨身份场景的必需组件。

## 4. 实验二:对齐三方对比(24fps / stride 1,主实验)

**协议**:源 192 帧全部作为驱动(stride 1),输出 fps 24 → **与源逐帧对齐、等速**。
- DisPose graft on:jubail 16507781,config `configs/test_sign_align.yaml`
- MimicMotion 手语版:jubail2 16507783,`batch_process.py --sample-stride 1 --fps 24`,其余参数对齐
- 帧数:源 192 / DisPose 192(内部 +1 参考首帧,保存时已丢)/ MimicMotion 193(padding 裁剪 off-by-one,**拼接时掐首帧**)

**交付物**:`outputs/sign_cmp_aligned/cmp_*.mp4`(3 条,1728×576,三栏 = Source | MimicMotion | DisPose graft,
源居中裁方 + drawtext 标注,全部 H.264;单模型原始输出在 `raw/` 下)。

**定性结论(用户审阅,2026-07-05)**:**DisPose + graft on 最优**,作为后续 baseline。

| 维度 | MimicMotion(手语版) | DisPose + graft |
|---|---|---|
| 手指结构 | 运动中涂抹/粘连(如交叉手指帧左手糊掉) | **基本还原源动作手形** |
| 伪影 | 画面底部反复出现文字状 watermark 伪影 | 无 |
| 身份保持 | 好(graft 内建) | 好(graft 移植生效) |

**观察(值得写进论文)**:DisPose 运动场不含手指关键点,手形却系统性更好 →
手臂/手腕轨迹的稠密流控制**间接**稳定了手部区域生成。结合 graft 逻辑本身
(在骨架层费力做手部重定向,正因为没有独立的手部运动控制通道),
"手部控制在现有运动场方案中缺位"是 SIREN/step3 方向的直接 motivation 素材。

## 5. 工程备忘

- 两账号分工:DisPose @ jubail(zl6890),手语 MimicMotion @ jubail2(yf23,`chatsign-175/MimicMotion/zhewen_cmp/`);jubail2 无 DisPose 仓库。
- 所有 sbatch 必带 `--mail-type=ALL --mail-user=zh3510@nyu.edu`;登录节点 slurm 命令需 `bash -lc`。
- MimicMotion 输出为 mpeg4 编码(VS Code 播不了),拉回后统一转 H.264(`libx264 -crf 16 -pix_fmt yuv420p`)。
- 本地↔集群链路不稳:mux socket 会坏(删 `~/.ssh/sockets/*` 重建);大文件用 `rsync --partial` 多轮重试 + **md5 核对**(出现过静默截断)。
- jubail2 旧脚本里 `ROOT=/scratch/yf23/MimicMotion` 已失效,实际路径在 `chatsign-175/` 下。

## 6. 下一步(待定规模后执行)

1. 正式定性实验:how2sign_100 挑 10~15 条(覆盖签者/语速/手形复杂度:双手交叉、指拼等);
2. 保持三栏交付(graft off 的身份漂移已在冒烟版定案,不再每条重跑);
3. 参考图增加 1~2 张 `ref_0x` 验证结论不依赖特定身份;
4. 回到 SIREN/step3 主线,以 **DisPose + graft** 为 baseline 叠加改进。
