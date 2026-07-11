# 定性对比:MimicMotion vs DisPose+graft+SIREN(grid figure)

> `siren_module.md` §5 两张定量表的配套定性图:在 8 个 asl27k 难例词条上,
> 同一源时刻三画幅并排(Source | MimicMotion | DisPose+SIREN)。
> 日期:2026-07-11。脚本:`scripts/hand_pilot/make_qual_grid.py`(本地 Mac,
> ffmpeg 抽帧 + PIL 排版;本地 homebrew ffmpeg 无 drawtext,故不用 filter 排版)。

![MimicMotion vs DisPose+SIREN](figs/mm_vs_siren_grid.png)

图:`figs/mm_vs_siren_grid.png`(同份在 `outputs/sign_cmp_hard27k/figs/`)。

## 1. 材料与对齐口径

| 列 | 来源 | 帧对齐 |
|---|---|---|
| Source | `assets/example_data/sign_videos/hard27k_orig/{id}.mp4`(640×360,中心裁方) | 帧 n |
| MimicMotion | P2 全量跑(jubail2),本地副本 `outputs/sign_cmp_hard27k/raw/mimicmotion/` | **帧 n+1**(输出首帧是 padding,501 vs 源 500 帧,实测核对) |
| DisPose+SIREN | **best-of-≤3** 交付(集群 `outputs/sign_siren_best/best/`,8 条已拉回本地同名目录) | 帧 n(内部参考帧存盘时已丢,逐帧对齐) |

与 §5.2 同一批生成视频、同一 best-of 选择;三列取**同一源时刻**,不做逐列
挑帧,可作配对对比读。

## 2. 选帧协议(showcase,须披露)

8 个词条沿用 `baseline/qualitative.md` §7 `mm_failures_grid.png` 的 clip 选择
(MimicMotion 失败最典型的难例)。帧号在每条 clip 的 12 候选帧接触表
(Source/MM/SIREN 三行)上目检重挑,判据 = **同一时刻 MM 失败明显且 SIREN
手部干净**;4 条旧帧已最优保留,4 条更换:

| 词条 | clip:帧 | 看点 |
|---|---|---|
| vulcanise | `0bsujxxpwd:425` | MM 蓝色文字水印爆发;SIREN 干净(沿用) |
| lethargic | `07imqjgcxc:99` | MM 涂鸦背景 + 断臂残肢;SIREN 干净(沿用) |
| cowboy | `05tcw2nou9:35`(原 64) | 源"手枪"手形 SIREN 精准复现;MM 塌成掌糊 |
| open book | `0byrxo0heb:56` | MM 整背景被源蓝色淹没;SIREN 干净(沿用) |
| backlight | `0db3uk2cqw:150` | MM 手中幻觉出黄色物体;SIREN 双手指点清晰(沿用) |
| hump | `0bcxsenqga:166`(原 174) | 源拱形五指罩掌 SIREN 复现;MM 模糊爪形 |
| turn off (tv) | `0ihmqp5iz6:53`(原 28) | 源 L 形手搭腕 SIREN 复现;MM 糊爪 |
| grade | `0ejbehccd4:21`(原 23) | MM 文字块 + 拳头拖影;SIREN 平掌搭臂清晰 |

**披露口径(论文 caption 必须带)**:定性图为 showcase——clip 取 MimicMotion
失败典型例,帧为人工挑选;SIREN 列 = best-of-≤3 seeds 按 DWPose 手部置信度
重排(与 §5.2 同一披露)。总体分布性结论以 `siren_module.md` §5 的 109 条
配对统计为准(mean_hand_conf 101/109,p=6.4e-22)。

## 3. 读图要点

- MimicMotion 的五类失败(文字爆发 / 背景渗漏 / 幻觉物体 / 断臂 / 糊手,
  taxonomy 见 `baseline/qualitative.md` §7)在 SIREN 列全部不出现;
- 新挑的 4 帧(cowboy/hump/turn off/grade)专门展示**手形结构级差异**:
  同一时刻 SIREN 复现源手形(手枪 / 拱形罩掌 / L 形搭腕 / 平掌),MM 为
  blob/爪形——这是 mean_hand_conf 配对优势的可视化;
- 身份列稳定(hijab 参考 `test2.jpg`),与 CSIM 结论一致。

## 4. 复现

```bash
# 前置:8 条 SIREN best 视频在 outputs/sign_siren_best/best/(可从集群 rsync)
python3 scripts/hand_pilot/make_qual_grid.py
# → outputs/sign_cmp_hard27k/figs/mm_vs_siren_grid.png (1752x1192)
```

换 clip/帧改脚本内 `SPECS`;候选帧接触表的生成逻辑见 git 历史或按 §2 协议
重写(12 均匀候选 + 原帧,三行并排目检)。
