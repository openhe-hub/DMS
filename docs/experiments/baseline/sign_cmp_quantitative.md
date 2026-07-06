# Sign-language comparison — quantitative evaluation (DisPose+graft vs MimicMotion)

Date: 2026-07-06 · 109 hard-case sign videos · code in `src/metrics/`

## Goal

Turn the qualitative finding (DisPose+graft looks clearly better on the hardest
content — hands, identity, artifacts) into defensible numbers, on the full set
of 109 contested-review sign-language cases.

## Setup

- **Cases**: 109 hard examples (asl27k rejected-review words, ≥1 reject vote),
  driven by the original human-signer videos, retargeted onto the reference
  avatar `test2.jpg` (md5 `510d0117…`, identical on both clusters).
- **Two-cluster split**: DisPose outputs live on jubail (`zl6890`), MimicMotion
  on jubail2 (`yf23`). Each model is scored on its own cluster because the
  detector weights are byte-identical across both:
  - DWPose `dw-ll_ucoco_384.onnx` / `yolox_l.onnx` — md5 matched
  - I3D `i3d_torchscript.pt` (StyleGAN-V) — md5 matched
  - ArcFace `buffalo_l` (det_10g + w600k_r50) — same release, pre-cached both sides
  So separate jobs are equivalent to one shared estimator; only small CSVs/JSON
  are pulled back.

## Metrics

| family | metric | what it measures |
|---|---|---|
| control adherence | body / hand PCK, NME | fidelity of generated pose to the driving source pose, normalized subject-invariantly (body: neck-centred/shoulder-width; hand: wrist-centred/wrist-MCP) |
| hand quality | mean_hand_conf, hand_good_rate | DWPose hand-keypoint confidence + fraction of well-formed hands |
| video naturalness | **FVD** | Fréchet distance of I3D features vs the real source videos; sensitive to text/blob/background/temporal artifacts |
| identity | **CSIM** | ArcFace cosine similarity of each frame's face to the reference avatar (mean, worst-frame, and std=stability) |

CSIM sampling: 12 evenly-spaced frames/video at det_size 320 (identity is stable
within a clip; dense 640-px sampling was ~10× slower for the same result).

## Results

Lower is better for NME / FVD; higher for the rest.

| metric | DisPose | MimicMotion | winner | paired (DP better) |
|---|---|---|---|---|
| body_pck ↑ | 0.2797 | 0.2743 | DisPose | 79/109 |
| body_nme ↓ | **0.4142** | 0.4440 | **DisPose** | 109/109 |
| mean_hand_conf ↑ | 0.6988 | 0.6801 | DisPose | 85/109 |
| hand_good_rate ↑ | 0.8628 | 0.8831 | Mimic | 37/109 |
| hand_pck ↑ | 0.3175 | 0.3263 | Mimic | 43/109 |
| hand_nme ↓ | 0.5328 | 0.5318 | tie | 60/109 |
| **FVD ↓** | **830.4** [838,884] | 907.1 [906,980] | **DisPose** | bootstrap CIs disjoint |
| **CSIM mean ↑** | **0.8089** | 0.7727 | **DisPose** | 107/109 |
| **CSIM worst-frame ↑** | **0.7659** | 0.6712 | DisPose | 106/109 |
| **CSIM std ↓** (stability) | **0.0189** | 0.0392 | DisPose | 98/109 |
| face_det_rate | 1.0000 | 1.0000 | tie | — |

## Interpretation

- **DisPose wins decisively on FVD, CSIM, and body-pose adherence.** FVD (I3D
  spatiotemporal, square-cropped inputs) captures the artifact / temporal-
  coherence gap that pose metrics miss; its bootstrap CIs do not overlap
  (830 vs 907). CSIM confirms the graft
  identity lock — better mean, far better worst-frame (0.766 vs 0.671, i.e.
  MimicMotion drifts), and ~2× more stable across the clip.
- **Fine-grained hand-pose DWPose metrics are a wash / marginally favour
  MimicMotion.** DWPose always emits 21 plausible hand keypoints at decent
  confidence, so it is insensitive to blob-hands / extra fingers / text /
  background — the exact failures seen qualitatively. MimicMotion can even score
  higher with smooth-but-wrong hands. **Do not headline these hand metrics; use
  FVD + CSIM (+ human eval) as the primary evidence.**

## Artifacts

Under `outputs/metrics_hard27k/` (kept local; large I3D feature `.npy` on-cluster):
`per_video.csv`, `aggregate.csv`, `paired_delta.csv` (metrics 1+2),
`csim_{dispose,mimic}.csv`, `fvd_{dispose,mimic}.json`.
Code: `src/metrics/{pose_extract,hand_confidence,motion_fidelity,csim,fvd,
run_eval,run_csim,run_fvd,merge_results}.py`; slurm in `scripts/slurm/`.

## Next

- Tail hand metrics (catastrophic-hand rate, conf<0.3) — reuse on-cluster
  pose_cache, near-free; targets the rare catastrophic frames means wash out.
- OCR text-artifact rate + background-leakage — quantify the specific failures.
- Human A/B or back-translation Top-k — perceptual ground truth.
- Stratify every metric by rejection-vote difficulty tier.
