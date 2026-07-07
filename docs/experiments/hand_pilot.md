# Hand-channel pilot — Gate A/B + 109-clip overfit & scaling (DRAFT)

> Hypothesis: DisPose's motion field carries only 18 body points; on sign
> language the missing HAND control channel (0→1, not covered by the step1/2
> kill verdict) plus a NIAF-style train-once trajectory prior may improve hand
> generation. Pre-registered kill chain: Gate A (channel causality) → Gate B
> (noise headroom) → P0 (capacity) → scaling slope (asl50k go/no-go).
> Date: 2026-07-07. Env: jubail A100 (extraction/generation/scaling) + local
> MPS (P0). Code: `src/dispose_siren/hand_*.py`, `scripts/hand_pilot/`,
> `mimicmotion/dwpose/hand_control.py`.

## 0. TL;DR (updating as results land)

- **V1 (hand order)**: `hands[0]=LEFT, hands[1]=RIGHT` CONFIRMED on real data
  (median wrist-match dist 0.01 vs 0.13+ opposite side; job 16540596).
  `pose_extract.py`'s old comment was wrong; `graft.py` was right.
- **Gate B (noise headroom): PASS.** Hand jitter 3.78 px vs body wrist
  2.14 px (ratio 1.76); dropout 5.9% (146 gaps, median 4, p90 48 frames);
  fingertip conf 0.63. Plus a harsher failure mode found while debugging P0:
  DWPose emits **collapsed ~2 px "hands" at 0.3–0.5 confidence** — structural
  garbage, not jitter. The denoising/inpainting headroom is real.
- **Regression gates: PASS** (blocking, job 16540597). K-generalized
  pose2track/points_to_flows bit-exact vs frozen originals at K=18 (synthetic
  + real); dead-hands (all conf<thr) injection is inert to the bit; original
  step2 `12_equiv_check` still PASS. Live hands change traj_flow with
  **~13× the body flow magnitude** (blurred max 1.32 vs 0.10) — the
  saturation risk the smooth/tips arms probe.
- **P0 (capacity): PASS.** Conf-weighted L_pos 0.0234 canonical² = **2.1×**
  the Gate B noise floor (0.0113), velocity roughness 0.96 (no ringing),
  w0=15 frozen. (First attempt diverged at loss_vel~1e17 — cause was the
  collapsed-hand windows above; fixed by min_bone_px=8 + |canon|≤12 gates,
  2273→2125 windows, and grad clipping.)
- **Scaling (v1, partial)**: healthy seeds show clean decrease
  16→32→64 clips: held-out MSE 2.4 → 1.2 → 0.85 (spline 0.042, linear
  0.056) — losing to spline at pilot scale as pre-registered/expected;
  naive power-law slope ≈ −0.75 → spline crossing ≈ a few ×10³ clips,
  well inside asl50k. Seed-0 runs collapsed in training (no LR warmup);
  **v2 with warmup + 3 seeds at every size is rerunning — numbers below
  will be replaced.**
- **Gate A (channel causality)**: 8 cases × {off, raw, smooth σ=1.5} running
  (jobs 16541178/79/80). Verdict = visual (≥ half the cases show consistent
  hand-region change ⇒ channel live); diagnostics via 42_gate_a_inspect.

## 1. Setup

- **Data**: 109 asl27k hard-case source videos (the P2 benchmark), DWPose
  body+hands+conf @stride 1 (`30_extract_hand_poses.py`, ~40 MB npz).
  Windows: span 32 / step 8, detected-all + ≥80% frames conf≥0.3 +
  median bone ≥ 8 px + |canonical| ≤ 12 → **2125 windows / 108 clips**
  (~19.7 windows/hand-side/clip). Canonical frame: conf-weighted wrist
  origin, median wrist→MCP scale (matches `metrics._norm_hand` semantics).
- **Model**: `HandSetSIREN` 1.0 M params — transformer encoder (d128×3) over
  per-frame tokens (21×(x,y,conf) + wrist/elbow + log-scale + side + τ),
  learnable queries Q=L(G+1)=12 (grouped per SIREN layer, NIAF §3.1.3),
  zero-init projections → (γ,β) modulating shared SIREN meta-params
  (Ŵ=W⊙(1+γ), b̂=b+β, NIAF eq. 7), H128×L4, w0 15. Analytic velocity via
  the closed-form cos recursion (NIAF §3.2), no autograd double-backward.
- **Training**: conf-weighted Gaussian pseudo-clean target (σ=1.25);
  conf-weighted L_pos + 0.5·analytic L_vel; obs patterns = uniform-16 with
  phase jitter (70%) / contiguous gap 2–8 (30%); noise aug from measured
  jitter (0, ½, 1, 2 × 0.106 hand-units); Adam 1e-3, 5% linear warmup +
  cosine, grad-clip 1.0, wd 1e-4.
- **Protocols** (`hand_eval.py`): even/odd holdout (GT only where held-out
  conf ≥ 0.3) and synthetic-honest gap inpainting (all-high-conf windows,
  gap lengths sampled from the measured histogram) vs linear / non-uniform
  natural cubic spline / best-σ gauss (σ tuned on eval data = best case).
  Clip-level splits only; 24 held-out clips frozen in `windows/split.json`.
  Caveat: signer identity unknown ⇒ clip-independent, not provably
  signer-independent.

## 2. Gate B numbers

| quantity | hands | body (wrist / elbow / shoulder) |
|---|---|---|
| jitter, smoothing residual (px) | **3.78** | 2.14 / 0.93 / 0.39 |
| jitter, relative to hand scale | 0.106 | — |
| dropout rate (person present) | **5.9%** | ~0 |
| gaps | 146 (med 4, p90 48, max 97) | — |
| conf: wrist / MCP / tips | 0.74 / 0.73 / **0.63** | — |

Kill test (hands ≤1.2× body AND dropout <2%): **not triggered → PASS.**
Worst clips (Gate A extras): `0glzpsqsrl, 0ddpfhlmff, 0b247hvyxo`.

## 3. Results (PENDING sections)

### 3.1 Gate A — PENDING (jobs 16541178/79/80 + 42_gate_a_inspect)

### 3.2 Scaling v2 — PENDING (warmup fix, 3 seeds × {16,32,64,84})

### 3.3 Decision — PENDING
Pre-registered: asl50k justified iff slope < −0.05 AND extrapolated spline
crossing < 50k clips AND Gate A live. Contamination note: every pilot
checkpoint trains on P2-benchmark clips → throwaway; the real model trains
on asl50k minus the 109 / same-signer / same-word.

## 4. Reproducibility notes

- jubail env: CPU `onnxruntime` (pulled in by insightface during metrics
  work) shadowed `onnxruntime-gpu` → DWPose silently on CPU. Fixed by
  uninstall + `pip install --force-reinstall --no-deps onnxruntime-gpu==1.19.2`
  (the two wheels share the `onnxruntime/` dir; plain uninstall breaks both).
- `pose2track`/`points_to_flows` now take `n_points=18`; K=18 is bit-exact
  (frozen-reference test in `41_equiv_check_hands.py`).
- hand_flow switches (yaml per test_case): `hand_flow`, `hand_flow_smooth`,
  `hand_conf_thr`, `hand_kp_subset: all|tips`; point-adapter branch stays
  body-18 by construction.
- Jobs: extract 16540596 (~35 min A100), check 16540597, Gate A arms
  16541178-80, scaling v1 16541400 (~25 min).
