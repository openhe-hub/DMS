"""Step 1 / Probe A metrics -- quantify how the generated video changes as
motion-field jitter grows.

  - divergence  : mean per-pixel MSE vs the sigma=0 output (same frames, same
                  diffusion seed) -> how much the trajectory perturbation moves
                  the video at all.
  - warp_error  : optical-flow temporal-consistency residual (cv2 Farneback);
                  proxy for flicker/jitter in the OUTPUT video. Lower = smoother.

Verdict logic: if divergence stays ~flat as jitter grows, the diffusion absorbs
trajectory differences -> pose-level motion-field improvements won't propagate.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import torch
import cv2

import _paths  # noqa: F401
from _paths import OUT, FIG_DIR


def gray(frame_chw_uint8):
    img = frame_chw_uint8.permute(1, 2, 0).numpy()           # H,W,3
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def warp_error(frames):
    """frames: (T,3,H,W) uint8 tensor -> mean optical-flow warp residual (MSE)."""
    T = frames.shape[0]
    errs = []
    for t in range(T - 1):
        a = gray(frames[t]); b = gray(frames[t + 1])
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        h, w = a.shape
        gx, gy = np.meshgrid(np.arange(w), np.arange(h))
        mapx = (gx + flow[..., 0]).astype(np.float32)
        mapy = (gy + flow[..., 1]).astype(np.float32)
        warped = cv2.remap(a.astype(np.float32), mapx, mapy, cv2.INTER_LINEAR)
        errs.append(float(np.mean((warped - b.astype(np.float32)) ** 2)))
    return float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    probe_dir = os.path.join(OUT, "video_probe")
    files = glob.glob(os.path.join(probe_dir, "frames_sigma*.pt"))
    if not files:
        raise SystemExit(f"no frames in {probe_dir} -- run 06 first")
    by_sigma = {}
    for f in files:
        s = int(re.search(r"sigma(\d+)", f).group(1))
        by_sigma[s] = torch.load(f)
    sigmas = sorted(by_sigma)
    base = by_sigma[sigmas[0]].float()

    print(f"{'sigma':>6} | {'warp_error':>12} | {'divergence_vs_0':>16}")
    print("-" * 42)
    rows = []
    for s in sigmas:
        fr = by_sigma[s]
        we = warp_error(fr)
        T = min(fr.shape[0], base.shape[0])
        div = float(np.mean((fr.float()[:T].numpy() - base[:T].numpy()) ** 2))
        rows.append({"sigma": s, "warp_error": we, "divergence": div})
        print(f"{s:>6} | {we:>12.2f} | {div:>16.2f}", flush=True)

    # sensitivity slopes (per unit sigma, vs the sigma=0 reference)
    ss = np.array([r["sigma"] for r in rows], float)
    div = np.array([r["divergence"] for r in rows])
    we = np.array([r["warp_error"] for r in rows])
    div_slope = float(np.polyfit(ss, div, 1)[0]) if len(ss) > 1 else 0.0
    we_slope = float(np.polyfit(ss, we, 1)[0]) if len(ss) > 1 else 0.0
    # relative divergence at max jitter: how far the video moved vs its own pixel variance
    rel_div_max = float(div[-1] / (base.numpy().var() + 1e-6))
    print("\n" + "=" * 60)
    print(f"divergence slope = {div_slope:.3f} / px-jitter")
    print(f"warp_error slope = {we_slope:.3f} / px-jitter")
    print(f"relative divergence @ max jitter (σ={int(ss[-1])}) = {rel_div_max:.4f} "
          f"(fraction of output pixel variance)")
    print("Interpretation: rel-divergence ~0 => diffusion absorbs trajectory")
    print("  differences => pose-level motion-field route is GATED OFF.")
    print("=" * 60, flush=True)

    out = {"rows": rows, "div_slope": div_slope, "we_slope": we_slope,
           "rel_div_max": rel_div_max}
    jpath = os.path.join(FIG_DIR, "step1_video_probe.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"wrote {jpath}", flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable ({e}); skipping figure", flush=True)
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(ss, div, "o-", color="tab:red"); ax[0].set_title("output divergence vs σ=0")
    ax[0].set_xlabel("injected keypoint jitter σ (px)"); ax[0].set_ylabel("pixel MSE")
    ax[1].plot(ss, we, "s-", color="tab:purple"); ax[1].set_title("output temporal warp-error")
    ax[1].set_xlabel("injected keypoint jitter σ (px)"); ax[1].set_ylabel("warp MSE")
    fig.suptitle("Probe A: does motion-field trajectory quality propagate to the video?")
    fig.tight_layout()
    fpath = os.path.join(FIG_DIR, "step1_video_probe.png")
    fig.savefig(fpath, dpi=120); plt.close(fig)
    print(f"wrote {fpath}", flush=True)


if __name__ == "__main__":
    main()
