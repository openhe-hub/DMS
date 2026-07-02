"""Step 1.4 -- diagnostic: inject synthetic jitter onto the REAL trajectories and
sweep the noise level. Disambiguates the Step-1 negative:

  - If learned-INR OVERTAKES the baseline as injected noise grows, the denoising
    mechanism works on real motion shapes -- real DWPose is simply too clean.
  - If it never overtakes, the synthetic motion prior does not match real motion
    (a genuine domain gap), independent of noise.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

import _paths  # noqa: F401
from _paths import TRAJ_DIR, CKPT_DIR, FIG_DIR
from dispose_siren.trajectory import load_npz
from dispose_siren.train import load_ckpt
from dispose_siren.eval_protocols import make_windows, protocol_holdout, protocol_pseudogt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--span", type=int, default=48)
    ap.add_argument("--step", type=int, default=24)
    ap.add_argument("--noises", type=float, nargs="+", default=[0, 2, 4, 8, 16, 32])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, meta = load_ckpt(os.path.join(CKPT_DIR, "amortized.pt"), device)
    all_w = []
    for t in sorted(glob.glob(os.path.join(TRAJ_DIR, "*.npz"))):
        pts, vis, m = load_npz(t)
        w = make_windows(pts, vis, span=args.span, step=args.step)
        if len(w):
            all_w.append(w)
    W = np.concatenate(all_w, 0)
    print(f"total windows={len(W)}  device={device}", flush=True)

    print("\n=== injected-noise sweep on REAL trajectories (lower=better) ===")
    print("Protocol A: held-out pos-MSE (px^2)   |   Protocol B: pseudo-GT vel-MSE")
    print(f"{'sigma_px':>8} | {'base_A':>10} {'learn_A':>10} {'winA':>7} | "
          f"{'base_B':>11} {'learn_B':>11} {'winB':>7}")
    print("-" * 78)
    rows = []
    for s in args.noises:
        rA = protocol_holdout(W, model, device, obs_noise=s)
        rB = protocol_pseudogt(W, model, device, obs_noise=s)
        baseA = min(rA["linint"], rA["gauss+lin"])
        baseB = min(rB["fd"], rB["fd+gauss"])
        wA = "LEARN" if rA["learned"] < baseA else "base"
        wB = "LEARN" if rB["learned"] < baseB else "base"
        print(f"{s:>8.1f} | {baseA:>10.2f} {rA['learned']:>10.2f} {wA:>7} | "
              f"{baseB:>11.1f} {rB['learned']:>11.1f} {wB:>7}", flush=True)
        rows.append({"sigma": s, "baseA": baseA, "learnA": rA["learned"], "winA": wA,
                     "baseB": baseB, "learnB": rB["learned"], "winB": wB})

    jpath = os.path.join(FIG_DIR, "step1_noise_sweep.json")
    with open(jpath, "w") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=2, default=float)
    print(f"\nwrote {jpath}", flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable ({e}); skipping figure", flush=True)
        return
    s = [r["sigma"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(s, [r["baseA"] for r in rows], "s-", color="tab:green", label="best baseline")
    ax[0].plot(s, [r["learnA"] for r in rows], "^-", color="tab:blue", label="learned-INR")
    ax[0].set_title("A: held-out pos-MSE vs injected jitter"); ax[0].set_xlabel("injected σ (px)")
    ax[0].set_ylabel("MSE (px²)"); ax[0].legend()
    ax[1].plot(s, [r["baseB"] for r in rows], "s-", color="tab:green", label="best baseline")
    ax[1].plot(s, [r["learnB"] for r in rows], "^-", color="tab:blue", label="learned-INR")
    ax[1].set_title("B: pseudo-GT vel-MSE vs injected jitter"); ax[1].set_xlabel("injected σ (px)")
    ax[1].set_ylabel("MSE"); ax[1].legend()
    fig.suptitle("Step 1 diagnostic: does the learned prior overtake baselines as REAL keypoints get noisier?")
    fig.tight_layout()
    fpath = os.path.join(FIG_DIR, "step1_noise_sweep.png")
    fig.savefig(fpath, dpi=120); plt.close(fig)
    print(f"wrote {fpath}", flush=True)


if __name__ == "__main__":
    main()
