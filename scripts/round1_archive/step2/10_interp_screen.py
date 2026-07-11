"""R010 -- pose-level INTERPOLATION screening on real DWPose trajectories.

Question: driving poses arrive at temporal stride s (low-fps driving); which
continuous representation best reconstructs the missing intermediate frames?
Winner goes into the R011 video pilot as the control-signal interpolator.

Protocol (self-supervised, real detections as target):
  window span = 15*s+1 dense frames; observed = 16 detections at stride s
  (both window ends included); held-out = ALL intermediate real detections.
  Score = position MSE (px^2) on held-out frames. Same detection-noise floor
  for every method -> ranking is fair.

Honesty rules:
  - per-clip SIREN configs (w0, lam) tuned on video1 (dev) ONLY; frozen for
    video2+3 (test). Best config reported.
  - gauss+linear sigma is tuned on the eval set itself = baseline best case.
  - amortized FiLM-SIREN = step-1 synthetic ckpt, unchanged (16-obs encoder).
"""
import argparse
import json
import os

import numpy as np
import torch

import _paths  # noqa: F401
from _paths import FIG_DIR, TRAJ_DIR, CKPT_DIR

from dispose_siren import baselines as B
from dispose_siren.round1 import normalize as Z
from dispose_siren.round1.eval_protocols import make_windows
from dispose_siren.round1.interp import natural_cubic_pos, perclip_fit_decode
from dispose_siren.round1.train import load_ckpt
from dispose_siren.round1.trajectory import load_npz

N_OBS = 16


def build_sets(stride, dev_videos=("video1",), test_videos=("video2", "video3")):
    span = 15 * stride + 1
    step = max(span // 2, 1)

    def collect(names):
        ws = []
        for v in names:
            pts, vis, _ = load_npz(os.path.join(TRAJ_DIR, f"{v}.npz"))
            w = make_windows(pts, vis, span=span, step=step)
            if len(w):
                ws.append(w)
        return np.concatenate(ws, 0) if ws else np.zeros((0, span, 2))

    return span, collect(dev_videos), collect(test_videos)


def eval_set(windows, stride, span, model, device, perclip_cfg=None,
             perclip_sweep=None, steps=800):
    """Returns dict method -> MSE, plus best per-clip config if sweeping."""
    obs_i = np.arange(0, span, stride)
    assert len(obs_i) == N_OBS and obs_i[-1] == span - 1
    hold_i = np.setdiff1d(np.arange(span), obs_i)
    tau = np.arange(span) / (span - 1)
    tf, th = tau[obs_i], tau[hold_i]
    obs = windows[:, obs_i]
    gt = windows[:, hold_i]

    def mse(p):
        return float(np.mean((p - gt) ** 2))

    out = {"S": len(windows)}
    out["linear"] = mse(B.linint_pos(obs, tf, th))
    ge, gsig, gpos = B.best_sigma_gauss_pos(obs, tf, th, gt)
    out["gauss+lin"] = float(ge)
    out["gauss_sigma"] = float(gsig)
    out["spline"] = mse(natural_cubic_pos(obs, hold_i / stride))

    if model is not None:
        _, film, mu, s = Z.infer(model, obs, th, device)
        out["amortized"] = mse(Z.decode_pos_px(model, film, mu, s, th, device))

    if perclip_sweep:
        best = None
        for w0 in perclip_sweep["w0"]:
            for lam in perclip_sweep["lam"]:
                p = perclip_fit_decode(obs, tf, th, w0=w0, lam=lam,
                                       steps=perclip_sweep.get("steps", 600))
                e = mse(p)
                print(f"    sweep w0={w0:<4} lam={lam:<5} -> {e:.2f}", flush=True)
                if best is None or e < best[0]:
                    best = (e, {"w0": w0, "lam": lam})
        out["perclip"] = float(best[0])
        out["perclip_cfg"] = best[1]
    elif perclip_cfg:
        p = perclip_fit_decode(obs, tf, th, w0=perclip_cfg["w0"],
                               lam=perclip_cfg["lam"], steps=steps)
        out["perclip"] = mse(p)
        out["perclip_cfg"] = perclip_cfg
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strides", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    device = "cpu"

    ck = os.path.join(CKPT_DIR, "amortized.pt")
    model = None
    if os.path.exists(ck):
        model, meta = load_ckpt(ck, device)
        print(f"loaded amortized ckpt ({meta})", flush=True)
    else:
        print("WARNING: no amortized ckpt, skipping that method", flush=True)

    sweep = {"w0": [3, 5, 8, 12], "lam": [0.0, 0.1, 1.0], "steps": 600}
    results = {}
    for s in args.strides:
        span, dev, test = build_sets(s)
        print(f"\n=== stride {s} (span {span})  dev S={len(dev)} test S={len(test)} ===",
              flush=True)
        print("  [dev sweep on video1]", flush=True)
        rdev = eval_set(dev, s, span, model, device, perclip_sweep=sweep)
        cfg = rdev["perclip_cfg"]
        print(f"  dev best per-clip cfg = {cfg}", flush=True)
        print("  [test on video2+3, frozen cfg]", flush=True)
        rtest = eval_set(test, s, span, model, device, perclip_cfg=cfg,
                         steps=args.steps)
        results[s] = {"dev": rdev, "test": rtest}

        hdr = ["linear", "spline", "gauss+lin", "perclip"] + (
            ["amortized"] if model is not None else [])
        print(f"\n  stride={s} TEST held-out pos-MSE (px^2), S={rtest['S']}:")
        for k in hdr:
            rel = rtest[k] / rtest["linear"]
            print(f"    {k:<10} {rtest[k]:>10.2f}   (x{rel:.2f} vs linear)", flush=True)

    jpath = os.path.join(FIG_DIR, "step2_interp_screen.json")
    with open(jpath, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nwrote {jpath}", flush=True)


if __name__ == "__main__":
    main()
