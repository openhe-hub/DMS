"""R011/R012 -- pilot metrics: every full-grid system vs the self-driven GT.

Metrics per system (vs gt.pt on the shared sampling grid):
  psnr_all / psnr_obs / psnr_mid : PSNR on all / observed / in-between frames.
       The obs-vs-mid split is THE diagnostic: interpolation quality only
       shows on mid frames; obs frames measure the shared diffusion ceiling.
  lpips_all / lpips_mid          : perceptual distance (AlexNet), lower better.
  warp    : optical-flow temporal-consistency residual of the OUTPUT video
            (Farneback, from 07) -- smoothness proxy, lower better.
  div_ub  : pixel MSE vs the orig_s1 upper-bound output (same seed) -- how far
            the low-fps system drifts from the full-pose generation.

Systems on the coarse grid only (orig_s4/8) get obs-frame metrics only.
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


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def frame_psnrs(gen, gt):
    return [psnr(gen[i].numpy(), gt[i].numpy()) for i in range(len(gen))]


def warp_error(frames):
    T = frames.shape[0]
    errs = []
    for t in range(T - 1):
        a = cv2.cvtColor(frames[t].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY)
        b = cv2.cvtColor(frames[t + 1].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        h, w = a.shape
        gx, gy = np.meshgrid(np.arange(w), np.arange(h))
        warped = cv2.remap(a.astype(np.float32),
                           (gx + flow[..., 0]).astype(np.float32),
                           (gy + flow[..., 1]).astype(np.float32), cv2.INTER_LINEAR)
        errs.append(float(np.mean((warped - b.astype(np.float32)) ** 2)))
    return float(np.mean(errs))


def lpips_scores(gen, gt, net, device, batch=8):
    outs = []
    with torch.no_grad():
        for i in range(0, len(gen), batch):
            a = gen[i:i + batch].to(device).float() / 127.5 - 1
            b = gt[i:i + batch].to(device).float() / 127.5 - 1
            outs.append(net(a, b).flatten().cpu())
    return torch.cat(outs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--no_lpips", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = None
    if not args.no_lpips:
        import lpips
        net = lpips.LPIPS(net="alex").to(device)

    all_results = {}
    for case in args.cases:
        cd = os.path.join(OUT, "pilot", f"case{case}")
        gt_p = os.path.join(cd, "gt.pt")
        if not os.path.exists(gt_p):
            print(f"case{case}: no gt.pt, skip", flush=True)
            continue
        gt = torch.load(gt_p)
        res = {}
        print(f"\n===== case{case} (GT {tuple(gt.shape)}) =====", flush=True)

        ub_p = os.path.join(cd, "orig_s1.pt")
        ub = torch.load(ub_p) if os.path.exists(ub_p) else None

        for f in sorted(glob.glob(os.path.join(cd, "*_s*.pt"))):
            name = os.path.basename(f)[:-3]
            if name in ("gt",):
                continue
            m = re.match(r"(\w+)_s(\d+)$", name)
            if not m:
                continue
            method, s = m.group(1), int(m.group(2))
            gen = torch.load(f)

            if method == "orig" and s > 1:
                # coarse grid: frame k corresponds to full-grid index k*s
                n = min(len(gen), (len(gt) - 1) // s + 1)
                ps = [psnr(gen[k].numpy(), gt[k * s].numpy()) for k in range(n)]
                r = {"grid": "coarse", "stride": s, "psnr_obs": float(np.mean(ps)),
                     "warp": warp_error(gen)}
                res[name] = r
                print(f"  {name:<12} psnr_obs={r['psnr_obs']:.2f}  warp={r['warp']:.1f}",
                      flush=True)
                continue

            n = min(len(gen), len(gt))
            g, t = gen[:n], gt[:n]
            ps = np.array(frame_psnrs(g, t))
            idx = np.arange(n)
            obs = idx % s == 0 if s > 1 else np.ones(n, bool)
            mid = ~obs
            r = {"grid": "full", "stride": s,
                 "psnr_all": float(ps.mean()),
                 "psnr_obs": float(ps[obs].mean()),
                 "psnr_mid": float(ps[mid].mean()) if mid.any() else None,
                 "warp": warp_error(g)}
            if net is not None:
                lp = lpips_scores(g, t, net, device)
                r["lpips_all"] = float(lp.mean())
                r["lpips_mid"] = float(lp[mid].mean()) if mid.any() else None
            if ub is not None and s > 1:
                nn_ = min(n, len(ub))
                r["div_ub"] = float(np.mean((g[:nn_].float().numpy()
                                             - ub[:nn_].float().numpy()) ** 2))
            res[name] = r
            lpstr = f" lpips={r.get('lpips_all'):.4f}" if net is not None else ""
            mid_s = f"{r['psnr_mid']:.2f}" if r["psnr_mid"] else "-"
            print(f"  {name:<12} psnr all={r['psnr_all']:.2f} obs={r['psnr_obs']:.2f} "
                  f"mid={mid_s}  warp={r['warp']:.1f}{lpstr}", flush=True)

        all_results[f"case{case}"] = res

    os.makedirs(FIG_DIR, exist_ok=True)
    jp = os.path.join(FIG_DIR, "step2_pilot_metrics.json")
    json.dump(all_results, open(jp, "w"), indent=2)
    print(f"\nwrote {jp}", flush=True)


if __name__ == "__main__":
    main()
