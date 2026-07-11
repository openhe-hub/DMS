"""R102 verdict -- per-frame metrics with motion-magnitude bucketing.

Motion magnitude of GT frame t = mean visible-keypoint displacement (target-res
pixels) to its GT neighbors, from the SHARED cached detections (same source all
systems saw). Mid frames are split into terciles (slow/mid/fast) per (case,
stride); we report per-bucket mean PSNR / LPIPS for:

  linear_s{s} (step2)   continuous control, no fusion   [G2 reference]
  rife_s{s}   (step2)   post-hoc interpolation          [the one to beat]
  fusion_s{s} (step3)   sampling-time latent fusion     [ours]

Pre-registered G1 (stride=8, fast bucket): fusion must beat rife on BOTH
PSNR and LPIPS in >=2/3 cases, else the direction is dead.
"""
import argparse
import glob
import json
import os
import pickle

import numpy as np
import torch

import _paths  # noqa: F401
from _paths import FIG_DIR, OUT, STEP2_PILOT


def psnr_frames(gen, gt):
    g = gen.float()
    t = gt.float()
    mse = ((g - t) ** 2).flatten(1).mean(1)
    return 10 * torch.log10(255.0 ** 2 / mse.clamp(min=1e-8))


def lpips_frames(gen, gt, net, device, batch=8):
    out = []
    for i in range(0, gen.shape[0], batch):
        a = gen[i:i + batch].to(device).float() / 127.5 - 1
        b = gt[i:i + batch].to(device).float() / 127.5 - 1
        with torch.no_grad():
            out.append(net(a, b).flatten().cpu())
    return torch.cat(out)


def motion_per_frame(detected, h, w):
    """(T,) mean visible-kp displacement to GT neighbors, in pixels."""
    cand = np.stack([d["bodies"]["candidate"][:18] for d in detected])
    sub = np.stack([np.asarray(d["bodies"]["subset"])[0][:18]
                    for d in detected])
    cand = cand * np.array([w, h])
    T = len(cand)
    mo = np.zeros(T)
    for t in range(T):
        ds = []
        for u in (t - 1, t + 1):
            if 0 <= u < T:
                vis = (sub[t] >= 0) & (sub[u] >= 0)
                if vis.any():
                    ds.append(np.linalg.norm(
                        cand[u][vis] - cand[t][vis], axis=1).mean())
        mo[t] = np.mean(ds) if ds else 0.0
    return mo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--strides", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--no_lpips", action="store_true")
    ap.add_argument("--dev_glob", default=None,
                    help="also score step3 dev runs matching this tag glob, "
                         "e.g. 'fusion_s8_a*'")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = None
    if not args.no_lpips:
        import lpips
        net = lpips.LPIPS(net="alex").to(device)

    results = {}
    for c in args.cases:
        s2 = os.path.join(STEP2_PILOT, f"case{c}")
        s3 = os.path.join(OUT, "pilot", f"case{c}")
        gt = torch.load(os.path.join(s2, "gt.pt"))
        with open(os.path.join(s2, "detections.pkl"), "rb") as f:
            detected, _ = pickle.load(f)
        T, _, H, W = gt.shape
        motion = motion_per_frame(detected, H, W)

        runs = {}
        for s in args.strides:
            for tag, d in ((f"linear_s{s}", s2), (f"rife_s{s}", s2),
                           (f"fusion_s{s}", s3)):
                p = os.path.join(d, f"{tag}.pt")
                if os.path.exists(p):
                    runs[tag] = p
        if args.dev_glob:
            for p in sorted(glob.glob(os.path.join(s3, args.dev_glob + ".pt"))):
                runs[os.path.splitext(os.path.basename(p))[0]] = p

        case_res = {}
        for tag, p in runs.items():
            gen = torch.load(p)
            n = min(gen.shape[0], T)
            g, t = gen[:n], gt[:n]
            s = int(tag.split("_s")[1].split("_")[0])
            idx = np.arange(n)
            mid = idx[idx % s != 0]
            if len(mid) == 0:
                continue
            ps = psnr_frames(g, t).numpy()
            lp = lpips_frames(g, t, net, device).numpy() if net else None

            mo_mid = motion[mid]
            q1, q2 = np.quantile(mo_mid, [1 / 3, 2 / 3])
            buckets = {"slow": mid[mo_mid <= q1],
                       "mid": mid[(mo_mid > q1) & (mo_mid <= q2)],
                       "fast": mid[mo_mid > q2]}
            r = {"psnr_mid_all": float(ps[mid].mean()),
                 "obs_psnr": float(ps[idx[idx % s == 0]].mean()),
                 "motion_cuts": [float(q1), float(q2)]}
            if lp is not None:
                r["lpips_mid_all"] = float(lp[mid].mean())
            for bn, bi in buckets.items():
                r[f"psnr_{bn}"] = float(ps[bi].mean())
                if lp is not None:
                    r[f"lpips_{bn}"] = float(lp[bi].mean())
                r[f"n_{bn}"] = int(len(bi))
            case_res[tag] = r
            lpstr = f" lpips_mid={r.get('lpips_mid_all'):.4f}" if lp is not None else ""
            print(f"case{c} {tag:>26}: psnr_mid={r['psnr_mid_all']:.2f} "
                  f"fast={r['psnr_fast']:.2f}{lpstr}", flush=True)
        results[f"case{c}"] = case_res

    # ---------------------------------------------------------- G1 verdict
    print("\n===== G1 (stride=8, fast bucket, fusion vs rife) =====")
    wins = 0
    judged = 0
    for c in args.cases:
        cr = results.get(f"case{c}", {})
        fu, ri = cr.get("fusion_s8"), cr.get("rife_s8")
        if not fu or not ri:
            print(f"case{c}: missing fusion_s8 or rife_s8 -- not judged")
            continue
        judged += 1
        dp = fu["psnr_fast"] - ri["psnr_fast"]
        win = dp > 0
        line = f"case{c}: dPSNR_fast={dp:+.2f}"
        if "lpips_fast" in fu and "lpips_fast" in ri:
            dl = ri["lpips_fast"] - fu["lpips_fast"]
            win = win and (dl > 0)
            line += f" dLPIPS_fast={dl:+.4f} (positive = fusion better)"
        wins += int(win)
        print(line + ("  WIN" if win else "  LOSS"))
    if judged:
        verdict = "PASS" if wins * 3 >= judged * 2 else "FAIL -> direction dead"
        print(f"G1: {wins}/{judged} wins -> {verdict}")

    jp = os.path.join(FIG_DIR, "step3_bucket_metrics.json")
    json.dump(results, open(jp, "w"), indent=2)
    print(f"\nsaved {jp}")


if __name__ == "__main__":
    main()
