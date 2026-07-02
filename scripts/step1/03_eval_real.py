"""Step 1.3 -- evaluate the learned INR vs fd / fd+Gaussian on REAL DWPose
trajectories under both honest protocols, aggregate over all videos, write a
JSON summary + figures.
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
from dispose_siren.eval_protocols import (
    make_windows, protocol_holdout, protocol_pseudogt, estimate_jitter)


def aggregate(per_video, keys):
    """Sample-weighted mean across videos for each key in keys."""
    tot = {k: 0.0 for k in keys}; n = 0
    for r in per_video:
        w = r["S"]
        for k in keys:
            tot[k] += r[k] * w
        n += w
    return {k: tot[k] / n for k in keys}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--span", type=int, default=48)
    ap.add_argument("--step", type=int, default=24)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = os.path.join(CKPT_DIR, "amortized.pt")
    model, meta = load_ckpt(ckpt, device)
    print(f"loaded {ckpt}  meta={meta}", flush=True)

    trajs = sorted(glob.glob(os.path.join(TRAJ_DIR, "*.npz")))
    if not trajs:
        raise SystemExit(f"no trajectories under {TRAJ_DIR} -- run 01 first")

    A_rows, B_rows, summary = [], [], {}
    for t in trajs:
        name = os.path.splitext(os.path.basename(t))[0]
        pts, vis, m = load_npz(t)
        wins = make_windows(pts, vis, span=args.span, step=args.step)
        if len(wins) == 0:
            print(f"[{name}] no fully-visible windows (T={m['T']}) -- skipped", flush=True)
            continue
        rA = protocol_holdout(wins, model, device)
        rB = protocol_pseudogt(wins, model, device)
        jit = estimate_jitter(wins)
        A_rows.append(rA); B_rows.append(rB)
        summary[name] = {"S": int(rA["S"]), "jitter": jit, "holdout": rA, "pseudogt": rB}
        print(f"[{name}] windows={rA['S']}  jitter abs={jit['abs_px']:.2f}px "
              f"relative={jit['relative']:.3f}", flush=True)
        print(f"   A held-out pos-MSE(px^2): linint={rA['linint']:.2f} "
              f"gauss+lin={rA['gauss+lin']:.2f}(σ={rA['gauss_sigma']}) learned={rA['learned']:.2f}", flush=True)
        print(f"   B pseudoGT vel-MSE:       fd={rB['fd']:.1f} "
              f"fd+gauss={rB['fd+gauss']:.1f}(σ={rB['fdg_sigma']}) learned={rB['learned']:.1f}", flush=True)

    aA, nA = aggregate(A_rows, ["linint", "gauss+lin", "learned"])
    aB, nB = aggregate(B_rows, ["fd", "fd+gauss", "learned"])

    def verdict(d, base_keys, lk="learned"):
        best_base = min(d[k] for k in base_keys)
        return ("LEARNED" if d[lk] < best_base else "baseline"), best_base / d[lk]

    vA, rA_ratio = verdict(aA, ["linint", "gauss+lin"])
    vB, rB_ratio = verdict(aB, ["fd", "fd+gauss"])

    print("\n" + "=" * 72)
    print(f"AGGREGATE over {len(A_rows)} videos, {nA} held-out windows")
    print(f"  Protocol A (held-out pos-MSE px^2): linint={aA['linint']:.2f}  "
          f"gauss+lin={aA['gauss+lin']:.2f}  learned={aA['learned']:.2f}  "
          f"-> {vA} (best_base/learned={rA_ratio:.2f}x)")
    print(f"  Protocol B (pseudoGT vel-MSE):      fd={aB['fd']:.1f}  "
          f"fd+gauss={aB['fd+gauss']:.1f}  learned={aB['learned']:.1f}  "
          f"-> {vB} (best_base/learned={rB_ratio:.2f}x)")
    print("  NOTE: Protocol B's pseudo-GT is fd of noisy detections -> favours fd-like")
    print("        methods; Protocol A is the neutral verdict.")
    print("=" * 72, flush=True)

    summary["_aggregate"] = {
        "n_videos": len(A_rows), "n_windows": int(nA),
        "holdout": aA, "holdout_verdict": vA, "holdout_ratio": rA_ratio,
        "pseudogt": aB, "pseudogt_verdict": vB, "pseudogt_ratio": rB_ratio,
        "ckpt_meta": meta,
    }
    jpath = os.path.join(FIG_DIR, "step1_summary.json")
    with open(jpath, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"wrote {jpath}", flush=True)

    # ---- figure: grouped bars for both protocols ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable ({e}); skipping figure", flush=True)
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].bar(["linint", "gauss+lin", "learned"],
              [aA["linint"], aA["gauss+lin"], aA["learned"]],
              color=["tab:orange", "tab:green", "tab:blue"])
    ax[0].set_title(f"A: held-out pos-MSE (px²)  [{vA}]"); ax[0].set_ylabel("MSE")
    ax[1].bar(["fd", "fd+gauss", "learned"],
              [aB["fd"], aB["fd+gauss"], aB["learned"]],
              color=["tab:orange", "tab:green", "tab:blue"])
    ax[1].set_title(f"B: pseudo-GT vel-MSE  [{vB}]"); ax[1].set_ylabel("MSE")
    fig.suptitle(f"Step 1: learned INR vs baselines on REAL DWPose ({nA} windows)")
    fig.tight_layout()
    fpath = os.path.join(FIG_DIR, "step1_real_bars.png")
    fig.savefig(fpath, dpi=120); plt.close(fig)
    print(f"wrote {fpath}", flush=True)


if __name__ == "__main__":
    main()
