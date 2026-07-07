"""Reconstruct dense hand trajectories with the trained HandSetSIREN.

For each clip/side: sliding windows (span 32, stride 16, tail-aligned) over
the extracted DWPose tracks; windows passing the same gates as training are
encoded (all frames as observations -- the conf channel tells the model which
are junk) and decoded densely; overlaps are blended with triangular weights;
frames not covered by any valid window keep the raw detection.

Output per clip: outputs/hand_pilot/hands_recon/{clip}.npz with
  hands       [T,2,21,2]  reconstructed (raw where uncovered), source coords
  hands_score [T,2,21]    original conf, floored to 0.61 on covered frames so
                          previously-invisible gap frames become usable by
                          the hand_flow channel (that is the point of
                          inpainting)
  covered     [T,2] bool

These feed generation via the `hand_recon_dir` yaml field (the SIREN arm of
the three-system comparison).
"""
import argparse
import glob
import os

import numpy as np

import _paths as P
from dispose_siren.hand_traj import (load_poses, hand_canon, hand_uncanon,
                                     HAND_ORDER, BODY_WRIST, BODY_ELBOW,
                                     WRIST, MID_MCP, CONF_THR)

SPAN, STRIDE = 32, 16
RECON_CONF = 0.61   # just above the 0.3/0.45 gates; marks "SIREN-filled"


def window_starts(T, span=SPAN, stride=STRIDE):
    if T < span:
        return []
    s = list(range(0, T - span + 1, stride))
    if s[-1] != T - span:
        s.append(T - span)
    return s


def valid_window(det, hs, bs, wi, sl, conf_thr=CONF_THR, min_good_frac=0.8):
    if not det[sl].all():
        return False
    mean_conf = np.nanmean(hs[sl], axis=1)
    if (mean_conf >= conf_thr).mean() < min_good_frac:
        return False
    if (bs[sl, wi] >= conf_thr).mean() < min_good_frac:
        return False
    return True


def reconstruct_clip(poses, model, device, min_bone_px=8.0, max_canon_amp=12.0):
    import torch
    det = poses["detected"].astype(bool)
    hands, hs = poses["hands"].astype(np.float64), poses["hands_score"].astype(np.float64)
    body, bs = poses["body"].astype(np.float64), poses["body_score"].astype(np.float64)
    meta = poses["meta"][0]
    T = len(det)
    Wpx, Hpx = meta["W"], meta["H"]
    out = np.where(np.isfinite(hands), hands, 0.0).copy()
    out_s = np.where(np.isfinite(hs), hs, 0.0).copy()
    covered = np.zeros((T, 2), bool)
    tri = np.minimum(np.arange(SPAN) + 1, SPAN - np.arange(SPAN))  # blend w

    for i, side in enumerate(HAND_ORDER):
        wi, ei = BODY_WRIST[side], BODY_ELBOW[side]
        acc = np.zeros((T, 21, 2))
        den = np.zeros(T)
        for s0 in window_starts(T):
            sl = slice(s0, s0 + SPAN)
            if not valid_window(det, hs[:, i], bs, wi, sl):
                continue
            traj = hands[sl, i]
            if not np.isfinite(traj).all():
                continue
            bone = np.linalg.norm((traj[:, WRIST] - traj[:, MID_MCP])
                                  * np.array([Wpx, Hpx]), axis=-1)
            gd = (hs[sl, i, WRIST] >= CONF_THR) & (hs[sl, i, MID_MCP] >= CONF_THR)
            med_bone = np.median(bone[gd]) if gd.any() else np.median(bone)
            if med_bone < min_bone_px:
                continue
            conf = hs[sl, i]
            tn, wr_n, el_n, mu, sc = hand_canon(
                traj[None], conf[None], body[sl, wi][None], body[sl, ei][None])
            if np.abs(tn).max() > max_canon_amp:
                continue
            tt = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
            tau = torch.linspace(0, 1, SPAN, device=device)
            with torch.no_grad():
                mod = model.encode(
                    tt(tn), tt(conf[None]), tt(wr_n), tt(el_n),
                    tt(np.log(sc[:, 0, 0, 0])),
                    tt(np.array([i * 2.0 - 1.0])), tau[None])
                pred = model.decode(mod, tau).cpu().numpy()
            pred = hand_uncanon(pred.reshape(1, SPAN, 21, 2), mu, sc)[0]
            acc[sl] += tri[:, None, None] * pred
            den[sl] += tri
        cov = den > 0
        if cov.any():
            out[cov, i] = acc[cov] / den[cov, None, None]
            out_s[cov, i] = np.maximum(out_s[cov, i], RECON_CONF)
            covered[cov, i] = True
    return out.astype(np.float32), out_s.astype(np.float32), covered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses_dir", default=P.POSES_DIR)
    ap.add_argument("--ckpt",
                    default=os.path.join(P.CKPT_DIR, "p0_crush.pt"))
    ap.add_argument("--out_dir",
                    default=os.path.join(P.OUT, "hands_recon"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from dispose_siren.hand_train import load_ckpt
    dev = args.device or ("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    model, extra = load_ckpt(args.ckpt, device=dev)
    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.poses_dir, "*.npz")))
    if args.limit:
        files = files[:args.limit]
    print(f"reconstructing {len(files)} clips with {args.ckpt} "
          f"({extra.get('mode', '?')}, S={extra.get('S')}) on {dev}")

    tot_cov = tot = 0
    for fp in files:
        clip = os.path.splitext(os.path.basename(fp))[0]
        poses = load_poses(fp)
        h, s, cov = reconstruct_clip(poses, model, dev)
        np.savez_compressed(os.path.join(args.out_dir, f"{clip}.npz"),
                            hands=h, hands_score=s, covered=cov)
        tot_cov += cov.sum()
        tot += cov.size
        print(f"  {clip}: covered {cov.mean():.0%}", flush=True)
    print(f"done: {tot_cov}/{tot} hand-frames covered "
          f"({tot_cov/max(tot,1):.0%}) -> {args.out_dir}")


if __name__ == "__main__":
    main()
