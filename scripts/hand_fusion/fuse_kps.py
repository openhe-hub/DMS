"""Confidence-gated fusion of DWPose hand keypoints with OmniHands projections.

Per frame and per hand: keep the DWPose detection when the person was found
and the hand's mean confidence clears CONF_THR (repo convention 0.3);
otherwise substitute the OmniHands 3D-recovered projection (score set to
RECON_CONF 0.61, same "model-recovered" convention as the SIREN arm). Whole
hands only -- no per-keypoint mixing, so a hand never blends two models.

Inputs (both normalized [0,1] source coords, hand 0 = left, 1 = right):
  --poses_dir     extract_hand_poses.py output: {clip}.npz with
                  hands[T,2,21,2] (NaN where undetected), hands_score, detected
  --omnihand_dir  omnihand_to_dwpose.py output: {clip}.npz with
                  hands[T,2,21,2], hands_score, covered

Output: --out_dir/{clip}.npz in the hand_recon_dir format
  {hands[T,2,21,2], hands_score[T,2,21], covered[T,2]}
plus fusion_stats.json and optional per-clip overlay mp4s (kept DWPose hands
cyan, substituted OmniHands hands purple).

CPU-only; on the jubail login node run with OMP_NUM_THREADS=1 etc.
"""
import argparse
import json
import os

import numpy as np

CONF_THR = 0.3
RECON_CONF = 0.61
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
              (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
DW_COLOR = (255, 229, 102)   # cyan-ish (BGR): kept DWPose
OH_COLOR = (255, 61, 148)    # purple (BGR): substituted OmniHands


def fuse_clip(poses, oh, conf_thr, recon_conf):
    hands_dw = poses["hands"].astype(np.float64)        # [T,2,21,2], NaN gaps
    score_dw = poses["hands_score"].astype(np.float64)
    detected = poses["detected"].astype(bool)
    hands_oh = oh["hands"].astype(np.float64)
    T = len(hands_dw)
    if len(hands_oh) != T:
        raise SystemExit(f"frame count mismatch: DWPose T={T} vs "
                         f"OmniHands T={len(hands_oh)}")

    mean_conf = np.nanmean(score_dw, axis=2)             # [T,2], NaN gaps
    keep_dw = (detected[:, None]
               & np.isfinite(hands_dw).all(axis=(2, 3))
               & (np.nan_to_num(mean_conf, nan=-1.0) >= conf_thr))  # [T,2]

    fused = np.where(keep_dw[..., None, None], np.nan_to_num(hands_dw),
                     hands_oh).astype(np.float32)
    fused_score = np.where(keep_dw[..., None], np.nan_to_num(score_dw),
                           recon_conf).astype(np.float32)
    covered = np.ones((T, 2), dtype=bool)
    stats = dict(
        T=T,
        detected_rate=float(detected.mean()),
        undetected_frames=int((~detected).sum()),
        replaced_left=int((~keep_dw[:, 0]).sum()),
        replaced_right=int((~keep_dw[:, 1]).sum()),
        replaced_frac=float((~keep_dw).mean()),
        mean_dw_conf=[float(np.nanmean(mean_conf[:, i])) for i in range(2)],
    )
    return dict(hands=fused, hands_score=fused_score, covered=covered), \
        keep_dw, stats


def write_overlay(video_path, out_path, fused, keep_dw):
    import cv2
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    px = fused["hands"] * np.array([W, H], dtype=np.float64)
    for t in range(len(px)):
        ok, frame = cap.read()
        if not ok:
            break
        for i in range(2):
            color = DW_COLOR if keep_dw[t, i] else OH_COLOR
            kps = px[t, i]
            for a, b in HAND_EDGES:
                cv2.line(frame, tuple(np.round(kps[a]).astype(int)),
                         tuple(np.round(kps[b]).astype(int)), color, 2,
                         cv2.LINE_AA)
            for p in kps:
                cv2.circle(frame, tuple(np.round(p).astype(int)), 3,
                           (255, 255, 255), -1, cv2.LINE_AA)
        vw.write(frame)
    vw.release()
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--omnihand_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--videos_dir", default="",
                    help="if set, write overlay mp4s next to the fused npz")
    ap.add_argument("--conf_thr", type=float, default=CONF_THR)
    ap.add_argument("--recon_conf", type=float, default=RECON_CONF)
    ap.add_argument("clips", nargs="*",
                    help="clip ids; default: intersection of both dirs")
    args = ap.parse_args()

    def ids(d):
        return {os.path.splitext(f)[0] for f in os.listdir(d)
                if f.endswith(".npz")}

    clips = args.clips or sorted(ids(args.poses_dir) & ids(args.omnihand_dir))
    if not clips:
        raise SystemExit("no overlapping clips between the two dirs")
    os.makedirs(args.out_dir, exist_ok=True)

    all_stats = {}
    for clip in clips:
        poses = np.load(os.path.join(args.poses_dir, f"{clip}.npz"),
                        allow_pickle=True)
        oh = np.load(os.path.join(args.omnihand_dir, f"{clip}.npz"))
        fused, keep_dw, stats = fuse_clip(poses, oh, args.conf_thr,
                                          args.recon_conf)
        np.savez_compressed(os.path.join(args.out_dir, f"{clip}.npz"), **fused)
        meta = poses["meta"][0] if "meta" in poses else {}
        if meta:
            stats["fps"] = float(meta["fps"])
            if int(stats["fps"] / 24) > 1:
                print(f"WARNING {clip}: fps={stats['fps']} -> DWPose stride "
                      f">1 at generation time, per-frame npz will misalign")
        all_stats[clip] = stats
        print(f"{clip}: T={stats['T']} det={stats['detected_rate']:.2f} "
              f"replaced L={stats['replaced_left']} R={stats['replaced_right']}"
              f" ({100 * stats['replaced_frac']:.1f}%) "
              f"dw_conf={np.round(stats['mean_dw_conf'], 3)}")
        if args.videos_dir:
            vp = os.path.join(args.videos_dir, f"{clip}.mp4")
            if os.path.exists(vp):
                write_overlay(vp, os.path.join(args.out_dir,
                                               f"fused_{clip}.mp4"),
                              fused, keep_dw)

    with open(os.path.join(args.out_dir, "fusion_stats.json"), "w") as f:
        json.dump(dict(conf_thr=args.conf_thr, recon_conf=args.recon_conf,
                       clips=all_stats), f, indent=1)
    print(f"wrote {len(clips)} fused clips -> {args.out_dir}")


if __name__ == "__main__":
    main()
