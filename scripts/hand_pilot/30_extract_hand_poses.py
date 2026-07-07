"""Extract DWPose body+hand trajectories for the hard27k source clips.

Runs on GPU (jubail); dwpose_detector resolves weight paths relative to the
repo root, so this script chdirs there. Outputs one npz per clip plus a
manifest, and performs the V1 hand-order verification on the first clips.

Usage (cluster):
  python scripts/hand_pilot/30_extract_hand_poses.py \
      --video_dir assets/example_data/sign_videos/hard27k_orig
Smoke: add --limit 2.
Optional cross-check against the metrics pose_cache: --check_cache <dir>.
"""
import argparse
import glob
import hashlib
import json
import os

import numpy as np

import _paths as P


def cache_key(video_path, stride):
    """Reproduce metrics.run_eval's cache key: md5(abspath|s{stride})[:16]."""
    key = hashlib.md5(f"{os.path.abspath(video_path)}|s{stride}"
                      .encode()).hexdigest()[:16]
    return key + ".npz"


def cross_check(poses, cache_dir, video_path, stride):
    from dispose_siren.hand_traj import load_poses
    cpath = os.path.join(cache_dir, cache_key(video_path, stride))
    if not os.path.exists(cpath):
        return None
    cached = load_poses(cpath)
    n = min(len(poses["detected"]), len(cached["detected"]))
    both = poses["detected"][:n] & cached["detected"][:n]
    if not both.any():
        return {"cache": cpath, "overlap_frames": 0}
    d = np.abs(poses["hands"][:n][both] - cached["hands"][:n][both])
    return {"cache": os.path.basename(cpath), "overlap_frames": int(both.sum()),
            "hands_maxdiff": float(np.nanmax(d))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir",
                    default="assets/example_data/sign_videos/hard27k_orig")
    ap.add_argument("--out_dir", default=P.POSES_DIR)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N clips")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--check_cache", default="", help="metrics pose_cache dir")
    args = ap.parse_args()

    os.chdir(P.REPO)  # dwpose weight paths are repo-relative
    from dispose_siren.hand_traj import (extract_hand_poses, save_poses,
                                         load_poses, verify_hand_order,
                                         HAND_ORDER)

    videos = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
    if args.limit:
        videos = videos[:args.limit]
    if not videos:
        raise SystemExit(f"no mp4 under {args.video_dir}")
    print(f"{len(videos)} clips from {args.video_dir}", flush=True)

    manifest, order_checks = [], []
    for vi, vp in enumerate(videos):
        clip_id = os.path.splitext(os.path.basename(vp))[0]
        npz = os.path.join(args.out_dir, f"{clip_id}.npz")
        if args.skip_existing and os.path.exists(npz):
            poses = load_poses(npz)
        else:
            poses = extract_hand_poses(vp, sample_stride=args.stride)
            save_poses(npz, poses)
        det = poses["detected"].astype(bool)
        mean_hconf = np.nanmean(poses["hands_score"][det], axis=2).mean(axis=0) \
            if det.any() else np.array([np.nan, np.nan])
        meta = poses["meta"][0]
        entry = dict(clip_id=clip_id, T=meta["T"], fps=meta["fps"],
                     W=meta["W"], H=meta["H"],
                     detected_rate=float(det.mean()),
                     mean_hand_conf={HAND_ORDER[i]: float(mean_hconf[i])
                                     for i in range(2)})
        if args.check_cache:
            entry["cache_check"] = cross_check(poses, args.check_cache, vp,
                                               args.stride)
        manifest.append(entry)
        if len(order_checks) < 3 and det.sum() >= 20:
            chk = verify_hand_order(poses)
            chk["clip_id"] = clip_id
            order_checks.append(chk)
        print(f"  [{vi+1}/{len(videos)}] {clip_id}: T={meta['T']} "
              f"det={det.mean():.2f} hconf={np.round(mean_hconf, 3)}", flush=True)

    print("\n===== V1 hand-order verification =====")
    for chk in order_checks:
        print(f"  {chk['clip_id']}: implied={chk['implied_order']} "
              f"provisional={HAND_ORDER} match={chk['matches_provisional']} "
              f"(n={chk['n_frames']}, med={chk['median_dist']})")
    agreed = {c["implied_order"] for c in order_checks}
    if len(agreed) == 1 and order_checks:
        verdict = ("CONFIRMED " + str(next(iter(agreed)))
                   if next(iter(agreed)) == HAND_ORDER
                   else f"MISMATCH: data says {next(iter(agreed))}, "
                        f"code says {HAND_ORDER} -- FIX hand_traj.HAND_ORDER")
    else:
        verdict = f"INCONCLUSIVE across clips: {sorted(agreed)}"
    print(f"  V1 verdict: {verdict}")

    out = dict(video_dir=args.video_dir, stride=args.stride,
               hand_order_provisional=list(HAND_ORDER),
               v1_verdict=verdict, v1_checks=order_checks, clips=manifest)
    mpath = os.path.join(args.out_dir, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"\nwrote {len(manifest)} clips -> {mpath}")


if __name__ == "__main__":
    main()
