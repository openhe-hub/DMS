"""Step 1.1 -- extract real DWPose keypoint trajectories (runs on the cluster).

Pairs assets/example_data/videos/video{i}.mp4 with images/ref{i}.png, runs
DisPose's DWPose + pose2track, and saves a dense (18,T,2) trajectory + visibility
mask per video to outputs/step1/traj/.
"""
import argparse
import glob
import os

import _paths  # noqa: F401  (sets sys.path + dirs)
from _paths import REPO, TRAJ_DIR
from dispose_siren.trajectory import extract_dense_trajectory, save_npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", default=os.path.join(REPO, "assets/example_data/videos"))
    ap.add_argument("--ref_dir", default=os.path.join(REPO, "assets/example_data/images"))
    ap.add_argument("--stride", type=int, default=1, help="DWPose sample_stride (1 = densest)")
    args = ap.parse_args()

    vids = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
    if not vids:
        raise SystemExit(f"no videos under {args.video_dir}")

    for v in vids:
        name = os.path.splitext(os.path.basename(v))[0]           # video1
        idx = "".join(c for c in name if c.isdigit())
        ref = os.path.join(args.ref_dir, f"ref{idx}.png")
        if not os.path.exists(ref):                               # fallback: any ref
            cands = sorted(glob.glob(os.path.join(args.ref_dir, "*.png")))
            ref = cands[0] if cands else None
        print(f"[extract] {name}  video={v}  ref={ref}", flush=True)
        pts, vis, meta = extract_dense_trajectory(v, ref, sample_stride=args.stride)
        out = os.path.join(TRAJ_DIR, f"{name}.npz")
        save_npz(out, pts, vis, meta)
        vk = vis.all(axis=1).sum()
        print(f"   T={meta['T']}  full-visible-keypoints={vk}/18  -> {out}", flush=True)


if __name__ == "__main__":
    main()
