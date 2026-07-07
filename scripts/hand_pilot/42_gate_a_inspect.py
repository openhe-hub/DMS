"""Gate A inspection: paired off/raw/smooth comparison of the generation runs.

Two stages:
  --stage extract  (cluster, GPU): DWPose over every generated mp4 under
      outputs/hand_pilot/gate_a -> {mp4}.poses.npz (reuses metrics.pose_extract,
      same detector as the benchmark).
  --stage report   (local, after rsyncing gate_a/ and poses/): per case,
      wrist-centred crop contact sheets (arms x fastest-motion frames) +
      paired diagnostics per arm vs the driving source:
        hand_confidence_metrics (structural quality proxy)
        motion_fidelity_metrics (control adherence; gen_offset=0 for DisPose)
      -> gate_a/inspect/{clip}_sheet.png, gate_a/report.md

Per the pre-registration these numbers are DIAGNOSTIC; the Gate A verdict is
the visual readout (consistent visible change in the hand region on >= half
the cases = channel is causally live), and the judgment metrics stay
FVD / tail / human at P2.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

import _paths as P

ARMS = ("off", "raw", "smooth")


def find_runs(arm):
    """{clip_id: mp4_path} for one arm (latest run wins)."""
    out = {}
    for mp4 in sorted(glob.glob(os.path.join(P.GATE_A_DIR, f"*_{arm}",
                                             "*.mp4"))):
        m = re.search(r"_to_([a-z0-9]+)_CFG", os.path.basename(mp4))
        if m:
            out[m.group(1)] = mp4
    return out


def stage_extract():
    os.chdir(P.REPO)
    from metrics.pose_extract import extract_video_poses, save_poses
    mp4s = sorted(glob.glob(os.path.join(P.GATE_A_DIR, "*_*", "*.mp4")))
    print(f"{len(mp4s)} generated videos")
    for i, mp4 in enumerate(mp4s):
        npz = mp4 + ".poses.npz"
        if os.path.exists(npz):
            continue
        save_poses(npz, extract_video_poses(mp4, sample_stride=1))
        print(f"  [{i+1}/{len(mp4s)}] {os.path.basename(mp4)}", flush=True)


def read_frames(mp4, idxs):
    import cv2
    cap = cv2.VideoCapture(mp4)
    frames = {}
    want = set(int(i) for i in idxs)
    t = 0
    while want:
        ok, frm = cap.read()
        if not ok:
            break
        if t in want:
            frames[t] = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
            want.discard(t)
        t += 1
    cap.release()
    return frames


def wrist_px(poses, t, side_idx, W, H):
    from dispose_siren.hand_traj import BODY_WRIST, HAND_ORDER
    j = BODY_WRIST[HAND_ORDER[side_idx]]
    xy = poses["body"][t, j]
    if not np.isfinite(xy).all():
        return None
    return int(xy[0] * W), int(xy[1] * H)


def crop(img, cx, cy, r=64):
    H, W = img.shape[:2]
    x0, y0 = np.clip(cx - r, 0, W - 2 * r), np.clip(cy - r, 0, H - 2 * r)
    return img[y0:y0 + 2 * r, x0:x0 + 2 * r]


def fastest_frames(src_poses, k=6):
    """Frame indices with the highest wrist speed in the source video."""
    b = src_poses["body"]
    sp = np.zeros(len(b))
    for j in (4, 7):
        v = np.linalg.norm(np.diff(b[:, j], axis=0), axis=-1)
        v = np.nan_to_num(v)
        sp[1:] += v
    order = np.argsort(-sp)
    picked = []
    for t in order:
        if all(abs(t - p) > 8 for p in picked):
            picked.append(int(t))
        if len(picked) == k:
            break
    return sorted(picked)


def stage_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, os.path.join(P.SRC, "metrics"))
    from hand_confidence import hand_confidence_metrics
    from motion_fidelity import motion_fidelity_metrics
    from dispose_siren.hand_traj import load_poses

    runs = {arm: find_runs(arm) for arm in ARMS}
    clips = sorted(set.intersection(*(set(r) for r in runs.values())))
    if not clips:
        raise SystemExit("no complete off/raw/smooth triplets under gate_a/")
    out_dir = os.path.join(P.GATE_A_DIR, "inspect")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for clip in clips:
        src = load_poses(os.path.join(P.POSES_DIR, f"{clip}.npz"))
        ts = fastest_frames(src)
        fig, ax = plt.subplots(len(ARMS) + 1, len(ts),
                               figsize=(2 * len(ts), 2 * (len(ARMS) + 1)),
                               squeeze=False)
        src_mp4 = os.path.join(P.REPO, "assets/example_data/sign_videos/"
                               f"hard27k_orig/{clip}.mp4")
        sf = read_frames(src_mp4, ts)
        meta = src["meta"][0]
        for c, t in enumerate(ts):
            w = wrist_px(src, t, 1, meta["W"], meta["H"])
            img = sf.get(t)
            if img is not None and w:
                ax[0][c].imshow(crop(img, *w))
            ax[0][c].set_title(f"src t={t}", fontsize=7)
        row = {"clip": clip}
        for r, arm in enumerate(ARMS, start=1):
            mp4 = runs[arm][clip]
            gp = load_poses(mp4 + ".poses.npz")
            gf = read_frames(mp4, ts)
            for c, t in enumerate(ts):
                if t >= len(gp["detected"]):
                    continue
                img = gf.get(t)
                w = wrist_px(gp, t, 1, img.shape[1] if img is not None else 576,
                             img.shape[0] if img is not None else 576)
                if img is not None and w:
                    ax[r][c].imshow(crop(img, *w))
                if c == 0:
                    ax[r][c].set_ylabel(arm, fontsize=9)
            hc = hand_confidence_metrics(gp)
            mf = motion_fidelity_metrics(gp, src, gen_offset=0)
            row[arm] = dict(mean_hand_conf=hc["mean_hand_conf"],
                            hand_good_rate=hc["hand_good_rate"],
                            hand_nme=mf["hand_nme"], hand_pck=mf["hand_pck"],
                            body_nme=mf["body_nme"])
        for a in ax.ravel():
            a.set_xticks([]), a.set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{clip}_sheet.png"), dpi=130)
        plt.close(fig)
        rows.append(row)
        print(f"  {clip}: sheet + diagnostics done", flush=True)

    md = ["# Gate A diagnostics (paired, per case)", "",
          "| clip | metric | off | raw | smooth |", "|---|---|---|---|---|"]
    for row in rows:
        for k in ("mean_hand_conf", "hand_good_rate", "hand_nme", "body_nme"):
            md.append(f"| {row['clip']} | {k} | "
                      + " | ".join(f"{row[a][k]:.4f}" for a in ARMS) + " |")
    with open(os.path.join(P.GATE_A_DIR, "report.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    json.dump(rows, open(os.path.join(P.GATE_A_DIR, "diagnostics.json"), "w"),
              indent=1, default=float)
    print(f"-> {out_dir}/*_sheet.png + gate_a/report.md\n"
          "Verdict is visual: consistent hand-region change on >= half the "
          "cases = channel live.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("extract", "report"), required=True)
    a = ap.parse_args()
    stage_extract() if a.stage == "extract" else stage_report()
