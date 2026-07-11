"""Gate B: noise profile of hand keypoints on the sign-language source clips.

Answers, with measurements instead of assumptions: are hand trajectories as
clean as body (known ~2-3 px jitter), or do they carry the outlier + dropout
structure a learned prior could exploit?

Per (clip, side): confidence profile (wrist / MCPs / fingertips vs body),
jitter (smoothing-residual + second-difference, in px AND relative to hand
scale, computed only on runs of >=8 consecutive confident frames), and
dropout/gap statistics. Pooled gap-length histogram feeds the inpainting
protocol; per-clip badness ranking feeds Gate A case selection.

Kill readout (pre-registered): hand jitter <= ~1.2x body jitter (px) AND
dropout < 2%  =>  no denoising edge, spline suffices.

CPU-only; run locally after rsyncing outputs/hand_pilot/poses from jubail.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

import _paths as P
from dispose_siren.baselines import _smooth
from dispose_siren.hand_traj import (load_poses, to_px, HAND_ORDER,
                                     BODY_WRIST, BODY_ELBOW, BODY_SHOULDER,
                                     WRIST, MID_MCP, CONF_THR)

MCPS = [1, 5, 9, 13, 17]
TIPS = [4, 8, 12, 16, 20]
MIN_RUN = 8
SMOOTH_SIG = 1.5


def runs_of(mask, min_len=MIN_RUN):
    """[(start, stop)] of consecutive-True runs of length >= min_len."""
    out, s = [], None
    for t, m in enumerate(mask):
        if m and s is None:
            s = t
        elif not m and s is not None:
            if t - s >= min_len:
                out.append((s, t))
            s = None
    if s is not None and len(mask) - s >= min_len:
        out.append((s, len(mask)))
    return out


def track_jitter(track_px):
    """(n,2) px track -> (residual std after smoothing, 2nd-diff median)."""
    sm = _smooth(track_px[None], SMOOTH_SIG)[0]
    resid = float((track_px - sm).std())
    if len(track_px) >= 3:
        d2 = track_px[:-2] - 2 * track_px[1:-1] + track_px[2:]
        second = float(np.median(np.abs(d2))) / 2.0
    else:
        second = float("nan")
    return resid, second


def hand_stats(poses, i):
    """Jitter/gap/conf stats for hand index i of one clip."""
    det = poses["detected"].astype(bool)
    hands, hs = poses["hands"], poses["hands_score"]
    meta = poses["meta"][0]
    W, H = meta["W"], meta["H"]
    mean_conf = np.nanmean(hs[:, i], axis=1)
    present = det & (mean_conf >= CONF_THR)

    st = {"T": int(len(det)), "present_rate": float(present[det].mean())
          if det.any() else 0.0}
    st["dropout_rate"] = 1.0 - st["present_rate"]

    dc = hs[det, i]
    st["conf_wrist"] = float(np.nanmean(dc[:, WRIST])) if det.any() else np.nan
    st["conf_mcp"] = float(np.nanmean(dc[:, MCPS])) if det.any() else np.nan
    st["conf_tips"] = float(np.nanmean(dc[:, TIPS])) if det.any() else np.nan

    # internal gaps between present frames
    idx = np.where(present)[0]
    gaps = []
    if len(idx) >= 2:
        holes = np.diff(idx) - 1
        gaps = [int(g) for g in holes[holes > 0]]
    st["n_gaps"], st["max_gap"] = len(gaps), (max(gaps) if gaps else 0)
    st["gap_lengths"] = gaps

    resids, seconds, rels = [], [], []
    for s, e in runs_of(present):
        seg = hands[s:e, i]
        seg_px = to_px(seg, W, H)
        bone = np.linalg.norm(seg_px[:, WRIST] - seg_px[:, MID_MCP], axis=-1)
        hand_scale = float(np.median(bone))
        for k in range(hands.shape[2]):
            if hs[s:e, i, k].min() < CONF_THR:
                continue
            r, d2 = track_jitter(seg_px[:, k])
            resids.append(r)
            seconds.append(d2)
            if hand_scale > 1e-3:
                rels.append(r / hand_scale)
    st["jitter_px"] = float(np.median(resids)) if resids else np.nan
    st["jitter_2diff_px"] = float(np.median(seconds)) if seconds else np.nan
    st["jitter_rel"] = float(np.median(rels)) if rels else np.nan
    st["n_jitter_tracks"] = len(resids)
    return st


def body_stats(poses, side):
    """Same jitter measures for body wrist/elbow/shoulder of one side."""
    det = poses["detected"].astype(bool)
    body, bs = poses["body"], poses["body_score"]
    meta = poses["meta"][0]
    W, H = meta["W"], meta["H"]
    out = {}
    for name, ji in (("wrist", BODY_WRIST[side]), ("elbow", BODY_ELBOW[side]),
                     ("shoulder", BODY_SHOULDER[side])):
        ok = det & (bs[:, ji] >= CONF_THR)
        resids = []
        for s, e in runs_of(ok):
            r, _ = track_jitter(to_px(body[s:e, ji], W, H))
            resids.append(r)
        out[f"body_{name}_jitter_px"] = (float(np.median(resids))
                                         if resids else np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses_dir", default=P.POSES_DIR)
    ap.add_argument("--out_dir", default=P.GATE_B_DIR)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.poses_dir, "*.npz")))
    if not files:
        raise SystemExit(f"no poses under {args.poses_dir}; run 30_ first")
    print(f"Gate B over {len(files)} clips", flush=True)

    rows, all_gaps = [], []
    for fp in files:
        clip = os.path.splitext(os.path.basename(fp))[0]
        poses = load_poses(fp)
        for i, side in enumerate(HAND_ORDER):
            st = hand_stats(poses, i)
            st.update(body_stats(poses, side))
            all_gaps.extend(st.pop("gap_lengths"))
            st.update(clip=clip, side=side)
            rows.append(st)

    cols = ["clip", "side", "T", "present_rate", "dropout_rate", "n_gaps",
            "max_gap", "jitter_px", "jitter_2diff_px", "jitter_rel",
            "n_jitter_tracks", "conf_wrist", "conf_mcp", "conf_tips",
            "body_wrist_jitter_px", "body_elbow_jitter_px",
            "body_shoulder_jitter_px"]
    with open(os.path.join(args.out_dir, "per_clip.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows({k: r.get(k) for k in cols} for r in rows)

    def med(key):
        v = np.array([r[key] for r in rows], dtype=float)
        return float(np.nanmedian(v))

    hand_j, body_j = med("jitter_px"), med("body_wrist_jitter_px")
    dropout = med("dropout_rate")
    ratio = hand_j / body_j if body_j > 0 else float("inf")
    kill = (ratio <= 1.2) and (dropout < 0.02)
    summary = dict(
        n_clips=len(files), n_hand_rows=len(rows),
        hand_jitter_px=hand_j, hand_jitter_rel=med("jitter_rel"),
        hand_jitter_2diff_px=med("jitter_2diff_px"),
        body_wrist_jitter_px=body_j,
        body_elbow_jitter_px=med("body_elbow_jitter_px"),
        body_shoulder_jitter_px=med("body_shoulder_jitter_px"),
        jitter_ratio_hand_over_body=ratio,
        dropout_rate=dropout,
        conf=dict(wrist=med("conf_wrist"), mcp=med("conf_mcp"),
                  tips=med("conf_tips")),
        gaps=dict(total=len(all_gaps),
                  median=float(np.median(all_gaps)) if all_gaps else 0.0,
                  p90=float(np.percentile(all_gaps, 90)) if all_gaps else 0.0,
                  max=max(all_gaps) if all_gaps else 0),
        kill_denoising_edge=bool(kill),
    )
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    hist = {}
    for g in all_gaps:
        hist[g] = hist.get(g, 0) + 1
    with open(os.path.join(args.out_dir, "gap_lengths.json"), "w") as f:
        json.dump(dict(histogram={str(k): hist[k] for k in sorted(hist)},
                       lengths=sorted(all_gaps)), f, indent=1)

    # clip-level badness = worst side (dropout + relative jitter)
    by_clip = {}
    for r in rows:
        bad = r["dropout_rate"] + (0.0 if np.isnan(r["jitter_rel"])
                                   else r["jitter_rel"])
        by_clip[r["clip"]] = max(by_clip.get(r["clip"], 0.0), bad)
    ranking = sorted(by_clip.items(), key=lambda kv: -kv[1])
    with open(os.path.join(args.out_dir, "case_ranking.csv"), "w",
              newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["clip", "badness"])
        wr.writerows(ranking)

    make_fig(rows, all_gaps, os.path.join(args.out_dir, "fig_gate_b.png"))

    print(json.dumps(summary, indent=1))
    print("\n===== GATE B READOUT =====")
    print(f"  hand jitter {hand_j:.2f}px vs body wrist {body_j:.2f}px "
          f"(ratio {ratio:.2f}); dropout {dropout:.1%}")
    if kill:
        print("  KILL: hands are as clean as body -> no denoising/inpainting "
              "edge; spline suffices (raw-hand channel claim unaffected).")
    else:
        print("  PASS: hands are measurably noisier/gappier than body -> "
              "learned-prior denoising/inpainting has real headroom.")
    print(f"  worst clips for Gate A: {[c for c, _ in ranking[:3]]}")


def make_fig(rows, all_gaps, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    a = ax[0, 0]
    confs = [[r["conf_wrist"] for r in rows], [r["conf_mcp"] for r in rows],
             [r["conf_tips"] for r in rows]]
    a.boxplot([np.array(c)[~np.isnan(c)] for c in map(np.asarray, confs)],
              tick_labels=["wrist", "MCPs", "fingertips"])
    a.axhline(CONF_THR, ls="--", c="r", lw=1)
    a.set_title("hand keypoint confidence")

    a = ax[0, 1]
    pairs = [("jitter_px", "hand"), ("body_wrist_jitter_px", "body wrist"),
             ("body_elbow_jitter_px", "body elbow")]
    data = [np.array([r[k] for r in rows], float) for k, _ in pairs]
    a.boxplot([d[~np.isnan(d)] for d in data],
              tick_labels=[n for _, n in pairs])
    a.set_ylabel("px")
    a.set_title("jitter: smoothing residual std")

    a = ax[1, 0]
    if all_gaps:
        a.hist(all_gaps, bins=np.arange(0.5, max(all_gaps) + 1.5), log=True)
    a.set_xlabel("gap length (frames)")
    a.set_title(f"hand dropout gaps (n={len(all_gaps)})")

    a = ax[1, 1]
    x = [r["conf_tips"] for r in rows]
    y = [r["jitter_rel"] for r in rows]
    a.scatter(x, y, s=12, alpha=0.6)
    a.set_xlabel("fingertip conf")
    a.set_ylabel("relative jitter")
    a.set_title("per clip-side: jitter vs confidence")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"fig -> {path}")


if __name__ == "__main__":
    main()
