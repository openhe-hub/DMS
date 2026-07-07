"""Hand-keypoint trajectory extraction + windowing for the hand-channel pilot.

Extraction wraps `metrics.pose_extract` (same DWPose estimator as inference and
the 109-case benchmark) and stores per-clip npz with hands + confidences.
Windowing keeps low-confidence frames INSIDE a window (they are the inpainting
substrate); validity gates only on person detection and on having enough
trustworthy frames overall.

Canonical frame: window-level, matching `metrics.motion_fidelity._norm_hand`
semantics (wrist-centred, wrist->mid-MCP scale) but robust across the window:
one confidence-weighted mean wrist origin and one median bone-length scale per
window, so per-frame detection noise does not leak into the normalization.

Import note: extraction needs the DWPose onnx chain (weights + repo-root cwd),
so `metrics.pose_extract` is imported lazily inside `extract_hand_poses`;
windowing/canon are importable with numpy only.
"""
import os

import numpy as np

HAND_N = 21
WRIST, MID_MCP = 0, 9                 # same as metrics.motion_fidelity
CONF_THR = 0.3                        # repo-wide keypoint confidence convention

# DWPose wholebody slices at dwpose_detector.py:58-59 stack candidate[:,92:113]
# first, then candidate[:,113:]. COCO-WholeBody assigns 92-112 = LEFT hand,
# 113-133 = RIGHT hand, so hands[0] should be LEFT -- which contradicts the
# comment in metrics/pose_extract.py ("hand 0 = right"). Provisional order
# below; `verify_hand_order` measures it against DWPose's own body wrists
# (V1 gate) and must be run before any per-side conclusion.
HAND_ORDER = ("left", "right")

# OpenPose-18 body indices per side
BODY_WRIST = {"right": 4, "left": 7}
BODY_ELBOW = {"right": 3, "left": 6}
BODY_SHOULDER = {"right": 2, "left": 5}


# ------------------------------------------------------------------ extraction
def extract_hand_poses(video_path, sample_stride=1, max_frames=None):
    """DWPose over one clip -> pose dict (body/hands + scores) plus meta.

    Must run from the repo root (dwpose_detector uses relative weight paths).
    """
    import decord
    from metrics.pose_extract import extract_video_poses

    poses = extract_video_poses(video_path, sample_stride=sample_stride,
                                max_frames=max_frames)
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    h, w = vr[0].shape[:2]
    poses["meta"] = np.array([dict(
        video=os.path.basename(video_path), fps=float(vr.get_avg_fps()),
        T=int(len(poses["detected"])), stride=int(sample_stride),
        W=int(w), H=int(h))], dtype=object)
    return poses


def save_poses(path, poses):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **poses)


def load_poses(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def verify_hand_order(poses, conf_thr=CONF_THR):
    """V1: check hands[i] against DWPose's own body wrists.

    For every frame where both body wrists and both hand-wrist kps are
    confident, measure ||hands[i][WRIST] - body[LWrist/RWrist]||. Returns the
    median distance matrix and the implied order.
    """
    det = poses["detected"].astype(bool)
    body, bs = poses["body"], poses["body_score"]
    hands, hs = poses["hands"], poses["hands_score"]
    dists = {i: {"left": [], "right": []} for i in range(2)}
    for t in np.where(det)[0]:
        if min(bs[t, BODY_WRIST["left"]], bs[t, BODY_WRIST["right"]]) < conf_thr:
            continue
        for i in range(2):
            if hs[t, i, WRIST] < conf_thr:
                continue
            for side in ("left", "right"):
                d = np.linalg.norm(hands[t, i, WRIST] - body[t, BODY_WRIST[side]])
                dists[i][side].append(d)
    med = {i: {s: (float(np.median(v)) if v else float("nan"))
               for s, v in dists[i].items()} for i in range(2)}
    order = tuple("left" if med[i]["left"] < med[i]["right"] else "right"
                  for i in range(2))
    n = min(len(dists[0]["left"]), len(dists[1]["left"]))
    return {"median_dist": med, "implied_order": order, "n_frames": int(n),
            "matches_provisional": order == HAND_ORDER}


# ------------------------------------------------------------------ windowing
def make_hand_windows(poses, clip_id, span=32, step=8, conf_thr=CONF_THR,
                      min_good_frac=0.8, min_bone_px=8.0):
    """Slide windows over one clip; one record per (side, start).

    A window is kept iff every frame has a detected person, >= min_good_frac of
    frames have per-hand mean conf >= conf_thr, and the same fraction of frames
    have a confident body wrist for that side. Low-conf frames inside a kept
    window are retained (handled downstream by confidence weights).

    min_bone_px: reject windows whose median wrist->mid-MCP length is below
    this (pixels). DWPose emits collapsed ~2px "hands" at moderate confidence;
    dividing by such a scale explodes the canonical frame (velocities 1e3+,
    the P0 blow-up), and a hand that small carries no articulation signal.

    Returns a dict of stacked arrays (possibly empty):
      traj (S,span,21,2)  conf (S,span,21)  wrist/elbow (S,span,2)
      wrist_conf/elbow_conf (S,span)  side (S,) int {0,1}  t0 (S,)
      clip (S,) str  fps/W/H (S,)
    """
    det = poses["detected"].astype(bool)
    body, bs = poses["body"], poses["body_score"]
    hands, hs = poses["hands"], poses["hands_score"]
    meta = poses["meta"][0] if "meta" in poses else {}
    T = len(det)
    rec = {k: [] for k in ("traj", "conf", "wrist", "elbow", "wrist_conf",
                           "elbow_conf", "side", "t0", "clip", "fps", "W", "H")}
    for i, side in enumerate(HAND_ORDER):
        wi, ei = BODY_WRIST[side], BODY_ELBOW[side]
        mean_conf = np.nanmean(hs[:, i], axis=1)                      # (T,)
        for s0 in range(0, T - span + 1, step):
            sl = slice(s0, s0 + span)
            if not det[sl].all():
                continue
            if (mean_conf[sl] >= conf_thr).mean() < min_good_frac:
                continue
            if (bs[sl, wi] >= conf_thr).mean() < min_good_frac:
                continue
            traj = hands[sl, i]
            if not np.isfinite(traj).all():
                continue
            bone = np.linalg.norm(
                (hands[sl, i, WRIST] - hands[sl, i, MID_MCP])
                * np.array([meta.get("W", 640), meta.get("H", 360)]), axis=-1)
            good_b = ((hs[sl, i, WRIST] >= conf_thr)
                      & (hs[sl, i, MID_MCP] >= conf_thr))
            med_bone = np.median(bone[good_b]) if good_b.any() else np.median(bone)
            if med_bone < min_bone_px:
                continue
            rec["traj"].append(traj)
            rec["conf"].append(hs[sl, i])
            rec["wrist"].append(body[sl, wi])
            rec["elbow"].append(body[sl, ei])
            rec["wrist_conf"].append(bs[sl, wi])
            rec["elbow_conf"].append(bs[sl, ei])
            rec["side"].append(i)
            rec["t0"].append(s0)
            rec["clip"].append(clip_id)
            rec["fps"].append(meta.get("fps", 0.0))
            rec["W"].append(meta.get("W", 0))
            rec["H"].append(meta.get("H", 0))
    if not rec["traj"]:
        return None
    out = {}
    for k, v in rec.items():
        out[k] = (np.array(v) if k == "clip"
                  else np.asarray(v, dtype=np.float64 if k in
                                  ("traj", "conf", "wrist", "elbow",
                                   "wrist_conf", "elbow_conf", "fps")
                                  else np.int64))
    return out


def concat_windows(parts):
    """Merge the per-clip dicts from make_hand_windows into one."""
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}


# ---------------------------------------------------------------- canonical
def hand_canon(traj, conf, wrist=None, elbow=None, conf_thr=CONF_THR):
    """Window-level canonicalization of a batch of hand windows.

    traj (S,span,21,2), conf (S,span,21); optional wrist/elbow (S,span,2) get
    the same transform. Origin = conf-weighted mean of the hand wrist kp over
    the window; scale = median wrist->mid-MCP length over confident frames
    (fallback: all frames). Returns (traj_n, wrist_n, elbow_n, mu, scale) with
    mu (S,1,1,2) and scale (S,1,1,1); invert with hand_uncanon.
    """
    traj = np.asarray(traj, dtype=np.float64)
    conf = np.asarray(conf, dtype=np.float64)
    S = len(traj)
    w = np.clip(conf[:, :, WRIST], 0.05, None)                       # (S,span)
    mu = (traj[:, :, WRIST] * w[..., None]).sum(1) / w.sum(1)[:, None]  # (S,2)
    bone = np.linalg.norm(traj[:, :, WRIST] - traj[:, :, MID_MCP], axis=-1)
    good = (conf[:, :, WRIST] >= conf_thr) & (conf[:, :, MID_MCP] >= conf_thr)
    scale = np.empty(S)
    for s in range(S):
        b = bone[s, good[s]] if good[s].any() else bone[s]
        scale[s] = np.median(b)
    if (scale <= 5e-4).any():
        bad = int((scale <= 5e-4).sum())
        raise ValueError(f"{bad} windows with degenerate hand scale <=5e-4 "
                         "(collapsed detections); tighten window gating")
    mu4 = mu[:, None, None, :]
    sc4 = scale[:, None, None, None]
    traj_n = (traj - mu4) / sc4
    ctx = []
    for c in (wrist, elbow):
        ctx.append(None if c is None
                   else (np.asarray(c, np.float64) - mu[:, None, :]) / scale[:, None, None])
    return traj_n, ctx[0], ctx[1], mu4, sc4


def hand_uncanon(traj_n, mu, scale):
    return traj_n * scale + mu


def to_px(arr_xy, W, H):
    """[0,1]-normalized (...,2) coords -> pixels, per-clip W/H broadcastable."""
    out = np.array(arr_xy, dtype=np.float64, copy=True)
    out[..., 0] *= W
    out[..., 1] *= H
    return out
