"""Hand keypoints -> motion-field point list (the hand_flow control channel).

DWPose detects 21 keypoints per hand but DisPose's motion-field branch consumes
only the 18 body points; these helpers append (a subset of) the hand keypoints
to the per-frame candidate/subset/score dicts so `points_to_flows` /
`pose2track` (with n_points=18+2k) scatter hand displacement into traj_flow and
the CMP sparse flow. The skeleton-image branch and the DIFT/point-adapter
branch are untouched.

Conventions preserved: low-confidence hand keypoints get subset=-1 (the
existing invisible-keypoint convention, skipped by points_to_flows), and the
appended body rows are person-0 only so multi-person frames cannot leak a
second person's joints into the extended index range.

Hand order: hands[0]=left, hands[1]=right per the COCO-WholeBody slicing in
dwpose_detector.py:58-59 and graft.py's usage. Order does not affect this
module (both hands are appended symmetrically).
"""
import numpy as np

HAND_N = 21
# wrist + fingertips: the coarse variant if 42 clustered points saturate the
# blurred traj_flow (kernel=199) or drown the nearby body-wrist flow.
TIPS = (0, 4, 8, 12, 16, 20)


def person0_hands(pose):
    """(hands (2,21,2), scores (2,21)) of person 0 from a full DWPose dict.

    pose['hands'] is (2*nums, 21, 2): rows [0..nums) = first-hand set,
    [nums..2*nums) = second-hand set, so person 0 owns rows 0 and nums --
    NOT rows 0 and 1 when nums > 1.
    """
    hands = np.asarray(pose.get("hands", np.zeros((0, HAND_N, 2))), dtype=np.float64)
    scores = pose.get("hands_score", None)
    if hands.shape[0] < 2:
        return np.zeros((2, HAND_N, 2)), np.zeros((2, HAND_N))
    nums = hands.shape[0] // 2
    h = hands.reshape(2, nums, HAND_N, 2)[:, 0]
    if scores is None or np.asarray(scores).shape[0] < 2 * nums:
        s = np.zeros((2, HAND_N))
    else:
        s = np.asarray(scores, dtype=np.float64).reshape(2, nums, HAND_N)[:, 0]
    return h, s


def smooth_hands(hand_list, sigma, conf_floor=0.05, r=None):
    """Confidence-weighted temporal Gaussian over a hand sequence.

    hand_list: list of dicts {hands (2,21,2), hands_score (2,21)} (video frames
    only -- do NOT include the static ref frame, it would bleed into t=0).
    Low-confidence detections contribute proportionally less to their
    neighbours instead of dragging them; scores are passed through unchanged.
    """
    if sigma <= 0 or len(hand_list) < 2:
        return hand_list
    if r is None:
        r = max(1, int(round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / float(sigma)) ** 2)
    T = len(hand_list)
    H = np.stack([np.asarray(e["hands"], dtype=np.float64) for e in hand_list])   # (T,2,21,2)
    S = np.stack([np.asarray(e["hands_score"], dtype=np.float64) for e in hand_list])  # (T,2,21)
    W = np.clip(S, conf_floor, 1.0)[..., None]                                     # (T,2,21,1)
    num = np.zeros_like(H)
    den = np.zeros_like(W)
    for j, kw in enumerate(k):
        off = j - r
        src = slice(max(0, -off), T - max(0, off))
        dst = slice(max(0, off), T - max(0, -off))
        num[dst] += kw * (H[src] * W[src])
        den[dst] += kw * W[src]
    Hs = num / np.maximum(den, 1e-12)
    out = []
    for t, e in enumerate(hand_list):
        e2 = dict(e)
        e2["hands"] = Hs[t]
        out.append(e2)
    return out


def append_hands_to_point_list(body_point_list, hand_list, conf_thr=0.3,
                               kp_subset="all"):
    """Build the motion-field point list: person-0 body 18 + 2k hand keypoints.

    body_point_list: per-frame dicts {candidate (nums*18,2), subset (nums,18),
    score (nums,18)} as consumed by pose2track. hand_list: per-frame dicts
    {hands (2,21,2), hands_score (2,21)}, same length, same (ref-image)
    coordinate space. Returns (flow_point_list, n_points).
    """
    assert len(body_point_list) == len(hand_list), \
        (len(body_point_list), len(hand_list))
    idx = list(range(HAND_N)) if kp_subset == "all" else list(TIPS)
    k = len(idx)
    n_points = 18 + 2 * k
    out = []
    for bp, hp in zip(body_point_list, hand_list):
        cand = np.asarray(bp["candidate"], dtype=np.float64)[:18]
        sub = np.asarray(bp["subset"], dtype=np.float64)[0, :18]
        sco = np.asarray(bp["score"], dtype=np.float64)[0, :18]
        hands = np.asarray(hp["hands"], dtype=np.float64)[:, idx]      # (2,k,2)
        hsco = np.asarray(hp["hands_score"], dtype=np.float64)[:, idx]  # (2,k)
        hc = hands.reshape(2 * k, 2)
        hs = hsco.reshape(2 * k)
        vis = (hs >= conf_thr) & np.isfinite(hc).all(axis=1)
        hsub = np.where(vis, 18 + np.arange(2 * k), -1.0)
        hc = np.where(np.isfinite(hc), hc, 0.0)
        out.append(dict(
            candidate=np.concatenate([cand, hc], axis=0),
            subset=np.concatenate([sub, hsub])[None],
            score=np.concatenate([sco, hs])[None],
        ))
    return out, n_points
