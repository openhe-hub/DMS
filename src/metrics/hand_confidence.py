"""Metric 1 -- hand structural quality via DWPose confidence.

Given the dense pose arrays from pose_extract, we measure, over frames where a
person is detected, how confident the hand keypoints are and how often a hand is
"well-formed" (mean keypoint confidence above a threshold). Garbled / hallucinated
hands (blobs, fused fingers) drive DWPose confidence down, so this is a direct,
alignment-free proxy for hand quality. Person-not-found frames are reported
separately as body_det_rate so the two failure modes never conflate.
"""
import numpy as np


def hand_confidence_metrics(poses, det_thr=0.3):
    det = poses["detected"]
    N = len(det)
    out = dict(n_frames=int(N), body_det_rate=float(det.mean()) if N else 0.0)

    if det.sum() == 0:
        out.update(mean_hand_conf=0.0, hand_good_rate=0.0, n_body_det=0)
        return out

    hs = poses["hands_score"][det]          # [M, 2, 21], no NaN (person present)
    per_hand = hs.mean(axis=2)              # [M, 2] mean keypoint conf per hand
    out.update(
        n_body_det=int(det.sum()),
        mean_hand_conf=float(per_hand.mean()),
        hand_good_rate=float((per_hand > det_thr).mean()),
    )
    return out
