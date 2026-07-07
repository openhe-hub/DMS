"""Run the inference-time DWPose whole-body estimator over a video and return
per-frame keypoints + confidences as dense arrays (NaN where no person found).

Coordinates are the DWPose native output: normalized to [0, 1] by image W/H.
"""
import os
import numpy as np

# Import the DWPose chain (torch / onnxruntime) BEFORE decord: on some envs
# (jubail2 mimicmotion) importing decord first double-initializes OpenMP and
# aborts (SIGABRT) when torch/onnxruntime then load. dwpose_detector is also
# instantiated here with relative weight paths ("<repo>/.../DWPose/..."), so the
# caller must run from the repo root.
from mimicmotion.dwpose.dwpose_detector import dwpose_detector as dwprocessor
import decord

# OpenPose-18 body ordering used by DWPose
BODY_N = 18
HAND_N = 21


def _primary_person(pose):
    """Index of the highest-total-confidence detected person, or None."""
    score = pose["bodies"]["score"]  # [nums, 18]
    if score is None or len(score) == 0:
        return None
    return int(np.argmax(score.sum(axis=1)))


def extract_video_poses(video_path, sample_stride=1, max_frames=None):
    """Return a dict of dense per-frame arrays for the primary signer.

    Keys:
        body        [N, 18, 2]   (NaN if frame has no person)
        body_score  [N, 18]      (NaN if no person)
        hands       [N, 2, 21, 2]  hand 0 = LEFT, 1 = RIGHT  (NaN if no person)
                    (COCO-WholeBody slicing order; verified against DWPose's
                    own body wrists on 3 hard27k clips, median dist 0.01 to
                    the matching wrist vs 0.13+ to the other -- see
                    hand_traj.verify_hand_order / job 16540596)
        hands_score [N, 2, 21]     (NaN if no person)
        detected    [N] bool       True where a person was found
    """
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    idxs = list(range(0, len(vr), sample_stride))
    if max_frames:
        idxs = idxs[:max_frames]
    frames = vr.get_batch(idxs).asnumpy()

    N = len(frames)
    body = np.full((N, BODY_N, 2), np.nan, np.float32)
    body_s = np.full((N, BODY_N), np.nan, np.float32)
    hands = np.full((N, 2, HAND_N, 2), np.nan, np.float32)
    hands_s = np.full((N, 2, HAND_N), np.nan, np.float32)
    detected = np.zeros(N, bool)

    for i, frm in enumerate(frames):
        pose = dwprocessor(frm)
        nums = len(pose["bodies"]["score"])
        pi = _primary_person(pose)
        if pi is None or nums == 0:
            continue
        body[i] = pose["bodies"]["candidate"].reshape(nums, BODY_N, 2)[pi]
        body_s[i] = pose["bodies"]["score"][pi]
        # pose['hands'] is [2*nums, 21, 2]: first nums rows = right hand set,
        # next nums = left hand set (see dwpose_detector.__call__).
        h = pose["hands"].reshape(2, nums, HAND_N, 2)[:, pi]      # [2, 21, 2]
        hs = pose["hands_score"].reshape(2, nums, HAND_N)[:, pi]  # [2, 21]
        hands[i] = h
        hands_s[i] = hs
        detected[i] = True

    dwprocessor.release_memory()
    return dict(body=body, body_score=body_s, hands=hands,
                hands_score=hands_s, detected=detected)


def save_poses(path, poses):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **poses)


def load_poses(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}
