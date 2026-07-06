"""Pose grafting / retargeting, ported verbatim from the jubail2 sign-language
MimicMotion fork (mimicmotion/dwpose/preprocess_unified.py, graft_pose_v2 +
blend_head_pose_only). Used as an OPT-IN switch in get_video_pose(graft=True)
so DisPose and MimicMotion can share the exact same retargeted pose condition
for sign-video comparisons. All poses are expected in the reference-image
coordinate space (i.e. video poses already rescaled by the a/b linear fit).
"""
import copy

import numpy as np


def graft_pose_v2(ref_pose: dict, video_pose: dict) -> dict:
    """Retarget video pose to match reference pose proportions.

    Starts from a deep copy of the REFERENCE pose (body/face/legs frozen to the
    reference), then transplants only the arms (elbow/wrist) and hand keypoints
    from the video, scaled per side by the shoulder-neck distance ratio.
    """
    NECK_ID = 1
    R_SHOULDER_ID = 2
    R_ELBOW_ID = 3
    R_WRIST_ID = 4
    L_SHOULDER_ID = 5
    L_ELBOW_ID = 6
    L_WRIST_ID = 7

    grafted_pose = copy.deepcopy(ref_pose)

    # Calculate shoulder-neck distances
    ref_body = ref_pose['bodies']['candidate']
    video_body = video_pose['bodies']['candidate']
    dist_ref_l_shoulder = np.linalg.norm(ref_body[L_SHOULDER_ID] - ref_body[NECK_ID])
    dist_video_l_shoulder = np.linalg.norm(video_body[L_SHOULDER_ID] - video_body[NECK_ID])
    dist_ref_r_shoulder = np.linalg.norm(ref_body[R_SHOULDER_ID] - ref_body[NECK_ID])
    dist_video_r_shoulder = np.linalg.norm(video_body[R_SHOULDER_ID] - video_body[NECK_ID])
    scale_left = dist_ref_l_shoulder / (dist_video_l_shoulder + 1e-6)
    scale_right = dist_ref_r_shoulder / (dist_video_r_shoulder + 1e-6)

    # Scale arm & elbow in body keypoints
    vec_video_l_shoulder_elbow = video_body[L_ELBOW_ID] - video_body[L_SHOULDER_ID]
    grafted_pose['bodies']['candidate'][L_ELBOW_ID] = ref_body[L_SHOULDER_ID] + vec_video_l_shoulder_elbow * scale_left
    vec_video_l_elbow_wrist = video_body[L_WRIST_ID] - video_body[L_ELBOW_ID]
    grafted_pose['bodies']['candidate'][L_WRIST_ID] = grafted_pose['bodies']['candidate'][L_ELBOW_ID] + vec_video_l_elbow_wrist * scale_left
    vec_video_r_shoulder_elbow = video_body[R_ELBOW_ID] - video_body[R_SHOULDER_ID]
    grafted_pose['bodies']['candidate'][R_ELBOW_ID] = ref_body[R_SHOULDER_ID] + vec_video_r_shoulder_elbow * scale_right
    vec_video_r_elbow_wrist = video_body[R_WRIST_ID] - video_body[R_ELBOW_ID]
    grafted_pose['bodies']['candidate'][R_WRIST_ID] = grafted_pose['bodies']['candidate'][R_ELBOW_ID] + vec_video_r_elbow_wrist * scale_right

    # Scale hand keypoints
    if video_pose['hands'].any():
        video_l_hand_kps = video_pose['hands'][0]
        grafted_l_wrist_coord = grafted_pose['bodies']['candidate'][L_WRIST_ID]
        grafted_pose['hands'][0] = grafted_l_wrist_coord + (video_l_hand_kps - video_body[L_WRIST_ID]) * scale_left

        video_r_hand_kps = video_pose['hands'][1]
        grafted_r_wrist_coord = grafted_pose['bodies']['candidate'][R_WRIST_ID]
        grafted_pose['hands'][1] = grafted_r_wrist_coord + (video_r_hand_kps - video_body[R_WRIST_ID]) * scale_right

    if 'hands_score' in video_pose and video_pose['hands_score'].any():
        grafted_pose['hands_score'] = video_pose['hands_score']

    return grafted_pose


def blend_head_pose_only(ref_pose: dict, video_pose: dict, blend_ratio: float = 0.15) -> dict:
    """Blend only the 5 head keypoints (nose, eyes, ears) from the video into
    the reference pose. Face landmarks stay 100% reference for stability.
    blend_ratio=0.15 was validated artifact-free on the jubail2 fork; 0.20+
    caused artifacts there.
    """
    HEAD_KEYPOINT_IDS = [0, 14, 15, 16, 17]  # nose, left eye, right eye, left ear, right ear

    result_pose = copy.deepcopy(ref_pose)

    for kp_id in HEAD_KEYPOINT_IDS:
        ref_kp = ref_pose['bodies']['candidate'][kp_id]
        video_kp = video_pose['bodies']['candidate'][kp_id]
        result_pose['bodies']['candidate'][kp_id] = ref_kp * (1 - blend_ratio) + video_kp * blend_ratio

    return result_pose
