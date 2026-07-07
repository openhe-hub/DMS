import copy

from tqdm import tqdm
import decord
import numpy as np

from .util import draw_pose
from .dwpose_detector import dwpose_detector as dwprocessor
from .graft import graft_pose_v2, blend_head_pose_only
from .hand_control import person0_hands

def get_video_pose(
        video_path: str,
        ref_image: np.ndarray,
        sample_stride: int=1,
        graft: bool=False,
        head_blend_ratio: float=0.15,
        return_hands: bool=False):
    """preprocess ref image pose and video pose

    Args:
        video_path (str): video pose path
        ref_image (np.ndarray): reference image
        sample_stride (int, optional): Defaults to 1.
        graft (bool, optional): apply graft_pose_v2 retargeting (ref body/face
            frozen, video arms+hands transplanted) + 15% head blend, matching
            the jubail2 sign-language MimicMotion fork. Defaults to False
            (exact original behavior).
        return_hands (bool, optional): additionally return the per-frame
            person-0 hand keypoints + confidences (post-rescale, post-graft,
            i.e. the same coordinates the drawn skeleton uses) for the
            hand_flow motion-field channel. Defaults to False (original
            3-tuple return).

    Returns:
        np.ndarray: sequence of video pose
    """
    # select ref-keypoint from reference pose for pose rescale
    ref_pose = dwprocessor(ref_image)
    ref_keypoint_id = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    ref_keypoint_id = [i for i in ref_keypoint_id \
        if len(ref_pose['bodies']['subset']) > 0 and ref_pose['bodies']['subset'][0][i] >= .0]
    ref_body = ref_pose['bodies']['candidate'][ref_keypoint_id]

    height, width, _ = ref_image.shape

    # read input video
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    sample_stride *= max(1, int(vr.get_avg_fps() / 24))

    frames = vr.get_batch(list(range(0, len(vr), sample_stride))).asnumpy()
    detected_poses = [dwprocessor(frm) for frm in tqdm(frames, desc="DWPose")]
    dwprocessor.release_memory()

    detected_bodies = np.stack(
        [p['bodies']['candidate'] for p in detected_poses if p['bodies']['candidate'].shape[0] == 18])[:,
                      ref_keypoint_id]
    # compute linear-rescale params
    ay, by = np.polyfit(detected_bodies[:, :, 1].flatten(), np.tile(ref_body[:, 1], len(detected_bodies)), 1)
    fh, fw, _ = vr[0].shape
    ax = ay / (fh / fw / height * width)
    bx = np.mean(np.tile(ref_body[:, 0], len(detected_bodies)) - detected_bodies[:, :, 0].flatten() * ax)
    a = np.array([ax, ay])
    b = np.array([bx, by])
    output_pose = []
    # pose rescale
    body_point = []
    face_point = []
    hand_point = []
    for detected_pose in detected_poses:
        detected_pose['bodies']['candidate'] = detected_pose['bodies']['candidate'] * a + b
        detected_pose['faces'] = detected_pose['faces'] * a + b
        detected_pose['hands'] = detected_pose['hands'] * a + b
        if graft and detected_pose['bodies']['candidate'].shape[0] == 18:
            # ref_pose is already in ref-image space; detected_pose was just
            # rescaled into it, so grafting operates in one coordinate frame.
            video_pose_transformed = copy.deepcopy(detected_pose)
            detected_pose = graft_pose_v2(ref_pose, detected_pose)
            detected_pose = blend_head_pose_only(detected_pose, video_pose_transformed,
                                                 blend_ratio=head_blend_ratio)
        im = draw_pose(detected_pose, height, width)
        output_pose.append(np.array(im))
        body_point.append(detected_pose['bodies'])
        face_point.append(detected_pose['faces'])
        if return_hands:
            h, s = person0_hands(detected_pose)
            hand_point.append(dict(hands=h, hands_score=s))
    if return_hands:
        return np.stack(output_pose), body_point, face_point, hand_point
    return np.stack(output_pose), body_point, face_point


def get_image_pose(ref_image):
    """process image pose

    Args:
        ref_image (np.ndarray): reference image pixel value

    Returns:
        np.ndarray: pose visual image in RGB-mode
    """
    height, width, _ = ref_image.shape
    ref_pose = dwprocessor(ref_image)
    pose_img = draw_pose(ref_pose, height, width)
    return np.array(pose_img), ref_pose
