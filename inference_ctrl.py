import os
import argparse
import logging
import math
from omegaconf import OmegaConf
from datetime import datetime
import time
from pathlib import Path
import PIL.Image
import numpy as np
import torch.jit
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import pil_to_tensor, resize, center_crop
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
import torch.nn.functional as F
from torchvision.transforms import PILToTensor
import torchvision

import decord
from einops import rearrange, repeat
from mimicmotion.utils.dift_utils import SDFeaturizer
from mimicmotion.utils.utils import points_to_flows, bivariate_Gaussian, sample_inputs_flow, get_cmp_flow, pose2track
from  mimicmotion.utils.visualizer import Visualizer, vis_flow_to_video
import cv2



from mimicmotion.utils.geglu_patch import patch_geglu_inplace
patch_geglu_inplace()

from constants import ASPECT_RATIO
from mimicmotion.utils.loader import create_ctrl_pipeline
from mimicmotion.utils.utils import save_to_mp4
from mimicmotion.dwpose.preprocess import get_video_pose, get_image_pose
from mimicmotion.dwpose.hand_control import (append_hands_to_point_list,
                                             smooth_hands, person0_hands)
from mimicmotion.modules.cmp_model import CMP


import pdb
logging.basicConfig(level=logging.INFO, format="%(asctime)s: [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_ref_image(image_path, resolution=576):
    """Load + resize/center-crop the reference image exactly as preprocess does.
    Returns (image_pixels_hw3 uint8 numpy, ref_img (3,H,W) tensor in [0,1],
    h_target, w_target)."""
    image_pixels = pil_loader(image_path)
    image_pixels = pil_to_tensor(image_pixels) # (c, h, w)
    h, w = image_pixels.shape[-2:]
    ############################ compute target h/w according to original aspect ratio ###############################
    if h>w:
        w_target, h_target = resolution, int(resolution / ASPECT_RATIO // 64) * 64
    elif h==w:
        w_target, h_target = resolution, resolution
    else:
        w_target, h_target = int(resolution / ASPECT_RATIO // 64) * 64, resolution
    h_w_ratio = float(h) / float(w)
    if h_w_ratio < h_target / w_target:
        h_resize, w_resize = h_target, math.ceil(h_target / h_w_ratio)
    else:
        h_resize, w_resize = math.ceil(w_target * h_w_ratio), w_target
    image_pixels = resize(image_pixels, [h_resize, w_resize], antialias=None)
    image_pixels = center_crop(image_pixels, [h_target, w_target])
    # h_target, w_target = image_pixels.shape[-2:]
    image_pixels = image_pixels.permute((1, 2, 0)).numpy()
    ##################################### get video flow #################################################
    transform = transforms.Compose(
        [

        transforms.Resize((h_target, w_target), antialias=None),
        transforms.CenterCrop((h_target, w_target)),
        transforms.ToTensor()
        ]
    )

    ref_img = transform(PIL.Image.fromarray(image_pixels))
    return image_pixels, ref_img, h_target, w_target


def preprocess(video_path, image_path, dift_model_path, resolution=576, sample_stride=2,
               kp_noise=0.0, kp_noise_seed=0, max_frames=None, graft=False,
               hand_flow=False, hand_flow_smooth=0.0, hand_conf_thr=0.3,
               hand_kp_subset="all", hand_recon_dir=""):
    """preprocess ref image pose and video pose

    Args:
        video_path (str): input video pose path
        image_path (str): reference image path
        resolution (int, optional):  Defaults to 576.
        sample_stride (int, optional): Defaults to 2.
        graft (bool, optional): apply graft_pose_v2 retargeting to the driving
            poses (see mimicmotion/dwpose/graft.py). Affects BOTH the drawn
            skeleton and the body/face point lists feeding the motion field,
            so the DisPose control branch stays consistent with the skeleton.
        hand_flow (bool, optional): append the DWPose hand keypoints (post
            rescale/graft, person 0) to the point list feeding the motion
            field (traj_flow + CMP sparse flow). The skeleton image and the
            point_list returned for the DIFT/point-adapter branch stay
            body-18. Defaults to False (exact original behavior).
        hand_flow_smooth (float, optional): sigma (frames) of a confidence-
            weighted temporal Gaussian over the injected hand keypoints;
            0 = raw detections.
        hand_conf_thr (float, optional): hand keypoints below this confidence
            are marked invisible (subset=-1) in the motion field.
        hand_kp_subset (str, optional): "all" (21/hand) or "tips"
            (wrist+fingertips, 6/hand) -- coarse fallback if 42 clustered
            points saturate the blurred traj_flow.
        hand_recon_dir (str, optional): directory of {clip}.npz reconstructed
            hand trajectories (43_reconstruct_hands); substituted for the
            detected hands before rescale/graft -- the SIREN arm of the
            three-system comparison. Requires hand_flow=True.
    """
    image_pixels, ref_img, h_target, w_target = load_ref_image(image_path, resolution)

    ##################################### get image&video pose value #################################################
    image_pose, ref_point = get_image_pose(image_pixels)
    ref_point_body, ref_point_head = ref_point["bodies"], ref_point["faces"]
    if hand_flow:
        hand_override = None
        if hand_recon_dir:
            clip = os.path.splitext(os.path.basename(video_path))[0]
            rp = os.path.join(hand_recon_dir, f"{clip}.npz")
            z = np.load(rp)
            hand_override = dict(hands=z["hands"], hands_score=z["hands_score"])
            logger.info(f"hand_recon override: {rp} "
                        f"(covered {z['covered'].mean():.0%})")
        video_pose, body_point, face_point, hand_point = get_video_pose(
            video_path, image_pixels, sample_stride=sample_stride, graft=graft,
            return_hands=True, hand_override=hand_override)
    else:
        video_pose, body_point, face_point = get_video_pose(video_path, image_pixels, sample_stride=sample_stride,
                                                            graft=graft)
    body_point_list = [ref_point_body] + body_point
    face_point_list = [ref_point_head] + face_point

    pose_pixels = np.concatenate([np.expand_dims(image_pose, 0), video_pose])
    image_pixels = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))

    # ---- sensitivity probe (training-free): optionally truncate the clip and
    # inject keypoint jitter into ONLY the motion-field branch. pose_pixels (the
    # drawn skeleton) is already built above from video_pose, so it stays clean;
    # the noise reaches traj_flow / CMP flow / point_adapter only. kp_noise is in
    # pixels (normalized internally). Defaults (0 / None) preserve exact behavior.
    if max_frames is not None:
        body_point_list = body_point_list[:max_frames]
        face_point_list = face_point_list[:max_frames]
        pose_pixels = pose_pixels[:max_frames]
    if kp_noise and kp_noise > 0:
        _rng = np.random.RandomState(kp_noise_seed)
        for _f in range(1, len(body_point_list)):          # keep ref frame 0 clean
            _cand = body_point_list[_f]['candidate'].copy()
            _sub = np.asarray(body_point_list[_f]['subset'])[0]
            for _i in range(_cand.shape[0]):
                if _i < len(_sub) and _sub[_i] != -1:        # only visible keypoints
                    _cand[_i, 0] += _rng.randn() * (kp_noise / w_target)
                    _cand[_i, 1] += _rng.randn() * (kp_noise / h_target)
            body_point_list[_f]['candidate'] = _cand

    # ---- hand_flow: a separate hand-augmented point list feeds ONLY the
    # motion field; body_point_list (returned as point_list for the DIFT /
    # point-adapter branch) stays body-18.
    flow_point_list, flow_n_points = body_point_list, 18
    if hand_flow:
        if hand_flow_smooth and hand_flow_smooth > 0:
            # smooth video frames only, then prepend the static ref frame, so
            # the reference hand cannot bleed into t=0
            hand_point = smooth_hands(hand_point, hand_flow_smooth)
        ref_h, ref_s = person0_hands(ref_point)  # ref_pose is already in ref space
        hand_list = [dict(hands=ref_h, hands_score=ref_s)] + hand_point
        if max_frames is not None:
            hand_list = hand_list[:max_frames]
        flow_point_list, flow_n_points = append_hands_to_point_list(
            body_point_list, hand_list, conf_thr=hand_conf_thr,
            kp_subset=hand_kp_subset)

    val_controlnet_flow, val_controlnet_image, dift_feats, traj_flow = build_control(
        image_pixels, ref_img, flow_point_list, dift_model_path, h_target, w_target,
        n_points=flow_n_points)

    return torch.from_numpy(pose_pixels.copy()) / 127.5 - 1, torch.from_numpy(image_pixels) / 127.5 - 1, val_controlnet_flow, val_controlnet_image, body_point_list, dift_feats, traj_flow


def build_control(image_pixels, ref_img, body_point_list, dift_model_path,
                  h_target, w_target, lite=False, n_points=18):
    """Build DisPose control inputs (DIFT feats, motion-field traj_flow, CMP dense
    flow) from an arbitrary body_point_list -- extracted from preprocess() so the
    step-2 low-fps experiments can feed INTERPOLATED keypoint sequences through
    the exact same control path. Call order (DIFT -> traj_flow -> CMP) preserved.

    lite=True skips the GPU-heavy DIFT/CMP stages and returns the pre-CMP control
    precursors instead -- used by the stride-1 equivalence check.

    image_pixels: (1,3,H,W) numpy, 0-255. ref_img: (3,H,W) tensor in [0,1].
    """
    if not lite:
        dift_model = SDFeaturizer(sd_id = dift_model_path, weight_dtype=torch.float16)
        category="human"
        prompt = f'photo of a {category}'
        dift_ref_img = (image_pixels / 255.0 - 0.5) *2
        dift_ref_img = torch.from_numpy(dift_ref_img).to(device, torch.float16)
        dift_feats = dift_model.forward(dift_ref_img, prompt=prompt, t=[261,0], up_ft_index=[1,2], ensemble_size=8)

    model_length = len(body_point_list)
    traj_flow = points_to_flows(body_point_list, model_length, h_target, w_target, n_points=n_points)
    blur_kernel = bivariate_Gaussian(kernel_size=199, sig_x=20, sig_y=20, theta=0, grid=None, isotropic=True)

    for i in range(0, model_length-1):
        traj_flow[i] = cv2.filter2D(traj_flow[i], -1, blur_kernel)

    traj_flow = rearrange(traj_flow, "f h w c -> f c h w")
    traj_flow = torch.from_numpy(traj_flow)
    traj_flow = traj_flow.unsqueeze(0)

    pc, ph, pw = ref_img.shape
    poses, poses_subset = pose2track(body_point_list, ph, pw, n_points=n_points)
    poses = torch.from_numpy(poses).permute(1,0,2)
    poses_subset = torch.from_numpy(poses_subset).permute(1,0,2)

    # pdb.set_trace()
    val_controlnet_image, val_sparse_optical_flow, \
    val_mask, val_first_frame_384, \
        val_sparse_optical_flow_384, val_mask_384 = sample_inputs_flow(ref_img.unsqueeze(0).float(), poses.unsqueeze(0), poses_subset.unsqueeze(0))

    if lite:
        return dict(traj_flow=traj_flow, controlnet_image=val_controlnet_image,
                    sparse_flow=val_sparse_optical_flow, mask=val_mask,
                    sparse_flow_384=val_sparse_optical_flow_384, mask_384=val_mask_384,
                    first_frame_384=val_first_frame_384)

    cmp = CMP(
        './mimicmotion/modules/cmp/experiments/semiauto_annot/resnet50_vip+mpii_liteflow/config.yaml',
        42000
    ).to(device)
    cmp.requires_grad_(False)

    fb, fl, fc, fh, fw = val_sparse_optical_flow.shape

    val_controlnet_flow = get_cmp_flow(
        cmp,
        val_first_frame_384.unsqueeze(0).repeat(1, fl, 1, 1, 1).to(device),
        val_sparse_optical_flow_384.to(device),
        val_mask_384.to(device)
    )

    if fh != 384 or fw != 384:
        scales = [fh / 384, fw / 384]
        val_controlnet_flow = F.interpolate(val_controlnet_flow.flatten(0, 1), (fh, fw), mode='nearest').reshape(fb, fl, 2, fh, fw)
        val_controlnet_flow[:, :, 0] *= scales[1]
        val_controlnet_flow[:, :, 1] *= scales[0]

    return val_controlnet_flow, val_controlnet_image, dift_feats, traj_flow


def run_pipeline(pipeline, image_pixels, pose_pixels,
                controlnet_flow, controlnet_image, point_list, dift_feats, traj_flow,
                device, task_config):
    image_pixels = [to_pil_image(img.to(torch.uint8)) for img in (image_pixels + 1.0) * 127.5]
    generator = torch.Generator(device=device)
    generator.manual_seed(task_config.seed)
    with torch.autocast("cuda"):
        frames = pipeline(
            image_pixels, image_pose=pose_pixels, num_frames=pose_pixels.size(0),
            tile_size=task_config.num_frames, tile_overlap=task_config.frames_overlap,
            height=pose_pixels.shape[-2], width=pose_pixels.shape[-1], fps=7,
            controlnet_flow=controlnet_flow, controlnet_image=controlnet_image, point_list=point_list, dift_feats=dift_feats, traj_flow=traj_flow,
            noise_aug_strength=task_config.noise_aug_strength, num_inference_steps=task_config.num_inference_steps,
            generator=generator, min_guidance_scale=task_config.guidance_scale, 
            max_guidance_scale=task_config.guidance_scale, decode_chunk_size=task_config.decode_chunk_size, output_type="pt", device=device
        ).frames.cpu()
    video_frames = (frames * 255.0).to(torch.uint8)

    for vid_idx in range(video_frames.shape[0]):
        # deprecated first frame because of ref image
        _video_frames = video_frames[vid_idx, 1:]

    return _video_frames


@torch.no_grad()
def main(args):
    if not args.no_use_float16 :
        torch.set_default_dtype(torch.float16)

    infer_config = OmegaConf.load(args.inference_config)
    pipeline = create_ctrl_pipeline(infer_config, device)

    for task in infer_config.test_case:
        ############################################## Pre-process data ##############################################
        pose_pixels, image_pixels, controlnet_flow, controlnet_image, point_list, dift_feats, traj_flow = preprocess(
            task.ref_video_path, task.ref_image_path, infer_config.dift_model_path,
            resolution=task.resolution, sample_stride=task.sample_stride,
            graft=task.get("graft_pose", False),
            hand_flow=task.get("hand_flow", False),
            hand_flow_smooth=task.get("hand_flow_smooth", 0.0),
            hand_conf_thr=task.get("hand_conf_thr", 0.3),
            hand_kp_subset=task.get("hand_kp_subset", "all"),
            hand_recon_dir=task.get("hand_recon_dir", "")
        )
        ########################################### Run MimicMotion pipeline ###########################################
        _video_frames = run_pipeline(
            pipeline, 
            image_pixels, pose_pixels, controlnet_flow, controlnet_image, point_list, dift_feats, traj_flow,
            device, task
        )
        ################################### save results to output folder. ###########################################
        save_to_mp4(
            _video_frames, 
            f"{args.output_dir}/{datetime.now().strftime('%Y%m%d')}_{args.name}/{datetime.now().strftime('%H%M%S')}_{os.path.basename(task.ref_image_path).split('.')[0]}_to_{os.path.basename(task.ref_video_path).split('.')[0]}" \
            f"_CFG{task.guidance_scale}_{task.num_frames}_{task.fps}.mp4",
            fps=task.fps,
        )

def set_logger(log_file=None, log_level=logging.INFO):
    log_handler = logging.FileHandler(log_file, "w")
    log_handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]: %(message)s")
    )
    log_handler.setLevel(log_level)
    logger.addHandler(log_handler)


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--inference_config", type=str, default="configs/test.yaml") #ToDo
    parser.add_argument("--output_dir", type=str, default="outputs/", help="path to output")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--no_use_float16",
                        action="store_true",
                        help="Whether use float16 to speed up inference",
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
    logger.info(f"--- Finished ---")

