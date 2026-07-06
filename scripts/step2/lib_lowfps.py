"""Step 2 low-fps driving library.

Core idea of the pilot (R011): DWPose runs ONCE on the full sampling grid; each
system then only sees every `stride`-th detection and must reconstruct the
control signal on the full grid (linear / spline / per-clip SIREN), or stay on
the coarse grid (original DisPose = the discrete baseline). Sharing one
detection pass + one rescale fit makes every comparison exactly paired.

Fairness rule: faces/hands (and all confidence scores) are ALWAYS linearly
interpolated, identically for every continuous method -- the compared variable
is ONLY how body-keypoint trajectories (the motion-field control) are
reconstructed between observed frames.
"""
import numpy as np
import torch
import torch.nn.functional as F

import _paths  # noqa: F401

from dispose_siren.interp import natural_cubic_pos, perclip_fit_decode


# --------------------------------------------------------------- detection
def detect_video_full(video_path, ref_image_hw3, sample_stride=1, max_frames=None):
    """get_video_pose twin that also returns the RGB frames (self-driven GT)
    and the FULL rescaled pose dicts (bodies+faces+hands+scores) per frame."""
    import decord
    from mimicmotion.dwpose.dwpose_detector import dwpose_detector as dwprocessor

    ref_pose = dwprocessor(ref_image_hw3)
    ref_keypoint_id = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    ref_keypoint_id = [i for i in ref_keypoint_id
                       if len(ref_pose['bodies']['subset']) > 0
                       and ref_pose['bodies']['subset'][0][i] >= .0]
    ref_body = ref_pose['bodies']['candidate'][ref_keypoint_id]
    height, width, _ = ref_image_hw3.shape

    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    stride = sample_stride * max(1, int(vr.get_avg_fps() / 24))
    idx = list(range(0, len(vr), stride))
    if max_frames is not None:
        idx = idx[:max_frames]
    frames = vr.get_batch(idx).asnumpy()
    detected = [dwprocessor(frm) for frm in frames]
    dwprocessor.release_memory()

    detected_bodies = np.stack([p['bodies']['candidate'] for p in detected
                                if p['bodies']['candidate'].shape[0] == 18])[:, ref_keypoint_id]
    ay, by = np.polyfit(detected_bodies[:, :, 1].flatten(),
                        np.tile(ref_body[:, 1], len(detected_bodies)), 1)
    fh, fw, _ = vr[0].shape
    ax = ay / (fh / fw / height * width)
    bx = np.mean(np.tile(ref_body[:, 0], len(detected_bodies))
                 - detected_bodies[:, :, 0].flatten() * ax)
    a, b = np.array([ax, ay]), np.array([bx, by])
    for p in detected:
        p['bodies']['candidate'] = p['bodies']['candidate'] * a + b
        p['faces'] = p['faces'] * a + b
        p['hands'] = p['hands'] * a + b
    return frames, detected, ref_pose


def frames_to_target(frames_thw3, h_target, w_target):
    """Raw RGB frames -> (T,3,H,W) uint8 on the model's output geometry (same
    resize + center-crop as the reference image path)."""
    from torchvision.transforms.functional import resize, center_crop
    t = torch.from_numpy(frames_thw3).permute(0, 3, 1, 2)          # (T,3,h,w)
    h, w = t.shape[-2:]
    hw = float(h) / float(w)
    if hw < h_target / w_target:
        hr, wr = h_target, int(np.ceil(h_target / hw))
    else:
        hr, wr = int(np.ceil(w_target * hw)), w_target
    t = resize(t, [hr, wr], antialias=None)
    return center_crop(t, [h_target, w_target]).to(torch.uint8)


def person0(detected):
    """Keep only the primary person in every pose dict. DisPose's control path
    (pose2track / points_to_flows) already uses person 0 exclusively; this makes
    the drawn skeleton consistent with the control signal and lets multi-person
    frames (spurious background detections) through the pilot."""
    out = []
    for d in detected:
        out.append(dict(
            bodies=dict(candidate=d["bodies"]["candidate"][:18],
                        subset=np.asarray(d["bodies"]["subset"])[:1, :18],
                        score=np.asarray(d["bodies"]["score"])[:1, :18]),
            faces=d["faces"][:1], faces_score=d["faces_score"][:1],
            hands=d["hands"][:2], hands_score=d["hands_score"][:2]))
    return out


# --------------------------------------------------------------- interpolation
def _lin(arr_obs, t_obs, t_eval):
    """Linear interp along axis 0 for any (n, ...) array."""
    flat = arr_obs.reshape(len(arr_obs), -1)
    out = np.stack([np.interp(t_eval, t_obs, flat[:, c])
                    for c in range(flat.shape[1])], 1)
    return out.reshape((len(t_eval),) + arr_obs.shape[1:])


def interp_pose_dicts(detected, stride, method, siren_cfg=None, verbose=False):
    """Subsample every `stride`-th pose dict, reconstruct on the full grid.

    Returns (dicts_eval, t_eval, obs_i). Grid truncated at the last observed
    frame so every method interpolates (never extrapolates).
    method: 'linear' | 'spline' | 'siren'.
    """
    T = len(detected)
    obs_i = np.arange(0, T, stride)
    last = int(obs_i[-1])
    t_eval = np.arange(last + 1)
    n = len(obs_i)

    cand = np.stack([d['bodies']['candidate'][:18] for d in detected])       # (T,18,2)
    sub = np.stack([np.asarray(d['bodies']['subset'])[0][:18] for d in detected])
    bsc = np.stack([np.asarray(d['bodies']['score'])[0][:18] for d in detected])
    faces = np.stack([d['faces'] for d in detected])
    fsc = np.stack([d['faces_score'] for d in detected])
    hands = np.stack([d['hands'] for d in detected])
    hsc = np.stack([d['hands_score'] for d in detected])

    co, so = cand[obs_i], (sub[obs_i] != -1)                                  # (n,18,2), (n,18)
    full_vis = so.all(axis=0)                                                 # kps visible at every obs

    # body coordinates: `method` for fully-visible kps, per-segment linear else
    new_cand = np.empty((len(t_eval), 18, 2))
    A = np.where(full_vis)[0]
    if len(A):
        obs_a = co[:, A].transpose(1, 0, 2)                                   # (S,n,2)
        if method == "linear":
            pos = _lin(obs_a.transpose(1, 0, 2), obs_i, t_eval)               # (E,S,2)
        elif method == "spline":
            pos = natural_cubic_pos(obs_a, t_eval / stride).transpose(1, 0, 2)
        elif method == "siren":
            cfg = siren_cfg or {"w0": 5.0, "lam": 0.0, "steps": 800}
            w0_eff = cfg["w0"] * max(1.0, (n - 1) / 15.0)
            pos = perclip_fit_decode(obs_a, obs_i / last, t_eval / last,
                                     w0=w0_eff, lam=cfg.get("lam", 0.0),
                                     steps=cfg.get("steps", 800),
                                     verbose=verbose).transpose(1, 0, 2)
        else:
            raise ValueError(method)
        new_cand[:, A] = pos
    for k in np.where(~full_vis)[0]:                                          # fallback: linear on visible obs
        vk = np.where(so[:, k])[0]
        if len(vk) >= 2:
            new_cand[:, k] = _lin(co[vk, k], obs_i[vk], t_eval)
        else:
            new_cand[:, k] = co[vk[0], k] if len(vk) else cand[0, k]

    # Observed frames keep the RAW detections for every keypoint (DisPose's
    # sparse-flow branch consumes coords regardless of subset visibility, so
    # even "garbage" invisible coords must match the original at obs times).
    # => systems differ ONLY in between-frame reconstruction, and the stride=1
    # path is bit-exact vs the unmodified pipeline.
    new_cand[obs_i] = co

    # visibility on the eval grid: visible iff both bracketing obs are visible
    visf = _lin(so.astype(float), obs_i, t_eval) >= 0.999                     # (E,18)
    new_sub = np.where(visf, np.arange(18)[None, :].astype(float), -1.0)
    new_bsc = _lin(bsc[obs_i], obs_i, t_eval)
    new_faces, new_fsc = _lin(faces[obs_i], obs_i, t_eval), _lin(fsc[obs_i], obs_i, t_eval)
    new_hands, new_hsc = _lin(hands[obs_i], obs_i, t_eval), _lin(hsc[obs_i], obs_i, t_eval)

    dicts = []
    for t in range(len(t_eval)):
        dicts.append(dict(
            bodies=dict(candidate=new_cand[t], subset=new_sub[t:t + 1],
                        score=new_bsc[t:t + 1]),
            faces=new_faces[t], faces_score=new_fsc[t],
            hands=new_hands[t], hands_score=new_hsc[t]))
    return dicts, t_eval, obs_i


def draw_sequence(dicts, h, w):
    from mimicmotion.dwpose.util import draw_pose
    return np.stack([np.array(draw_pose(d, h, w)) for d in dicts])


# --------------------------------------------------------------- control build
def apply_cmp(cmp, lite, device):
    """CMP sparse->dense flow on build_control(lite=True) outputs (same math as
    the non-lite branch of build_control)."""
    from mimicmotion.utils.utils import get_cmp_flow
    sf = lite["sparse_flow"]
    fb, fl, fc, fh, fw = sf.shape
    flow = get_cmp_flow(
        cmp,
        lite["first_frame_384"].unsqueeze(0).repeat(1, fl, 1, 1, 1).to(device),
        lite["sparse_flow_384"].to(device),
        lite["mask_384"].to(device))
    if fh != 384 or fw != 384:
        scales = [fh / 384, fw / 384]
        flow = F.interpolate(flow.flatten(0, 1), (fh, fw),
                             mode="nearest").reshape(fb, fl, 2, fh, fw)
        flow[:, :, 0] *= scales[1]
        flow[:, :, 1] *= scales[0]
    return flow


def compute_dift(image_pixels_1x3hw, dift_model_path, device):
    """DIFT reference features, once per case (identical to build_control)."""
    from mimicmotion.utils.dift_utils import SDFeaturizer
    dift_model = SDFeaturizer(sd_id=dift_model_path, weight_dtype=torch.float16)
    img = (image_pixels_1x3hw / 255.0 - 0.5) * 2
    img = torch.from_numpy(img).to(device, torch.float16)
    feats = dift_model.forward(img, prompt="photo of a human", t=[261, 0],
                               up_ft_index=[1, 2], ensemble_size=8)
    del dift_model
    torch.cuda.empty_cache()
    return feats
