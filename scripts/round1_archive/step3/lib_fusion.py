"""Sampling-time latent inter-frame fusion (step3, R101/R102).

Mid-frame latents are softly pulled toward flow-warped bracketing keyframe
latents after each denoising step, via the pipeline's callback_on_step_end
hook (pipeline_ctrl.py allows the callback to rewrite `latents`).

Backward flows m->k are built directly from the interpolated keypoints:
visible-keypoint displacements splatted at the mid frame's keypoint
positions, Nadaraya-Watson normalized with a Gaussian kernel. Deterministic,
no CMP involved (avoids the sample_optical_flow scatter race). Where no
keypoint has support the flow decays to 0 (static background).

Latent frame layout of Ctrl_Pipeline: frame 0 is the reference pose frame
(dropped from the output video), so video frame j lives at latent index j+1.
"""
import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------- flow builder
def _nw_flow(cand_src, cand_dst, vis, h_lat, w_lat, sigma):
    """Backward flow field src->dst on the src frame's grid, latent pixels.

    cand_*: (18,2) normalized (x,y) in [0,1]; vis: (18,) bool = visible in
    BOTH frames. Returns (2,h_lat,w_lat) float32 numpy (dx,dy) such that the
    content of dst can be sampled at grid + flow to synthesize src.
    """
    flow = np.zeros((2, h_lat, w_lat), dtype=np.float32)
    if vis.sum() == 0:
        return flow
    px = cand_src[vis, 0] * w_lat
    py = cand_src[vis, 1] * h_lat
    dx = (cand_dst[vis, 0] - cand_src[vis, 0]) * w_lat
    dy = (cand_dst[vis, 1] - cand_src[vis, 1]) * h_lat
    yy, xx = np.mgrid[0:h_lat, 0:w_lat].astype(np.float32)
    d2 = (yy[..., None] - py) ** 2 + (xx[..., None] - px) ** 2   # (h,w,J)
    w = np.exp(-d2 / (2.0 * sigma * sigma))
    den = w.sum(-1)
    # decay to zero flow away from keypoint support (background static):
    # the +1 in the denominator acts as a zero-displacement pseudo-sample.
    flow[0] = (w * dx).sum(-1) / (den + 1.0)
    flow[1] = (w * dy).sum(-1) / (den + 1.0)
    return flow


def build_fusion_plan(cand, sub, obs_idx, h_lat, w_lat, sigma=6.0):
    """Precompute, for every mid frame, the grid_sample grids toward its two
    bracketing keyframes.

    cand: (T,18,2) normalized coords on the FULL grid (interpolated),
    sub:  (T,18)   visibility (>=0 visible) -- raw at obs, interp elsewhere,
    obs_idx: sorted array of observed (keyframe) video-frame indices.

    Returns list of dicts with LATENT indices (video idx + 1) and grids
    (1,h,w,2) float32 torch tensors for F.grid_sample(align_corners=True).
    """
    obs_idx = np.asarray(sorted(obs_idx))
    base_y, base_x = np.mgrid[0:h_lat, 0:w_lat].astype(np.float32)
    plan = []
    for a, b in zip(obs_idx[:-1], obs_idx[1:]):
        for m in range(a + 1, b):
            entries = []
            for k in (a, b):
                vis = (sub[m] >= 0) & (sub[k] >= 0)
                fl = _nw_flow(cand[m], cand[k], vis, h_lat, w_lat, sigma)
                gx = 2.0 * (base_x + fl[0]) / max(w_lat - 1, 1) - 1.0
                gy = 2.0 * (base_y + fl[1]) / max(h_lat - 1, 1) - 1.0
                grid = torch.from_numpy(np.stack([gx, gy], -1))[None]  # (1,h,w,2)
                entries.append(grid)
            w0 = float(b - m) / float(b - a)
            plan.append(dict(m_lat=m + 1, k0_lat=int(a) + 1, k1_lat=int(b) + 1,
                             w0=w0, grid0=entries[0], grid1=entries[1]))
    return plan


# ------------------------------------------------------------------ callback
class FusionCallback:
    """callback_on_step_end: lat[m] <- (1-a)*lat[m] + a*(w0*warp(lat[k0]) +
    (1-w0)*warp(lat[k1])), active only for progress in [win_lo, win_hi]."""

    def __init__(self, plan, num_steps, alpha=0.3, win=(0.3, 0.9)):
        self.plan = plan
        self.n = int(num_steps)
        self.alpha = float(alpha)
        self.win = win
        self.calls = 0

    def __call__(self, pipe, step_i, t, kw):
        lat = kw["latents"]                       # (1,F,4,h,w)
        prog = step_i / max(self.n - 1, 1)
        if not (self.win[0] <= prog <= self.win[1]):
            return {"latents": lat}
        a = self.alpha
        for e in self.plan:
            if e["grid0"].device != lat.device or e["grid0"].dtype != lat.dtype:
                e["grid0"] = e["grid0"].to(lat.device, lat.dtype)
                e["grid1"] = e["grid1"].to(lat.device, lat.dtype)
            wa = F.grid_sample(lat[:, e["k0_lat"]], e["grid0"],
                               mode="bilinear", padding_mode="border",
                               align_corners=True)
            wb = F.grid_sample(lat[:, e["k1_lat"]], e["grid1"],
                               mode="bilinear", padding_mode="border",
                               align_corners=True)
            ref = e["w0"] * wa + (1.0 - e["w0"]) * wb
            lat[:, e["m_lat"]] = (1.0 - a) * lat[:, e["m_lat"]] + a * ref
        self.calls += 1
        return {"latents": lat}


# ------------------------------------------------ x0-space fusion (round 2)
class X0FusionScheduler:
    """Wraps EulerDiscreteScheduler: fuse pred_original_sample (x0) across
    frames, then recompute prev_sample with the fused x0 -- an exact Euler
    step, so noise statistics are untouched (round-1 latent-space blending
    destroyed them: LPIPS 2x worse at alpha=0.15, catastrophic in early
    windows)."""

    def __init__(self, scheduler, plan, num_steps, alpha=0.3, win=(0.0, 0.85)):
        self._s = scheduler
        self.plan = plan
        self.n = int(num_steps)
        self.alpha = float(alpha)
        self.win = win
        self.calls = 0

    def __getattr__(self, k):
        return getattr(object.__getattribute__(self, "_s"), k)

    def step(self, model_output, timestep, sample, **kw):
        s = self._s
        kw.pop("return_dict", None)
        i = int((s.timesteps == timestep).nonzero()[0].item())
        out = s.step(model_output, timestep, sample, return_dict=True, **kw)
        prog = i / max(self.n - 1, 1)
        if not (self.win[0] <= prog <= self.win[1]) or i + 1 >= len(s.sigmas):
            return (out.prev_sample,)
        x0 = out.pred_original_sample.clone()          # (1,F,4,h,w)
        a = self.alpha
        for e in self.plan:
            if e["grid0"].device != x0.device or e["grid0"].dtype != x0.dtype:
                e["grid0"] = e["grid0"].to(x0.device, x0.dtype)
                e["grid1"] = e["grid1"].to(x0.device, x0.dtype)
            wa = F.grid_sample(x0[:, e["k0_lat"]], e["grid0"], mode="bilinear",
                               padding_mode="border", align_corners=True)
            wb = F.grid_sample(x0[:, e["k1_lat"]], e["grid1"], mode="bilinear",
                               padding_mode="border", align_corners=True)
            ref = e["w0"] * wa + (1.0 - e["w0"]) * wb
            x0[:, e["m_lat"]] = (1.0 - a) * x0[:, e["m_lat"]] + a * ref
        sc = s.sigmas[i].to(torch.float32)
        sn = s.sigmas[i + 1].to(torch.float32)
        prev = (sample.float() + (sn - sc) *
                (sample.float() - x0.float()) / sc).to(sample.dtype)
        self.calls += 1
        return (prev,)


# ------------------------------------------------- run_pipeline with callback
def run_pipeline_cb(pipeline, image_pixels, pose_pixels, controlnet_flow,
                    controlnet_image, point_list, dift_feats, traj_flow,
                    device, task_config, callback):
    """inference_ctrl.run_pipeline + callback_on_step_end passthrough."""
    from torchvision.transforms.functional import to_pil_image
    image_pixels = [to_pil_image(img.to(torch.uint8))
                    for img in (image_pixels + 1.0) * 127.5]
    generator = torch.Generator(device=device)
    generator.manual_seed(task_config.seed)
    with torch.autocast("cuda"):
        frames = pipeline(
            image_pixels, image_pose=pose_pixels,
            num_frames=pose_pixels.size(0),
            tile_size=task_config.num_frames,
            tile_overlap=task_config.frames_overlap,
            height=pose_pixels.shape[-2], width=pose_pixels.shape[-1], fps=7,
            controlnet_flow=controlnet_flow, controlnet_image=controlnet_image,
            point_list=point_list, dift_feats=dift_feats, traj_flow=traj_flow,
            noise_aug_strength=task_config.noise_aug_strength,
            num_inference_steps=task_config.num_inference_steps,
            generator=generator,
            min_guidance_scale=task_config.guidance_scale,
            max_guidance_scale=task_config.guidance_scale,
            decode_chunk_size=task_config.decode_chunk_size,
            output_type="pt", device=device,
            callback_on_step_end=callback,
        ).frames.cpu()
    video_frames = (frames * 255.0).to(torch.uint8)
    return video_frames[0, 1:]
