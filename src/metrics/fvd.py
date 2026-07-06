"""FVD -- Frechet Video Distance using the StyleGAN-V I3D (Kinetics-400).

Reference distribution = the real source signer videos. FVD(model_outputs,
sources) measures how "natural human-signing video" each model's output is;
spatiotemporal I3D features are sensitive to the artifacts DWPose ignored
(text overlays, blob hands, background leakage, temporal flicker). The avatar-
vs-source appearance gap is a common offset to both models, so the *difference*
FVD_mimic - FVD_dispose reflects quality/coherence, not identity.

Lower FVD = better.
"""
import numpy as np

import torch          # torch before decord (OpenMP init order)
import decord

I3D_CLIP = 16
I3D_RES = 224


def load_i3d(path, device):
    return torch.jit.load(path, map_location=device).eval().to(device)


def _preprocess_clip(frames_rgb):
    """frames_rgb: [T,H,W,3] uint8 RGB -> tensor [1,3,T,224,224] in [-1,1].

    Centre-crops to a square first so that non-square sources (640x360 signer
    clips) are not aspect-distorted relative to the square generated frames."""
    import torch.nn.functional as F
    x = torch.from_numpy(frames_rgb).float()            # [T,H,W,3]
    x = x.permute(0, 3, 1, 2)                           # [T,3,H,W]
    h, w = x.shape[2], x.shape[3]
    s = min(h, w)                                       # centre square crop
    x = x[:, :, (h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    x = F.interpolate(x, size=(I3D_RES, I3D_RES),
                      mode="bilinear", align_corners=False)  # [T,3,224,224]
    x = x / 127.5 - 1.0
    x = x.permute(1, 0, 2, 3)[None]                     # [1,3,T,224,224]
    return x


@torch.no_grad()
def video_features(video_path, i3d, device, clip_len=I3D_CLIP, stride=None):
    """Non-overlapping (or strided) clips -> [n_clips, 400] I3D features."""
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    n = len(vr)
    stride = stride or clip_len
    starts = list(range(0, max(1, n - clip_len + 1), stride))
    if not starts:
        starts = [0]
    feats = []
    for s in starts:
        idx = list(range(s, s + clip_len))
        if idx[-1] >= n:                      # pad short tail by repeating last frame
            idx = [min(i, n - 1) for i in idx]
        frames = vr.get_batch(idx).asnumpy()  # [T,H,W,3] RGB
        x = _preprocess_clip(frames).to(device)
        f = i3d(x, rescale=False, resize=False, return_features=True)
        feats.append(f.cpu().numpy().reshape(-1))
    return np.stack(feats)  # [n_clips, 400]


def frechet_distance(feat_a, feat_b, eps=1e-6):
    from scipy.linalg import sqrtm
    mu_a, mu_b = feat_a.mean(0), feat_b.mean(0)
    ca = np.cov(feat_a, rowvar=False)
    cb = np.cov(feat_b, rowvar=False)
    # stabilize sqrtm for near-singular covariance (e.g. small subsets where
    # n_clips < feature dim), matching the standard pytorch-fid recipe.
    offset = np.eye(ca.shape[0]) * eps
    covmean = sqrtm((ca + offset) @ (cb + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(np.sum((mu_a - mu_b) ** 2) + np.trace(ca + cb - 2 * covmean))
