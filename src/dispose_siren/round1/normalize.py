"""Per-trajectory z-score normalization -- the bridge that makes the synthetic
amortized prior SCALE-INVARIANT, so it transfers to real DWPose keypoints whose
absolute position / amplitude differ per keypoint and per video.

At both train and test time we normalize using ONLY the observed (noisy) frames'
per-axis mean and std -- never any clean/GT info -- so the procedure is identical
in both regimes. Caveat (reported honestly): at high noise the std is inflated by
the noise, mildly under-scaling the recovered velocity.
"""
import numpy as np
import torch


def zscore_stats(noisy):  # (B,n,2) -> mu (B,1,2), s (B,1,2)
    mu = noisy.mean(axis=1, keepdims=True)
    s = noisy.std(axis=1, keepdims=True) + 1e-6
    return mu, s


@torch.no_grad()
def infer(model, noisy_px, tg, device="cpu"):
    """Run the learned INR on raw-pixel observed keypoints.

    noisy_px : (B, n_obs, 2) raw pixel coords of the observed frames
    tg       : (T,) dense eval grid in [0,1]
    Returns dense position (B,T,2) and per-frame velocity (B,T,2) in raw pixels.
    """
    mu, s = zscore_stats(noisy_px)
    norm = (noisy_px - mu) / s
    nt = torch.tensor(norm, dtype=torch.float32, device=device)
    film = model.encode(nt)
    tgt = torch.tensor(tg, dtype=torch.float32, device=device)
    pos_n = model.decode(film, tgt).cpu().numpy()        # (B,T,2)
    # velocity needs grad -> compute outside no_grad
    pos = pos_n * s + mu
    return pos, film, mu, s


def velocity_px(model, film, mu, s, teval, device="cpu"):
    """Denormalized per-frame velocity at teval (needs autograd -> not no_grad)."""
    from .models import velocity
    tt = torch.tensor(teval, dtype=torch.float32, device=device)
    V = velocity(model, film, tt).detach().cpu().numpy()  # (B,T,2) normalized
    return V * s                                          # raw px/frame


def decode_pos_px(model, film, mu, s, teval, device="cpu"):
    with torch.no_grad():
        tt = torch.tensor(teval, dtype=torch.float32, device=device)
        pos_n = model.decode(film, tt).cpu().numpy()
    return pos_n * s + mu
