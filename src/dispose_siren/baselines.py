"""Baselines.

Velocity (motion-field) reconstructors -- what DisPose effectively consumes:
  - fd_dense   : finite difference of the raw observed keypoints
  - fdg_dense  : Gaussian-smoothed positions then finite difference (DisPose-style)

Position reconstructors -- for the held-out-frame protocol (fair position analogues):
  - linint_pos : linear interpolation between observed frames
  - gauss_pos  : Gaussian-smoothed observed frames then linear interpolation

All operate on a batch (B, n_obs, 2) of observed keypoints and a dense/eval
time grid in [0,1]. tf is the observed (sparse) time grid.
"""
import numpy as np


def _gauss_kernel(sig, r=5):
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
    return k / k.sum()


def _smooth(noisy, sig):  # (B,n,2) edge-padded gaussian smoothing along frames
    k = _gauss_kernel(sig)
    p = len(k) // 2
    B, n, _ = noisy.shape
    out = np.empty_like(noisy)
    for i in range(B):
        for c in range(2):
            out[i, :, c] = np.convolve(np.pad(noisy[i, :, c], p, 'edge'), k, 'valid')
    return out


def fd_dense(noisy, tf, tg):                       # (B,n,2)->(B,T,2) per-frame vel
    fd = np.diff(noisy, axis=1)
    tm = (tf[:-1] + tf[1:]) / 2
    B = len(noisy)
    return np.stack([[np.interp(tg, tm, fd[i, :, c]) for c in range(2)] for i in range(B)], 0).transpose(0, 2, 1)


def fdg_dense(noisy, tf, tg, sig):
    return fd_dense(_smooth(noisy, sig), tf, tg)


def linint_pos(noisy, tf, teval):                  # (B,n,2)->(B,len(teval),2)
    B = len(noisy)
    return np.stack([[np.interp(teval, tf, noisy[i, :, c]) for c in range(2)] for i in range(B)], 0).transpose(0, 2, 1)


def gauss_pos(noisy, tf, teval, sig):
    return linint_pos(_smooth(noisy, sig), tf, teval)


def best_sigma_fdg(noisy, tf, tg, gt_vel, grid=(0.6, 0.9, 1.2, 1.6, 2.2, 3.0)):
    """Strongest fd+Gaussian baseline: pick the smoothing sigma minimising
    velocity MSE on THIS data (gives the baseline its best-case advantage)."""
    best = None
    for sg in grid:
        v = fdg_dense(noisy, tf, tg, sg)
        e = np.mean((v - gt_vel) ** 2)
        if best is None or e < best[0]:
            best = (e, sg, v)
    return best  # (mse, sigma, vel)


def best_sigma_gauss_pos(noisy, tf, teval, gt_pos, grid=(0.6, 0.9, 1.2, 1.6, 2.2, 3.0)):
    """Strongest Gaussian+linear position baseline for the held-out protocol."""
    best = None
    for sg in grid:
        p = gauss_pos(noisy, tf, teval, sg)
        e = np.mean((p - gt_pos) ** 2)
        if best is None or e < best[0]:
            best = (e, sg, p)
    return best  # (mse, sigma, pos)
