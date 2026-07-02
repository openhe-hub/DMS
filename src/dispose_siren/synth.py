"""Synthetic 2D keypoint-trajectory distribution with ANALYTIC ground-truth velocity.

Smooth band-limited motion (sum of K sinusoids per axis), random amplitude /
frequency / phase / center. This is the prior we train the amortized INR on:
"human keypoint motion over a ~16-frame window is smooth and band-limited".

Returned positions are in an absolute pixel-like frame; train.py applies the
per-trajectory z-score (mean/std over the observed frames) so the learned prior
is SCALE-INVARIANT and transfers to real DWPose trajectories of any keypoint.
"""
import numpy as np
from . import N_FRAMES, DENSE_T

TF = np.linspace(0, 1, N_FRAMES)     # observed (sparse) frame times
TG = np.linspace(0, 1, DENSE_T)      # dense evaluation grid


def sample_traj(rng, n, K=3, amp=(20, 90), freq=(0.5, 3.0), center=(180, 330)):
    """Sample n trajectories.

    Returns:
        cp : (n, DENSE_T, 2)  clean dense position
        cv : (n, DENSE_T, 2)  GT velocity, per-FRAME (d pos / d frame)
        cf : (n, N_FRAMES, 2) clean position at the sparse observed frames
    """
    A = rng.uniform(*amp, (n, 2, K))
    f = rng.uniform(*freq, (n, 2, K))
    ph = rng.uniform(0, 2 * np.pi, (n, 2, K))
    c = rng.uniform(*center, (n, 2))

    def pos(t):  # (T,) -> (n,T,2)
        out = np.zeros((n, len(t), 2))
        for d in range(2):
            s = np.zeros((n, len(t)))
            for k in range(K):
                s += A[:, d, k:k + 1] * np.sin(2 * np.pi * f[:, d, k:k + 1] * t[None] + ph[:, d, k:k + 1])
            out[:, :, d] = c[:, d:d + 1] + s
        return out

    def vel(t):  # d pos / d frame
        out = np.zeros((n, len(t), 2))
        for d in range(2):
            s = np.zeros((n, len(t)))
            for k in range(K):
                s += A[:, d, k:k + 1] * 2 * np.pi * f[:, d, k:k + 1] * \
                     np.cos(2 * np.pi * f[:, d, k:k + 1] * t[None] + ph[:, d, k:k + 1])
            out[:, :, d] = s / (N_FRAMES - 1)
        return out

    return pos(TG), vel(TG), pos(TF)
