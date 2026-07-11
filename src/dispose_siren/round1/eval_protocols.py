"""Honest real-data evaluation. No GT velocity exists for real video, so:

Protocol A -- held-out frame reconstruction (NEUTRAL):
  From `span` real detections take 2N evenly-spaced ones, split even/odd.
  Each method reconstructs POSITION at the held-out (odd) times from the
  observed (even) ones; score = MSE vs the real held-out detections (px^2).
  Both methods pay the same detection-noise floor, so lower => better captures
  the true smooth motion.

Protocol B -- high-fps finite-diff pseudo-GT velocity (FAVORS fd-like methods):
  Use all `span` (denser) detections, lightly smoothed, finite-diff'd as a
  pseudo-GT velocity; observed = a sub-sampled N-frame DisPose window. Score =
  velocity MSE vs pseudo-GT. CAVEAT: pseudo-GT is itself fd of noisy detections,
  structurally favouring fd/fd+Gaussian -- so this is the harder, less neutral
  test for the learned model; Protocol A is the primary verdict.

All velocities are expressed per unit-tau (displacement scale across the window)
so methods on the N-frame grid and pseudo-GT on the span grid are comparable.
"""
import numpy as np
from .. import N_FRAMES, DENSE_T
from .synth import TG
from .. import baselines as B
from . import normalize as Z

VEL_GRID = (0.6, 0.9, 1.2, 1.6, 2.2, 3.0)


def estimate_jitter(windows, sig=1.5):
    """Diagnostic: median high-frequency residual after light smoothing, and the
    relative jitter (residual std / motion amplitude std) per axis -- tells us
    which noise regime the real trajectories live in."""
    if len(windows) == 0:
        return {"abs_px": 0.0, "relative": 0.0}
    sm = B._smooth(windows, sig)
    resid = (windows - sm).reshape(len(windows), -1).std(axis=1)        # per-window abs jitter
    amp = windows.reshape(len(windows), windows.shape[1], 2).std(axis=1).mean(axis=1)
    return {"abs_px": float(np.median(resid)),
            "relative": float(np.median(resid / (amp + 1e-6)))}


def make_windows(points, vis, span=48, step=24):
    """-> dense windows (S, span, 2) for keypoints fully visible across the window."""
    K, T, _ = points.shape
    out = []
    for k in range(K):
        for s0 in range(0, T - span + 1, step):
            sl = slice(s0, s0 + span)
            if vis[k, sl].all():
                out.append(points[k, sl])
    return np.asarray(out, dtype=np.float64) if out else np.zeros((0, span, 2))


# ---------------------------------------------------------------- Protocol A
def protocol_holdout(windows, model, device="cpu", n=N_FRAMES, obs_noise=0.0, seed=0):
    span = windows.shape[1]
    idx = np.linspace(0, span - 1, 2 * n).astype(int)
    obs_i, hold_i = idx[0::2], idx[1::2]
    tau = idx / (span - 1)
    tf, th = tau[0::2], tau[1::2]
    obs = windows[:, obs_i].copy()          # (S,n,2) observed real detections
    gt = windows[:, hold_i]                 # (S,n,2) held-out real detections (target)
    if obs_noise > 0:                       # corrupt ONLY the observations
        obs = obs + np.random.RandomState(seed).randn(*obs.shape) * obs_noise

    lin = B.linint_pos(obs, tf, th)
    gpe, gsig, gpos = B.best_sigma_gauss_pos(obs, tf, th, gt)  # baseline best-case
    _, film, mu, s = Z.infer(model, obs, TG, device)
    le = Z.decode_pos_px(model, film, mu, s, th, device)

    def mse(p): return np.mean((p - gt) ** 2)
    return {"linint": mse(lin), "gauss+lin": mse(gpos), "learned": mse(le),
            "gauss_sigma": gsig, "S": len(windows)}


# ---------------------------------------------------------------- Protocol B
def _pseudo_gt_vel(windows, dense_sigma=1.0):
    """Per-unit-tau velocity from the dense window (lightly smoothed fd)."""
    span = windows.shape[1]
    sm = B._smooth(windows, dense_sigma)
    fd = np.diff(sm, axis=1) * (span - 1)               # d pos / d tau
    tau_mid = (np.linspace(0, 1, span)[:-1] + np.linspace(0, 1, span)[1:]) / 2
    S = len(windows)
    return np.stack([[np.interp(TG, tau_mid, fd[i, :, c]) for c in range(2)]
                     for i in range(S)], 0).transpose(0, 2, 1)


def protocol_pseudogt(windows, model, device="cpu", n=N_FRAMES, obs_noise=0.0, seed=0):
    span = windows.shape[1]
    gt = _pseudo_gt_vel(windows)                         # (S,T,2) per-tau, from clean-ish dense
    obs_i = np.linspace(0, span - 1, n).astype(int)
    tf = obs_i / (span - 1)
    obs = windows[:, obs_i].copy()
    if obs_noise > 0:                                    # corrupt ONLY the observations
        obs = obs + np.random.RandomState(seed).randn(*obs.shape) * obs_noise

    fd = B.fd_dense(obs, tf, TG) * (n - 1)               # per-tau
    fde, fsig, fdg = B.best_sigma_fdg(obs, tf, TG, gt / (n - 1))  # tune in same units
    fdg = fdg * (n - 1)
    _, film, mu, s = Z.infer(model, obs, TG, device)
    le = Z.velocity_px(model, film, mu, s, TG, device) * (n - 1)

    def mse(v): return np.mean((v - gt) ** 2)
    return {"fd": mse(fd), "fd+gauss": mse(fdg), "learned": mse(le),
            "fdg_sigma": fsig, "S": len(windows)}
