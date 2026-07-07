"""Evaluation protocols + baselines for the hand-trajectory pilot.

Protocol HOLDOUT (neutral, real-GT): even/odd split of the 32 real detections;
methods see the even frames, are scored on the odd ones -- but ONLY where the
held-out detection is itself trustworthy (conf >= thr), because low-conf
detections are garbage, not ground truth.

Protocol GAP-INPAINT (the actual use case, honest variant): on windows whose
frames are ALL high-confidence, mask a contiguous gap with length drawn from
Gate B's measured real gap-length histogram; all methods see the same
remaining observations; score on the masked (trustworthy) frames, stratified
by gap length. Real low-conf segments have no GT and are NOT scored here.

Baselines get best-case treatment (Gaussian sigma tuned on the eval data with
the same masked MSE the methods are scored with).
"""
import numpy as np
import torch

from .baselines import linint_pos, _smooth
from .hand_train import prepare_windows

GAUSS_GRID = (0.6, 0.9, 1.2, 1.6, 2.2, 3.0)


def subset_windows(W, mask):
    return {k: np.asarray(W[k])[mask] for k in W}


# ------------------------------------------------- non-uniform cubic spline
def natural_cubic_nonuniform(obs, tf, teval):
    """Natural cubic spline with arbitrary knot positions, batched.

    obs (B,n,C), tf (n,) increasing, teval (E,) inside [tf[0], tf[-1]]
    -> (B,E,C). Needed because gap-inpainting observations are not equally
    spaced (interp.natural_cubic_pos assumes unit knots).
    """
    B, n, C = obs.shape
    if n < 3:
        return linint_pos(obs.reshape(B, n, -1), tf, teval).reshape(B, len(teval), C)
    h = np.diff(tf)                                               # (n-1,)
    A = np.zeros((n - 2, n - 2))
    np.fill_diagonal(A, 2 * (h[:-1] + h[1:]))
    np.fill_diagonal(A[1:], h[1:-1])
    np.fill_diagonal(A[:, 1:], h[1:-1])
    d = (obs[:, 2:] - obs[:, 1:-1]) / h[1:, None] \
        - (obs[:, 1:-1] - obs[:, :-2]) / h[:-1, None]             # (B,n-2,C)
    rhs = 6.0 * d.transpose(1, 0, 2).reshape(n - 2, -1)
    M = np.zeros((n, B * C))
    M[1:-1] = np.linalg.solve(A, rhs)
    M = M.reshape(n, B, C).transpose(1, 0, 2)                     # (B,n,C)

    i = np.clip(np.searchsorted(tf, teval, side="right") - 1, 0, n - 2)
    hi = h[i][None, :, None]
    xl = (teval - tf[i])[None, :, None]
    xr = (tf[i + 1] - teval)[None, :, None]
    y0, y1, m0, m1 = obs[:, i], obs[:, i + 1], M[:, i], M[:, i + 1]
    return (m0 * xr ** 3 + m1 * xl ** 3) / (6 * hi) \
        + (y0 / hi - m0 * hi / 6) * xr + (y1 / hi - m1 * hi / 6) * xl


# ------------------------------------------------------------- baselines
def _masked_mse(pred, gt, w):
    return float((((pred - gt) ** 2) * w).sum() / w.sum())


def hand_baselines(obs, tf, teval, gt, w):
    """obs (B,n,21,2) -> dict of reconstructions (B,E,21,2).

    The Gaussian baseline tunes its sigma against gt with the SAME masked MSE
    used for scoring (best-case treatment, as in baselines.best_sigma_*).
    """
    B, n = obs.shape[:2]
    E = len(teval)
    o2 = obs.transpose(0, 2, 1, 3).reshape(B * 21, n, 2)
    back = lambda a: a.reshape(B, 21, E, 2).transpose(0, 2, 1, 3)
    out = {"linear": back(linint_pos(o2, tf, teval)),
           "spline": back(natural_cubic_nonuniform(o2, tf, teval))}
    best = None
    for sg in GAUSS_GRID:
        g = back(linint_pos(_smooth(o2, sg), tf, teval))
        e = _masked_mse(g, gt, w)
        if best is None or e < best[0]:
            best = (e, sg, g)
    out["gauss+lin"] = best[2]
    out["gauss_sigma"] = best[1]
    return out


# ------------------------------------------------------------- model recon
def model_recon(model, D, sel, obs_i, tau_eval, device="cpu"):
    """Reconstruct canonical positions at tau_eval for window subset `sel`.

    D = prepare_windows(...) dict, sel = index array, obs_i = observed frame
    indices. Returns (B,E,21,2) numpy (canonical units).
    """
    span = D["traj_n"].shape[1]
    oit = torch.tensor(obs_i, dtype=torch.long, device=device)
    tau_o = (oit.float() / (span - 1)).unsqueeze(0).expand(len(sel), -1)
    te = torch.tensor(tau_eval, dtype=torch.float32, device=device)
    idx = torch.tensor(sel, dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        mod = model.encode(D["traj_n"][idx][:, oit], D["conf"][idx][:, oit],
                           D["wrist"][idx][:, oit], D["elbow"][idx][:, oit],
                           D["log_scale"][idx], D["side"][idx], tau_o)
        pos = model.decode(mod, te)                               # (B,E,42)
    return pos.cpu().numpy().reshape(len(sel), len(tau_eval), 21, 2)


def _px_scale(W_dict, sel):
    """(B,1,1,2) canonical->px factor: scale_norm * (W,H) per window."""
    from .hand_traj import hand_canon
    _, _, _, _, s4 = hand_canon(W_dict["traj"][sel], W_dict["conf"][sel])
    wh = np.stack([W_dict["W"][sel], W_dict["H"][sel]], -1).astype(float)
    return s4[:, :, :, 0, None] * wh[:, None, None, :]            # (B,1,1,2)


# --------------------------------------------------------------- protocols
def protocol_hand_holdout(W, model, conf_thr=0.3, device="cpu", D=None):
    """Even/odd frame split; masked MSE in canonical units and px^2."""
    D = D or prepare_windows(W, device)
    S, span = W["traj"].shape[:2]
    idx = np.arange(span)
    obs_i, hold_i = idx[0::2], idx[1::2]
    tf, th = obs_i / (span - 1), hold_i / (span - 1)

    tn = D["traj_n_np"]
    obs, gt = tn[:, obs_i], tn[:, hold_i]
    w = (W["conf"][:, hold_i] >= conf_thr).astype(float)[..., None]
    sel = np.arange(S)

    recons = hand_baselines(obs, tf, th, gt=gt, w=w)
    recons["learned"] = model_recon(model, D, sel, obs_i, th, device)

    pxf = _px_scale(W, sel)
    out = {"S": S, "held_frames_scored": float(w.sum())}
    for k in ("linear", "spline", "gauss+lin", "learned"):
        out[k] = _masked_mse(recons[k], gt, w)
        out[k + "_px"] = _masked_mse(recons[k] * pxf, gt * pxf, w)
    out["gauss_sigma"] = recons["gauss_sigma"]
    return out


def clean_window_mask(W, conf_thr=0.3):
    """Windows where EVERY keypoint of EVERY frame is confident -- the only
    windows where a synthetic gap has trustworthy GT everywhere."""
    return (W["conf"] >= conf_thr).all(axis=(1, 2))


def protocol_gap_inpaint(W, model, gap_lengths, conf_thr=0.3, n_patterns=8,
                         max_gap=12, seed=0, device="cpu"):
    """Synthetic-honest inpainting on all-high-conf windows.

    gap_lengths: list of real gap lengths (Gate B). Windows are split across
    n_patterns sampled (start, length) patterns; each pattern's observations =
    all frames except the gap. Scores stratified by gap length.
    """
    rng = np.random.RandomState(seed)
    mask = clean_window_mask(W, conf_thr)
    if mask.sum() < n_patterns:
        return {"S": int(mask.sum()), "skipped": "too few clean windows"}
    Wc = subset_windows(W, mask)
    D = prepare_windows(Wc, device)
    S, span = Wc["traj"].shape[:2]
    tn = D["traj_n_np"]
    pxf = _px_scale(Wc, np.arange(S))

    gl = np.asarray([g for g in gap_lengths if 2 <= g <= max_gap])
    if len(gl) == 0:
        gl = np.array([2, 3, 4, 6, 8])
    order = rng.permutation(S)
    groups = np.array_split(order, n_patterns)

    acc = {}
    for grp in groups:
        if len(grp) == 0:
            continue
        L = int(gl[rng.randint(len(gl))])
        s0 = rng.randint(1, span - 1 - L)
        keep = np.ones(span, bool)
        keep[s0:s0 + L] = False
        obs_i, gap_i = np.where(keep)[0], np.where(~keep)[0]
        tf, tg = obs_i / (span - 1), gap_i / (span - 1)

        obs, gt = tn[grp][:, obs_i], tn[grp][:, gap_i]
        w = np.ones_like(gt[..., :1])
        recons = hand_baselines(obs, tf, tg, gt=gt, w=w)
        recons["learned"] = model_recon(model, D, grp, obs_i, tg, device)
        b = acc.setdefault(L, {k: [0.0, 0.0, 0] for k in
                                ("linear", "spline", "gauss+lin", "learned")})
        pf = pxf[grp]
        for k in b:
            b[k][0] += (((recons[k] - gt) ** 2) * w).sum()
            b[k][1] += ((((recons[k] - gt) * pf) ** 2) * w).sum()
            b[k][2] += w.sum() * 2  # xy
    out = {"S": S, "by_gap_len": {}}
    for L in sorted(acc):
        out["by_gap_len"][L] = {
            k: dict(mse=v[0] / (v[2] / 2), mse_px=v[1] / (v[2] / 2))
            for k, v in acc[L].items()}
    return out
