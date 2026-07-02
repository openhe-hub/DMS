"""Self-supervised training of the FiLM-SIREN prior on REAL DWPose trajectories
-- the fix for the Step-1 domain gap (synthetic sinusoids != real motion).

Supervision signal (no clean GT exists for real video):
  - pseudo-clean target = the dense high-fps real trajectory, lightly smoothed
    to suppress per-frame detection noise (the "motion manifold").
  - input = the dense trajectory sub-sampled to the N-frame DisPose window,
    plus synthetic jitter augmentation (so the prior learns to denoise).
  - losses = L_pos (to pseudo-clean dense position) + L_vel (to its finite-diff
    velocity), both in the per-trajectory z-score space (scale-invariant).

This matches the prior to real motion statistics instead of synthetic sines.
"""
import numpy as np
import torch

from . import N_FRAMES
from .models import FiLMSIREN, velocity
from .baselines import _smooth


def real_train(train_windows, epochs=600, lr=1e-3, vel_w=0.5, weight_decay=1e-4,
               aug_noise=(0.0, 2.0, 4.0, 8.0), target_sigma=2.0,
               device="cpu", seed=0, log=True):
    # target_sigma>1 because the pseudo-GT velocity = diff(target)*(span-1)
    # amplifies detection noise; vel_w is small so this noisy term can't dominate.
    rng = np.random.RandomState(seed)
    W = np.asarray(train_windows, dtype=np.float64)          # (S, span, 2)
    S, span, _ = W.shape
    target = _smooth(W, target_sigma)                        # pseudo-clean dense
    obs_i = np.linspace(0, span - 1, N_FRAMES).astype(int)
    tau_dense = np.linspace(0, 1, span)
    tau_mid = (tau_dense[:-1] + tau_dense[1:]) / 2
    tgt_vel = np.diff(target, axis=1) * (span - 1)           # per-tau velocity

    tb = torch.tensor(target, dtype=torch.float32, device=device)
    vb = torch.tensor(tgt_vel, dtype=torch.float32, device=device)
    Wt = torch.tensor(W, dtype=torch.float32, device=device)
    tau_d = torch.tensor(tau_dense, dtype=torch.float32, device=device)
    tau_m = torch.tensor(tau_mid, dtype=torch.float32, device=device)
    obs_it = torch.tensor(obs_i, dtype=torch.long, device=device)

    m = FiLMSIREN().to(device)
    opt = torch.optim.Adam(m.parameters(), lr, weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bs = min(256, S)

    for ep in range(epochs):
        perm = torch.randperm(S, device=device); lp = lv = 0.0; nb = 0
        for i in range(0, S, bs):
            idx = perm[i:i + bs]
            sigma = float(rng.choice(aug_noise))
            obs = Wt[idx][:, obs_it]                          # real noisy observations
            if sigma > 0:
                obs = obs + torch.randn_like(obs) * sigma     # augmentation
            mu = obs.mean(dim=1, keepdim=True)
            s = obs.std(dim=1, keepdim=True) + 1e-6
            obs_n = (obs - mu) / s
            tb_n = (tb[idx] - mu) / s
            vb_n = vb[idx] / s

            film = m.encode(obs_n)
            pos = m.decode(film, tau_d)
            loss_pos = ((pos - tb_n) ** 2).mean()
            V = velocity(m, film, tau_m) * (N_FRAMES - 1)     # per-tau, normalized
            loss_vel = ((V - vb_n) ** 2).mean()
            loss = loss_pos + vel_w * loss_vel
            opt.zero_grad(); loss.backward(); opt.step()
            lp += loss_pos.item(); lv += loss_vel.item(); nb += 1
        sch.step()
        if log and (ep % 50 == 0 or ep == epochs - 1):
            print(f"   ep{ep:3d} loss_pos={lp/nb:.4f} loss_vel={lv/nb:.4f}", flush=True)
    return m
