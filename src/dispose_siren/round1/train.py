"""Train the amortized, scale-invariant FiLM-SIREN on the synthetic prior.

Each trajectory is z-score normalized using its NOISY observed frames (identical
to test time). The model maps normalized-noisy-observed -> normalized clean
position + per-frame velocity (NIAF recipe: L_pos + L_vel). This learns a
scale-free smooth-motion prior, not an absolute-coordinate fit.

Mixed-noise training (sigmas drawn per batch) so a SINGLE checkpoint transfers
across the unknown real DWPose jitter level.
"""
import numpy as np
import torch
from .. import N_FRAMES, DENSE_T
from .synth import sample_traj, TF, TG
from .models import FiLMSIREN, velocity
from .normalize import zscore_stats


def train_amortized(epochs=400, ntr=2000, sigmas=(2, 4, 8, 14, 20), lr=1e-3,
                    vel_w=5.0, device="cpu", seed=0, log=True):
    rng = np.random.RandomState(seed)
    cp, cv, cf = sample_traj(rng, ntr)             # clean dense pos, gt vel, clean@frames
    cp_t = torch.tensor(cp, dtype=torch.float32, device=device)
    cv_t = torch.tensor(cv, dtype=torch.float32, device=device)
    cf_t = torch.tensor(cf, dtype=torch.float32, device=device)

    m = FiLMSIREN().to(device)
    opt = torch.optim.Adam(m.parameters(), lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    TGt = torch.tensor(TG, dtype=torch.float32, device=device)
    vi = np.linspace(0, DENSE_T - 1, 64).astype(int)
    TGv = TGt[vi]
    bs = 256

    for ep in range(epochs):
        perm = torch.randperm(ntr); lp = lv = 0.0; nb = 0
        for i in range(0, ntr, bs):
            idx = perm[i:i + bs]
            B = len(idx)
            sigma = float(rng.choice(sigmas))
            noisy = cf_t[idx] + torch.randn(B, N_FRAMES, 2, device=device) * sigma
            # per-trajectory z-score from the NOISY observed frames (test-identical)
            mu = noisy.mean(dim=1, keepdim=True)
            s = noisy.std(dim=1, keepdim=True) + 1e-6
            noisy_n = (noisy - mu) / s
            cp_n = (cp_t[idx] - mu) / s
            cv_n = cv_t[idx] / s

            film = m.encode(noisy_n)
            pos_g = m.decode(film, TGt)
            loss_pos = ((pos_g - cp_n) ** 2).mean()
            V = velocity(m, film, TGv)
            loss_vel = ((V - cv_n[:, vi]) ** 2).mean()
            loss = loss_pos + vel_w * loss_vel
            opt.zero_grad(); loss.backward(); opt.step()
            lp += loss_pos.item(); lv += loss_vel.item(); nb += 1
        sch.step()
        if log and (ep % 25 == 0 or ep == epochs - 1):
            print(f"   ep{ep:3d} loss_pos={lp/nb:.4f} loss_vel={lv/nb:.4f}", flush=True)
    return m


def save_ckpt(model, path, meta=None):
    torch.save({"state_dict": model.state_dict(),
                "config": {"H": model.H, "L": model.L, "w0": model.w0,
                           "n_obs": model.n_obs},
                "meta": meta or {}}, path)


def load_ckpt(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    m = FiLMSIREN(**ck["config"]).to(device)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck.get("meta", {})
