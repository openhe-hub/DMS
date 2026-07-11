"""Amortized continuous keypoint representation: an encoder produces FiLM
modulation (gamma, beta) of a shared SIREN decoder Phi(tau; z)
(NIAF / PA-HiRes-style hypernetwork modulation).

The decoder is continuous in tau, so velocity is the EXACT analytic derivative
d Phi / d tau (autograd), not a finite difference. Trained across many
trajectories it learns to *denoise onto the motion manifold* rather than
interpolate noise (which is why per-clip test-time SIREN fitting fails).
"""
import numpy as np
import torch
import torch.nn as nn
from .. import N_FRAMES


class FiLMSIREN(nn.Module):
    def __init__(self, H=64, L=3, w0=15., zdim=96, n_obs=N_FRAMES):
        super().__init__()
        self.H, self.L, self.w0, self.n_obs = H, L, w0, n_obs
        self.W = nn.ParameterList()
        self.b = nn.ParameterList()
        din = 1
        for i in range(L):
            w = nn.Parameter(torch.empty(H, din))
            with torch.no_grad():
                if i == 0:
                    w.uniform_(-1 / din, 1 / din)
                else:
                    bb = np.sqrt(6 / din) / w0
                    w.uniform_(-bb, bb)
            self.W.append(w)
            self.b.append(nn.Parameter(torch.zeros(H)))
            din = H
        self.out = nn.Linear(H, 2)
        self.enc = nn.Sequential(
            nn.Linear(2 * n_obs, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, zdim),
        )
        self.to_film = nn.Linear(zdim, L * H * 2)

    def encode(self, noisy):  # (B, n_obs, 2) -> film (B, L, H, 2)
        z = self.enc(noisy.reshape(noisy.shape[0], -1))
        return self.to_film(z).reshape(-1, self.L, self.H, 2)

    def decode(self, film, tau):  # film:(B,L,H,2), tau:(T,) or (B,T) -> pos (B,T,2)
        B = film.shape[0]
        if tau.dim() == 1:
            tau = tau.unsqueeze(0).expand(B, -1)
        u = (2 * tau - 1).unsqueeze(-1)            # (B,T,1) in [-1,1]
        h = u
        for i in range(self.L):
            g = film[:, i, :, 0].unsqueeze(1)       # (B,1,H)
            be = film[:, i, :, 1].unsqueeze(1)
            lin = torch.einsum('btd,hd->bth', h, self.W[i]) + self.b[i]
            h = torch.sin(self.w0 * (lin * (1 + g) + be))
        return self.out(h)


def velocity(model, film, tau):
    """Per-FRAME velocity via exact autograd derivative of the decoder.

    Uses a per-sample tau leaf so the gradient is per-trajectory: pos[b,j]
    depends only on tau[b,j]. (Bug history: a shared tau leaf made autograd
    sum derivatives over the batch.)
    """
    B = film.shape[0]
    if tau.dim() == 1:
        tau = tau.unsqueeze(0).expand(B, -1)
    t = tau.clone().requires_grad_(True)           # (B,T)
    pos = model.decode(film, t)                     # (B,T,2)
    v = []
    for c in range(2):
        g = torch.autograd.grad(pos[:, :, c].sum(), t, create_graph=True, retain_graph=True)[0]
        v.append(g)
    V = torch.stack(v, -1) / (N_FRAMES - 1)         # (B,T,2) per-frame velocity
    return V
