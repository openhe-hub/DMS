"""Continuous-time INTERPOLATION of keypoint trajectories (Step 2 / R010).

Task shift vs step-1: the observations are (near-)clean real detections at a
coarse temporal stride; the job is to reconstruct the trajectory at the missing
intermediate times ("pose-control temporal super-resolution"), NOT to denoise.
Per-clip test-time fitting -- which lost the denoising task in Experiment A --
is a legitimate candidate here: smoothly passing through clean observations is
the desired behaviour, no learned prior required.

Contents:
  - natural_cubic_pos : pure-numpy natural cubic spline (no scipy on cluster)
  - PerClipSIREN      : one small SIREN per window, all windows fitted jointly
                        as a single batched optimisation (fast on CPU)
"""
import numpy as np
import torch


# ---------------------------------------------------------- cubic spline
def natural_cubic_pos(obs, x_eval, n=None):
    """Natural cubic spline through equally spaced knots, batched.

    obs    : (S, n, 2) observed positions at knot indices 0..n-1
    x_eval : (E,) evaluation coords in KNOT units (same grid for all windows)
    -> (S, E, 2)
    """
    S, n_, _ = obs.shape
    n = n or n_
    # second derivatives M: natural BC, unit knot spacing
    # M[i-1] + 4 M[i] + M[i+1] = 6 (y[i-1] - 2 y[i] + y[i+1]),  M[0]=M[n-1]=0
    A = np.zeros((n - 2, n - 2))
    np.fill_diagonal(A, 4.0)
    np.fill_diagonal(A[1:], 1.0)
    np.fill_diagonal(A[:, 1:], 1.0)
    rhs = 6.0 * (obs[:, :-2] + obs[:, 2:] - 2.0 * obs[:, 1:-1])   # (S, n-2, 2)
    Mi = np.linalg.solve(A, rhs.transpose(1, 0, 2).reshape(n - 2, -1))
    M = np.zeros((n, S * 2))
    M[1:-1] = Mi
    M = M.reshape(n, S, 2).transpose(1, 0, 2)                     # (S, n, 2)

    i = np.clip(np.floor(x_eval).astype(int), 0, n - 2)           # (E,)
    t = (x_eval - i)[None, :, None]                               # (1,E,1)
    y0, y1 = obs[:, i], obs[:, i + 1]
    m0, m1 = M[:, i], M[:, i + 1]
    return (y0 * (1 - t) + y1 * t
            + ((1 - t) ** 3 - (1 - t)) * m0 / 6.0
            + (t ** 3 - t) * m1 / 6.0)


# ---------------------------------------------------------- per-clip SIREN
class PerClipSIREN:
    """One independent SIREN per window, optimised jointly as a batch.

    Training-free in the paper sense: fitted at test time to the observed
    frames of the clip only; no cross-clip weights, no dataset. z-scored per
    window (same stats recipe as normalize.py -- observed frames only).
    """

    def __init__(self, S, H=64, L=3, w0=5.0, seed=0, device="cpu"):
        g = torch.Generator().manual_seed(seed)
        self.S, self.H, self.L, self.w0 = S, H, L, w0
        self.device = device
        self.W, self.b = [], []
        din = 1
        for i in range(L):
            bound = 1.0 / din if i == 0 else np.sqrt(6.0 / din) / w0
            W = (torch.rand(S, H, din, generator=g) * 2 - 1) * bound
            self.W.append(W.to(device).requires_grad_(True))
            self.b.append(torch.zeros(S, H, device=device, requires_grad=True))
            din = H
        bo = np.sqrt(6.0 / H) / w0
        self.Wo = ((torch.rand(S, 2, H, generator=g) * 2 - 1) * bo).to(device).requires_grad_(True)
        self.bo = torch.zeros(S, 2, device=device, requires_grad=True)

    def params(self):
        return self.W + self.b + [self.Wo, self.bo]

    def forward(self, tau):                    # tau (T,) in [0,1] -> (S,T,2)
        u = (2.0 * tau - 1.0).reshape(1, -1, 1).expand(self.S, -1, -1)
        h = u
        for i in range(self.L):
            lin = torch.einsum("std,shd->sth", h, self.W[i]) + self.b[i][:, None, :]
            h = torch.sin(self.w0 * lin)
        return torch.einsum("sth,soh->sto", h, self.Wo) + self.bo[:, None, :]

    def fit(self, obs_px, tf, steps=800, lr=3e-3, lam_smooth=0.0, dense=128,
            verbose=False):
        """obs_px (S, n, 2) raw pixels; tf (n,) observed taus in [0,1].
        Returns (mu, s) z-score stats for decoding back to pixels."""
        obs = torch.tensor(obs_px, dtype=torch.float32, device=self.device)
        mu = obs.mean(dim=1, keepdim=True)
        s = obs.std(dim=1, keepdim=True) + 1e-6
        target = (obs - mu) / s
        tft = torch.tensor(tf, dtype=torch.float32, device=self.device)
        tdense = torch.linspace(0, 1, dense, device=self.device)
        opt = torch.optim.Adam(self.params(), lr=lr)
        for it in range(steps):
            opt.zero_grad()
            pred = self.forward(tft)
            loss = ((pred - target) ** 2).mean()
            if lam_smooth > 0:
                pd = self.forward(tdense)
                d2 = pd[:, 2:] - 2 * pd[:, 1:-1] + pd[:, :-2]
                loss = loss + lam_smooth * (d2 ** 2).mean() * (dense - 1) ** 2
            loss.backward()
            opt.step()
            if verbose and (it % 200 == 0 or it == steps - 1):
                print(f"    perclip it={it} loss={float(loss):.5f}", flush=True)
        return mu, s

    @torch.no_grad()
    def decode_px(self, teval, mu, s):
        t = torch.tensor(teval, dtype=torch.float32, device=self.device)
        return (self.forward(t) * s + mu).cpu().numpy()


def perclip_fit_decode(obs_px, tf, teval, w0=5.0, lam=0.0, steps=800, seed=0,
                       device="cpu", verbose=False):
    m = PerClipSIREN(len(obs_px), w0=w0, seed=seed, device=device)
    mu, s = m.fit(obs_px, tf, steps=steps, lam_smooth=lam, verbose=verbose)
    return m.decode_px(teval, mu, s)
