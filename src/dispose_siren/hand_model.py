"""NIAF-style amortized hand-trajectory model: HandSetSIREN.

A transformer encoder over the observed window (all 21 hand keypoints jointly,
with confidences and wrist/elbow context) replaces NIAF's MLLM as the
"hierarchical spectral modulator" (arXiv 2603.01766 sec 3.1): learnable query
embeddings, Q = L*(G+1) grouped per SIREN layer, cross-attend to the encoded
context and are projected to per-layer modulation (gamma, beta) that
reconfigure shared SIREN meta-parameters:

    pre_l = (W_l h) * (1 + gamma_l) + (b_l + beta_l)      # NIAF eq. (7)
    h     = sin(w0 * pre_l)                                # eq. (6)

Train-once / single forward at inference; no MLLM, no per-clip fitting.
Output is the full 42-dim hand (21 kp x 2, canonical frame) as one function
A(tau), so hand-shape structure is modeled jointly -- unlike the per-keypoint
FiLMSIREN in models.py, whose independence assumption could not carry a
hand-manifold prior. Velocity is the exact autograd derivative dA/dtau
(per-sample tau leaf; see models.velocity for the bug history).
"""
import numpy as np
import torch
import torch.nn as nn

HAND_N = 21
OUT_DIM = HAND_N * 2
MAX_OBS = 64


class HandSetSIREN(nn.Module):
    """cfg: d_model, n_enc_layers, n_head, ff, H (siren width), L (siren
    layers), G (gamma groups/layer), w0, dropout."""

    def __init__(self, d_model=128, n_enc_layers=3, n_head=4, ff=256,
                 H=128, L=4, G=2, w0=15.0, dropout=0.1):
        super().__init__()
        assert H % G == 0
        self.cfg = dict(d_model=d_model, n_enc_layers=n_enc_layers,
                        n_head=n_head, ff=ff, H=H, L=L, G=G, w0=w0,
                        dropout=dropout)
        self.H, self.L, self.G, self.w0 = H, L, G, w0

        # ---- context encoder (the spectral modulator's backbone)
        # token: 21*(x,y,conf)=63 + wrist 2 + elbow 2 + log_scale 1 + side 1
        #        + tau 1 = 70
        self.tok_in = nn.Linear(70, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(MAX_OBS, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_head, dim_feedforward=ff, dropout=dropout,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_enc_layers)

        # ---- grouped queries: per SIREN layer, G weight-queries + 1 bias-query
        self.n_queries = L * (G + 1)
        self.queries = nn.Parameter(torch.randn(self.n_queries, d_model) * 0.02)
        self.cross = nn.TransformerDecoderLayer(
            d_model, n_head, dim_feedforward=ff, dropout=dropout,
            batch_first=True, norm_first=True)

        # ---- projection MLPs psi (zero-init last layer => identity modulation
        # at start: the shared meta-prior trains first, adaptation grows in)
        def proj(out_dim):
            m = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                              nn.Linear(d_model, out_dim))
            nn.init.zeros_(m[-1].weight)
            nn.init.zeros_(m[-1].bias)
            return m
        self.psi_gamma = nn.ModuleList(
            [proj(H // G) for _ in range(L * G)])
        self.psi_beta = nn.ModuleList([proj(H) for _ in range(L)])

        # ---- shared SIREN meta-parameters (standard SIREN init)
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
        self.out = nn.Linear(H, OUT_DIM)

    # -------------------------------------------------------------- encode
    def encode(self, obs_traj, obs_conf, wrist, elbow, log_scale, side,
               tau_obs, pad_mask=None):
        """obs_traj (B,n,21,2) canonical, obs_conf (B,n,21), wrist/elbow
        (B,n,2), log_scale (B,), side (B,) in {-1,+1}, tau_obs (B,n) in [0,1],
        pad_mask (B,n) True=PAD. -> modulation dict."""
        B, n = obs_traj.shape[:2]
        tok = torch.cat([
            torch.cat([obs_traj, obs_conf.unsqueeze(-1)], -1).reshape(B, n, -1),
            wrist, elbow,
            log_scale[:, None, None].expand(B, n, 1),
            side[:, None, None].expand(B, n, 1).to(obs_traj.dtype),
            tau_obs.unsqueeze(-1),
        ], dim=-1)                                                # (B,n,70)
        x = self.tok_in(tok) + self.pos_emb[:n][None]
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        q = self.queries[None].expand(B, -1, -1)
        z = self.cross(q, x, memory_key_padding_mask=pad_mask)    # (B,Q,d)

        gammas, betas = [], []
        for l in range(self.L):
            g = torch.cat([self.psi_gamma[l * self.G + j](z[:, l * (self.G + 1) + j])
                           for j in range(self.G)], dim=-1)       # (B,H)
            be = self.psi_beta[l](z[:, l * (self.G + 1) + self.G])  # (B,H)
            gammas.append(g)
            betas.append(be)
        return {"gamma": gammas, "beta": betas}

    # -------------------------------------------------------------- decode
    def decode(self, mod, tau, with_velocity=False):
        """mod from encode; tau (T,) or (B,T) in [0,1] -> pos (B,T,42).

        with_velocity=True also returns the EXACT analytic dA/dtau (per
        unit-tau) via the closed-form SIREN derivative recursion
        (d/dx sin = cos preserves the network structure -- NIAF sec 3.2):
            hdot_0 = 2                          (h_0 = 2*tau - 1)
            pre_l  = (W_l h)*(1+gamma) + b+beta
            hdot_l = cos(w0*pre_l) * w0 * (W_l hdot_{l-1})*(1+gamma)
            v      = W_out hdot_L
        Cheap, differentiable for training, and device-agnostic (no autograd
        double-backward / forward-mode needed).
        """
        B = mod["gamma"][0].shape[0]
        if tau.dim() == 1:
            tau = tau.unsqueeze(0).expand(B, -1)
        h = (2 * tau - 1).unsqueeze(-1)                           # (B,T,1)
        hdot = torch.full_like(h, 2.0) if with_velocity else None
        for l in range(self.L):
            g1 = (1 + mod["gamma"][l]).unsqueeze(1)               # (B,1,H)
            lin = torch.einsum("btd,hd->bth", h, self.W[l])
            pre = lin * g1 + (self.b[l] + mod["beta"][l]).unsqueeze(1)
            h = torch.sin(self.w0 * pre)
            if with_velocity:
                lind = torch.einsum("btd,hd->bth", hdot, self.W[l])
                hdot = torch.cos(self.w0 * pre) * self.w0 * lind * g1
        pos = self.out(h)
        if with_velocity:
            vel = torch.einsum("bth,oh->bto", hdot, self.out.weight)
            return pos, vel
        return pos

    def forward(self, tau, **enc_kwargs):
        return self.decode(self.encode(**enc_kwargs), tau)


def velocity(model, mod, tau):
    """Exact analytic dA/dtau (B,T,42); see decode(with_velocity=True)."""
    _, v = model.decode(mod, tau, with_velocity=True)
    return v
