"""Training of HandSetSIREN on real sign-language hand windows.

Modeled on real_train.py with three deliberate changes:
  1. samples are whole hand windows (42-dim joint), not independent keypoints;
  2. the pseudo-clean target is a CONFIDENCE-WEIGHTED Gaussian smoothing of the
     dense window (low-conf garbage cannot contaminate the target), and both
     losses are confidence-weighted;
  3. observation patterns mix uniform subsampling (with phase jitter) and
     contiguous-gap masking, so the model trains for both protocols
     (held-out interpolation and dropout inpainting).

All coordinates are in the window-canonical frame of hand_traj.hand_canon
(wrist-centred, median-bone scale), so noise-augmentation sigmas are in
hand-scale units -- set them from Gate B's measured relative jitter.
"""
import numpy as np
import torch

from .baselines import _gauss_kernel
from .hand_model import HandSetSIREN
from .hand_traj import hand_canon


def conf_smooth(traj, conf, sig, conf_floor=0.05):
    """Confidence-weighted Gaussian smoothing along frames (edge-padded).

    traj (S,span,21,2), conf (S,span,21) -> same shape. With conf == const it
    reduces exactly to baselines._smooth (same kernel, same edge padding).
    """
    k = _gauss_kernel(sig)
    p = len(k) // 2
    w = np.clip(conf, conf_floor, 1.0)[..., None]                 # (S,span,21,1)
    tp = np.pad(traj, ((0, 0), (p, p), (0, 0), (0, 0)), mode="edge")
    wp = np.pad(w, ((0, 0), (p, p), (0, 0), (0, 0)), mode="edge")
    num = np.zeros_like(traj)
    den = np.zeros_like(w)
    for j, kj in enumerate(k):
        sl = slice(j, j + traj.shape[1])
        num += kj * (tp[:, sl] * wp[:, sl])
        den += kj * wp[:, sl]
    return num / den


def obs_uniform(span, n_obs, rng, jitter=True):
    idx = np.round(np.linspace(0, span - 1, n_obs)).astype(int)
    if jitter:
        j = rng.randint(-1, 2, size=n_obs)
        j[0] = j[-1] = 0
        idx = np.clip(idx + j, 0, span - 1)
        idx = np.maximum.accumulate(idx)                          # keep sorted
    return np.unique(idx)


def obs_gap(span, rng, gap_lens=(2, 3, 4, 5, 6, 8)):
    L = int(rng.choice(gap_lens))
    s = rng.randint(1, span - 1 - L)
    keep = np.ones(span, bool)
    keep[s:s + L] = False
    return np.where(keep)[0]


def prepare_windows(W, device="cpu"):
    """32_build_windows dict -> canonical torch tensors + numpy targets base."""
    traj_n, wrist_n, elbow_n, mu, sc = hand_canon(W["traj"], W["conf"],
                                                  W["wrist"], W["elbow"])
    side_pm = W["side"].astype(np.float64) * 2.0 - 1.0
    log_scale = np.log(sc[:, 0, 0, 0])
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32,
                               device=device)
    return dict(traj_n=t(traj_n), conf=t(W["conf"]), wrist=t(wrist_n),
                elbow=t(elbow_n), side=t(side_pm), log_scale=t(log_scale),
                traj_n_np=traj_n, conf_np=W["conf"], mu=mu, scale=sc,
                clip=W["clip"], side_idx=W["side"])


def train_hand_model(W, epochs=600, lr=1e-3, vel_w=0.5, weight_decay=1e-4,
                     aug_noise=(0.0, 0.02, 0.05, 0.10), target_sigma=1.25,
                     n_obs=16, gap_prob=0.3, gap_lens=(2, 3, 4, 5, 6, 8),
                     batch_size=128, model_cfg=None, device="cpu", seed=0,
                     log=True, obs_jitter=True, even_prob=0.0):
    """W = dict from 32_build_windows (or a subset). Returns (model, history).

    target_sigma <= 0 = memorization mode: L_pos targets the RAW detections
    (overfit ceiling test); the velocity target stays on a lightly smoothed
    track (raw finite-diff would amplify detection noise (span-1)x).
    even_prob: probability of using the eval protocol's exact even-frame
    observation pattern (so retrieval is trained on the pattern it is
    tested with).
    """
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    D = prepare_windows(W, device)
    S, span = D["traj_n_np"].shape[:2]

    if target_sigma and target_sigma > 0:
        target = conf_smooth(D["traj_n_np"], D["conf_np"], target_sigma)
        vel_src = target
    else:
        target = D["traj_n_np"]
        vel_src = conf_smooth(D["traj_n_np"], D["conf_np"], 1.0)
    tgt_vel = np.diff(vel_src, axis=1) * (span - 1)               # per-tau
    wv = np.minimum(D["conf_np"][:, :-1], D["conf_np"][:, 1:])    # (S,span-1,21)

    tb = torch.tensor(target.reshape(S, span, -1), dtype=torch.float32,
                      device=device)
    vb = torch.tensor(tgt_vel.reshape(S, span - 1, -1), dtype=torch.float32,
                      device=device)
    wp = D["conf"].clamp(0, 1).repeat_interleave(2, dim=-1)       # (S,span,42)
    wvb = torch.tensor(wv, dtype=torch.float32,
                       device=device).clamp(0, 1).repeat_interleave(2, dim=-1)
    tau_d = torch.linspace(0, 1, span, device=device)
    tau_m = (tau_d[:-1] + tau_d[1:]) / 2

    m = HandSetSIREN(**(model_cfg or {})).to(device)
    opt = torch.optim.Adam(m.parameters(), lr, weight_decay=weight_decay)
    # linear warmup then cosine: without warmup, Adam+transformer diverges on
    # some seeds (observed: seed-0 scaling runs collapsed to train_loss ~1.5)
    warm = max(1, int(0.05 * epochs))
    sch = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, 0.05, 1.0, warm),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs - warm)],
        milestones=[warm])
    bs = min(batch_size, S)
    hist = []

    for ep in range(epochs):
        m.train()
        perm = torch.randperm(S, device=device)
        lp = lv = 0.0
        nb = 0
        for i in range(0, S, bs):
            idx = perm[i:i + bs]
            r = rng.rand()
            if r < gap_prob:
                oi = obs_gap(span, rng, gap_lens)
            elif r < gap_prob + even_prob:
                oi = np.arange(0, span, 2)                        # protocol A obs
            else:
                oi = obs_uniform(span, n_obs, rng, jitter=obs_jitter)
            oit = torch.tensor(oi, dtype=torch.long, device=device)
            tau_o = tau_d[oit].unsqueeze(0).expand(len(idx), -1)

            obs = D["traj_n"][idx][:, oit]                        # (B,n,21,2)
            sigma = float(rng.choice(aug_noise))
            if sigma > 0:
                obs = obs + torch.randn_like(obs) * sigma
            mod = m.encode(obs, D["conf"][idx][:, oit],
                           D["wrist"][idx][:, oit], D["elbow"][idx][:, oit],
                           D["log_scale"][idx], D["side"][idx], tau_o)

            pos = m.decode(mod, tau_d)                            # (B,span,42)
            w = wp[idx]
            loss_pos = ((pos - tb[idx]) ** 2 * w).sum() / w.sum()
            _, V = m.decode(mod, tau_m, with_velocity=True)
            wv_ = wvb[idx]
            loss_vel = ((V - vb[idx]) ** 2 * wv_).sum() / wv_.sum()
            loss = loss_pos + vel_w * loss_vel
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            lp += loss_pos.item()
            lv += loss_vel.item()
            nb += 1
        sch.step()
        hist.append(dict(ep=ep, loss_pos=lp / nb, loss_vel=lv / nb))
        if log and (ep % 25 == 0 or ep == epochs - 1):
            print(f"   ep{ep:4d} loss_pos={lp/nb:.5f} loss_vel={lv/nb:.4f}",
                  flush=True)
    return m, hist


def save_ckpt(path, model, extra=None):
    torch.save(dict(cfg=model.cfg, state=model.state_dict(),
                    extra=extra or {}), path)


def load_ckpt(path, device="cpu"):
    d = torch.load(path, map_location=device, weights_only=False)
    m = HandSetSIREN(**d["cfg"]).to(device)
    m.load_state_dict(d["state"])
    m.eval()
    return m, d.get("extra", {})
