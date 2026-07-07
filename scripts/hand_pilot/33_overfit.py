"""P0 overfit sanity: can HandSetSIREN represent real sign hand trajectories?

Stage judged by (pre-registered): conf-weighted L_pos approaching the Gate B
noise floor (measured relative jitter squared, canonical units) with smooth
analytic velocity (no ringing). All augmentation OFF (train = eval,
aug_noise=0, no gap masking, no obs jitter, dropout 0) -- this stage tests
capacity + pipeline, nothing else. --sweep runs the one-off w0 sweep
(frozen afterwards).

Usage:
  python scripts/hand_pilot/33_overfit.py --limit 64 --epochs 200   # smoke
  python scripts/hand_pilot/33_overfit.py                            # full
  python scripts/hand_pilot/33_overfit.py --sweep                    # w0 pick
"""
import argparse
import json
import os

import numpy as np

import _paths as P
from dispose_siren.hand_train import (train_hand_model, conf_smooth,
                                      prepare_windows, save_ckpt)
from dispose_siren.hand_model import velocity  # noqa: F401 (used in fig)


def load_windows(path, limit=0, seed=0):
    z = np.load(path, allow_pickle=True)
    W = {k: z[k] for k in z.files}
    if limit and limit < len(W["traj"]):
        idx = np.random.RandomState(seed).choice(len(W["traj"]), limit,
                                                 replace=False)
        W = {k: W[k][idx] for k in W}
    return W


def noise_floor():
    p = os.path.join(P.GATE_B_DIR, "summary.json")
    if not os.path.exists(p):
        return None
    s = json.load(open(p))
    jr = s.get("hand_jitter_rel")
    return None if jr is None else float(jr) ** 2   # per-coord var, canonical


def overfit_run(W, epochs, w0, device, log=True):
    import torch
    cfg = dict(w0=w0, dropout=0.0)
    m, hist = train_hand_model(
        W, epochs=epochs, aug_noise=(0.0,), gap_prob=0.0, obs_jitter=False,
        model_cfg=cfg, device=device, log=log)
    # ringing metric: analytic velocity vs fd of the pseudo-clean target,
    # excess high-frequency energy ratio (1.0 = matched smoothness)
    D = prepare_windows(W, device)
    span = W["traj"].shape[1]
    tgt = conf_smooth(D["traj_n_np"], W["conf"], 1.25)
    tgt_vel = np.diff(tgt, axis=1) * (span - 1)
    tau_d = torch.linspace(0, 1, span, device=device)
    tau_m = (tau_d[:-1] + tau_d[1:]) / 2
    oi = np.round(np.linspace(0, span - 1, 16)).astype(int)
    vs = []
    with torch.no_grad():
        for i in range(0, len(W["traj"]), 256):
            sl = slice(i, min(i + 256, len(W["traj"])))
            idx = torch.arange(sl.start, sl.stop, device=device)
            oit = torch.tensor(oi, dtype=torch.long, device=device)
            mod = m.encode(D["traj_n"][idx][:, oit], D["conf"][idx][:, oit],
                           D["wrist"][idx][:, oit], D["elbow"][idx][:, oit],
                           D["log_scale"][idx], D["side"][idx],
                           (oit.float() / (span - 1))[None].expand(len(idx), -1))
            _, V = m.decode(mod, tau_m, with_velocity=True)
            vs.append(V.cpu().numpy())
    V = np.concatenate(vs).reshape(len(W["traj"]), span - 1, 21, 2)
    hf = lambda a: np.abs(np.diff(a, axis=1)).mean()   # velocity roughness
    ring = float(hf(V) / (hf(tgt_vel) + 1e-9))
    return m, hist, ring


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows",
                    default=os.path.join(P.WINDOWS_DIR, "windows_span32.npz"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--device", default=None)
    ap.add_argument("--sweep", action="store_true", help="w0 in {5,15,30}")
    ap.add_argument("--w0", type=float, default=15.0)
    args = ap.parse_args()

    import torch
    dev = args.device or ("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    W = load_windows(args.windows, args.limit)
    S = len(W["traj"])
    floor = noise_floor()
    print(f"P0 overfit: {S} windows, device={dev}, "
          f"noise floor={'%.5f' % floor if floor else 'unknown (no Gate B)'}")

    if args.sweep:
        rows = []
        for w0 in (5.0, 15.0, 30.0):
            print(f"--- w0={w0}")
            _, hist, ring = overfit_run(W, min(args.epochs, 400), w0, dev)
            rows.append(dict(w0=w0, loss_pos=hist[-1]["loss_pos"],
                             loss_vel=hist[-1]["loss_vel"], ringing=ring))
        print(f"{'w0':>6} {'loss_pos':>10} {'loss_vel':>10} {'ringing':>8}")
        for r in rows:
            print(f"{r['w0']:>6} {r['loss_pos']:>10.5f} "
                  f"{r['loss_vel']:>10.4f} {r['ringing']:>8.2f}")
        json.dump(rows, open(os.path.join(P.FIG_DIR, "w0_sweep.json"), "w"),
                  indent=1)
        return

    m, hist, ring = overfit_run(W, args.epochs, args.w0, dev)
    lp = hist[-1]["loss_pos"]
    save_ckpt(os.path.join(P.CKPT_DIR, "p0_overfit.pt"), m,
              extra=dict(S=S, epochs=args.epochs, w0=args.w0))
    json.dump(hist, open(os.path.join(P.FIG_DIR, "p0_history.json"), "w"))
    make_figs(W, m, dev)

    print("\n===== P0 READOUT =====")
    print(f"  final conf-weighted L_pos = {lp:.5f} (canonical^2)")
    if floor:
        print(f"  Gate B noise floor       = {floor:.5f}  "
              f"(ratio {lp/floor:.1f}x)")
        verdict = "PASS" if lp < 4 * floor else "MARGINAL" if lp < 20 * floor \
            else "FAIL (capacity or pipeline problem)"
    else:
        verdict = "run Gate B for the floor comparison"
    print(f"  velocity roughness ratio = {ring:.2f} (~1 = smooth, >2 ringing)")
    print(f"  verdict: {verdict}")


def make_figs(W, m, device, n_show=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from dispose_siren.hand_train import prepare_windows

    D = prepare_windows(W, device)
    span = W["traj"].shape[1]
    tau_d = torch.linspace(0, 1, span, device=device)
    oi = np.round(np.linspace(0, span - 1, 16)).astype(int)
    oit = torch.tensor(oi, dtype=torch.long, device=device)
    idx = torch.arange(min(n_show, len(W["traj"])), device=device)
    with torch.no_grad():
        mod = m.encode(D["traj_n"][idx][:, oit], D["conf"][idx][:, oit],
                       D["wrist"][idx][:, oit], D["elbow"][idx][:, oit],
                       D["log_scale"][idx], D["side"][idx],
                       (oit.float() / (span - 1))[None].expand(len(idx), -1))
        pos, vel = m.decode(mod, tau_d, with_velocity=True)
    pos = pos.cpu().numpy().reshape(len(idx), span, 21, 2)
    tgt = conf_smooth(D["traj_n_np"][:len(idx)], W["conf"][:len(idx)], 1.25)
    raw = D["traj_n_np"][:len(idx)]

    kps = [0, 8, 20]  # wrist, index tip, pinky tip
    fig, ax = plt.subplots(len(idx), len(kps), figsize=(13, 3 * len(idx)),
                           squeeze=False)
    t = np.arange(span)
    for r in range(len(idx)):
        for c, k in enumerate(kps):
            a = ax[r][c]
            a.plot(t, raw[r, :, k, 0], ".", ms=3, alpha=0.5, label="raw x")
            a.plot(t, tgt[r, :, k, 0], "-", lw=1, label="target x")
            a.plot(t, pos[r, :, k, 0], "-", lw=1.5, label="model x")
            lo = W["conf"][r, :, k] < 0.3
            if lo.any():
                a.plot(t[lo], raw[r, lo, k, 0], "rx", ms=5, label="low conf")
            if r == 0 and c == 0:
                a.legend(fontsize=7)
            a.set_title(f"win{r} kp{k} ({W['clip'][r]})", fontsize=8)
    fig.tight_layout()
    out = os.path.join(P.FIG_DIR, "p0_overlay.png")
    fig.savefig(out, dpi=130)
    print(f"fig -> {out}")


if __name__ == "__main__":
    main()
