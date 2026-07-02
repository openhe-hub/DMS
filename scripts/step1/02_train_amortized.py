"""Step 1.2 -- train the scale-invariant amortized FiLM-SIREN on the synthetic
prior, sanity-check it still beats fd+Gaussian on SYNTHETIC held-out data
(velocity MSE), and save the checkpoint for real-data evaluation.

The sanity check is the gate: if the scale-invariant variant no longer wins on
synthetic, transfer to real is meaningless and we say so.
"""
import argparse
import os

import numpy as np
import torch

import _paths  # noqa: F401
from _paths import CKPT_DIR
from dispose_siren import N_FRAMES
from dispose_siren.synth import sample_traj, TF, TG
from dispose_siren import baselines as B
from dispose_siren import normalize as Z
from dispose_siren.train import train_amortized, save_ckpt


def synthetic_sanity(model, device, sigmas=(3, 6, 12, 20), nte=200, seeds=3):
    print("\n=== synthetic sanity: velocity MSE vs GT (lower=better) ===", flush=True)
    print(f"{'sigma':>6} | {'finite-diff':>14} | {'fd+Gauss(best)':>16} | {'learned-INR':>14} | win")
    print("-" * 70)
    rows = []
    for sigma in sigmas:
        r = {"fd": [], "fdg": [], "learned": []}
        for sd in range(seeds):
            rng = np.random.RandomState(1000 + sd)
            cp, cv, cf = sample_traj(rng, nte)
            noisy = cf + np.random.RandomState(5000 + sd).randn(nte, N_FRAMES, 2) * sigma
            v_fd = B.fd_dense(noisy, TF, TG)
            _, _, v_fdg = B.best_sigma_fdg(noisy, TF, TG, cv)
            _, film, mu, s = Z.infer(model, noisy, TG, device)
            v_le = Z.velocity_px(model, film, mu, s, TG, device)
            r["fd"].append(np.mean((v_fd - cv) ** 2))
            r["fdg"].append(np.mean((v_fdg - cv) ** 2))
            r["learned"].append(np.mean((v_le - cv) ** 2))
        m = {k: (float(np.mean(v)), float(np.std(v))) for k, v in r.items()}
        win = "LEARNED" if m["learned"][0] < min(m["fd"][0], m["fdg"][0]) else \
              ("fd+G" if m["fdg"][0] < m["fd"][0] else "fd")
        print(f"{sigma:>6} | {m['fd'][0]:8.1f}±{m['fd'][1]:4.0f} | "
              f"{m['fdg'][0]:8.1f}±{m['fdg'][1]:4.0f} | "
              f"{m['learned'][0]:8.1f}±{m['learned'][1]:4.0f} | {win}", flush=True)
        rows.append((sigma, m, win))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--ntr", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 120 if args.smoke else args.epochs
    ntr = 1200 if args.smoke else args.ntr
    print(f"device={device} epochs={epochs} ntr={ntr}", flush=True)

    model = train_amortized(epochs=epochs, ntr=ntr, device=device, log=True)
    rows = synthetic_sanity(model, device)

    ck = os.path.join(CKPT_DIR, "amortized.pt")
    save_ckpt(model, ck, meta={"epochs": epochs, "ntr": ntr,
                               "sanity_win": [r[2] for r in rows]})
    print(f"\nsaved checkpoint -> {ck}", flush=True)


if __name__ == "__main__":
    main()
