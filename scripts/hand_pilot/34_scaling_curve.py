"""P1/P2: clip-level split + scaling curve {16, 32, 64, all} x seeds.

The pre-registered judgment is NOT "beat spline at 85 train clips" (data-wall
regime, losing is expected) but the SLOPE: does held-out error fall with
training-set size steeply enough that asl50k plausibly crosses the spline
line? Splits are clip-level (windows within a clip are near-duplicates);
held-out = 24 clips stratified by window count and mean confidence, frozen in
split.json. Train-clip scores are logged too (memorization vs generalization).

Caveat for the memo: signer identity is unknown, so this is clip-independent,
not provably signer-independent.
"""
import argparse
import json
import os

import numpy as np

import _paths as P
from dispose_siren.hand_train import train_hand_model
from dispose_siren.hand_eval import (protocol_hand_holdout,
                                     protocol_gap_inpaint, subset_windows)


def clip_split(W, n_held=24, seed=0):
    """Stratified clip-level split -> (train_clips, held_clips)."""
    clips, counts = np.unique(W["clip"], return_counts=True)
    conf = np.array([W["conf"][W["clip"] == c].mean() for c in clips])
    order = np.lexsort((conf, counts))          # sort by count, then conf
    rng = np.random.RandomState(seed)
    stride = max(1, len(clips) // n_held)
    held = []
    for i in range(0, len(clips), stride):
        blk = order[i:i + stride]
        held.append(clips[rng.choice(blk)])
        if len(held) == n_held:
            break
    held = sorted(set(held))
    train = sorted(set(clips) - set(held))
    return train, held


def gap_lengths():
    p = os.path.join(P.GATE_B_DIR, "gap_lengths.json")
    if os.path.exists(p):
        gl = json.load(open(p))["lengths"]
        if gl:
            return gl
    return [2, 3, 4, 6, 8]


def aug_grid():
    p = os.path.join(P.GATE_B_DIR, "summary.json")
    if os.path.exists(p):
        jr = json.load(open(p)).get("hand_jitter_rel")
        if jr:
            return (0.0, 0.5 * jr, jr, 2 * jr)
    return (0.0, 0.02, 0.05, 0.10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows",
                    default=os.path.join(P.WINDOWS_DIR, "windows_span32.npz"))
    ap.add_argument("--sizes", default="16,32,64,all")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--n_held", type=int, default=24)
    ap.add_argument("--w0", type=float, default=15.0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    dev = args.device or ("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    z = np.load(args.windows, allow_pickle=True)
    W = {k: z[k] for k in z.files}
    train_clips, held_clips = clip_split(W, args.n_held)
    json.dump(dict(train=train_clips, held=held_clips), open(
        os.path.join(P.WINDOWS_DIR, "split.json"), "w"), indent=1)

    held_mask = np.isin(W["clip"], held_clips)
    W_held = subset_windows(W, held_mask)
    gl, aug = gap_lengths(), aug_grid()
    print(f"{len(train_clips)} train / {len(held_clips)} held clips; "
          f"{held_mask.sum()} held windows; device={dev}; aug={aug}")

    results = []
    sizes = [len(train_clips) if s == "all" else int(s)
             for s in args.sizes.split(",")]
    for size in sizes:
        n_seeds = 1 if size >= len(train_clips) else args.seeds
        for seed in range(n_seeds):
            rng = np.random.RandomState(1000 + seed)
            sub = (train_clips if size >= len(train_clips)
                   else list(rng.choice(train_clips, size, replace=False)))
            m_tr = np.isin(W["clip"], sub)
            W_tr = subset_windows(W, m_tr)
            print(f"--- size={size} seed={seed}: {m_tr.sum()} windows")
            model, hist = train_hand_model(
                W_tr, epochs=args.epochs, aug_noise=aug,
                model_cfg=dict(w0=args.w0), device=dev, seed=seed, log=False)
            rec = dict(size=size, seed=seed, n_windows=int(m_tr.sum()),
                       train_loss=hist[-1]["loss_pos"])
            rec["held_holdout"] = protocol_hand_holdout(W_held, model,
                                                        device=dev)
            rec["held_gap"] = protocol_gap_inpaint(W_held, model, gl,
                                                   seed=seed, device=dev)
            rec["train_holdout"] = protocol_hand_holdout(W_tr, model,
                                                         device=dev)
            print(f"    held holdout: learned={rec['held_holdout']['learned']:.5f} "
                  f"spline={rec['held_holdout']['spline']:.5f} "
                  f"linear={rec['held_holdout']['linear']:.5f}")
            results.append(rec)
            json.dump(results, open(os.path.join(P.OUT, "scaling.json"), "w"),
                      indent=1, default=float)
    print(f"-> {os.path.join(P.OUT, 'scaling.json')}")


if __name__ == "__main__":
    main()
