"""Step 1.5 -- does training the prior on REAL motion fix the Step-1 domain gap?

Leave-one-video-out: train the FiLM-SIREN self-supervised on 2 videos' windows,
test on the held-out video (true generalization, not overfit). Compare, on the
held-out video, both honest protocols across:
    baseline (best of fd/linear, tuned) | synthetic-trained INR | real-trained INR

If real-trained overtakes baseline where synthetic-trained lost, the domain gap
was the cause and is fixable. Preliminary: only 3 videos -- needs more data for a
paper-grade claim.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

import _paths  # noqa: F401
from _paths import TRAJ_DIR, CKPT_DIR, FIG_DIR
from dispose_siren.trajectory import load_npz
from dispose_siren.train import load_ckpt
from dispose_siren.real_train import real_train
from dispose_siren.eval_protocols import make_windows, protocol_holdout, protocol_pseudogt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--span", type=int, default=48)
    ap.add_argument("--step", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=600)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # synthetic-trained reference model (from 02)
    synth_model, _ = load_ckpt(os.path.join(CKPT_DIR, "amortized.pt"), device)

    per_video = {}
    for t in sorted(glob.glob(os.path.join(TRAJ_DIR, "*.npz"))):
        name = os.path.splitext(os.path.basename(t))[0]
        pts, vis, _ = load_npz(t)
        w = make_windows(pts, vis, span=args.span, step=args.step)
        if len(w):
            per_video[name] = w
    names = sorted(per_video)
    print(f"videos={names}  windows={[len(per_video[n]) for n in names]}  device={device}", flush=True)

    folds = []
    for held in names:
        train_w = np.concatenate([per_video[n] for n in names if n != held], 0)
        test_w = per_video[held]
        print(f"\n=== fold: train on {[n for n in names if n!=held]} ({len(train_w)} win) "
              f"-> test {held} ({len(test_w)} win) ===", flush=True)
        real_model = real_train(train_w, epochs=args.epochs, device=device, seed=0, log=True)

        def run(model):
            rA = protocol_holdout(test_w, model, device)
            rB = protocol_pseudogt(test_w, model, device)
            return rA, rB

        rA_s, rB_s = run(synth_model)
        rA_r, rB_r = run(real_model)
        baseA = min(rA_s["linint"], rA_s["gauss+lin"])
        baseB = min(rB_s["fd"], rB_s["fd+gauss"])
        row = {"held": held, "S": int(len(test_w)),
               "baseA": baseA, "synthA": rA_s["learned"], "realA": rA_r["learned"],
               "baseB": baseB, "synthB": rB_s["learned"], "realB": rB_r["learned"]}
        folds.append(row)
        print(f"  A held-out pos-MSE:  base={baseA:.2f}  synth={rA_s['learned']:.2f}  "
              f"real={rA_r['learned']:.2f}  -> real {'BEATS' if rA_r['learned']<baseA else 'loses'} base", flush=True)
        print(f"  B pseudoGT vel-MSE:  base={baseB:.1f}  synth={rB_s['learned']:.1f}  "
              f"real={rB_r['learned']:.1f}  -> real {'BEATS' if rB_r['learned']<baseB else 'loses'} base", flush=True)

    def agg(key):
        tot = sum(f[key] * f["S"] for f in folds); n = sum(f["S"] for f in folds)
        return tot / n
    aA = {k: agg(k) for k in ["baseA", "synthA", "realA"]}
    aB = {k: agg(k) for k in ["baseB", "synthB", "realB"]}
    print("\n" + "=" * 72)
    print(f"LOVO AGGREGATE ({sum(f['S'] for f in folds)} held-out windows)")
    print(f"  A pos-MSE:  base={aA['baseA']:.2f}  synth-INR={aA['synthA']:.2f}  "
          f"real-INR={aA['realA']:.2f}  | real/base={aA['realA']/aA['baseA']:.2f}x")
    print(f"  B vel-MSE:  base={aB['baseB']:.1f}  synth-INR={aB['synthB']:.1f}  "
          f"real-INR={aB['realB']:.1f}  | real/base={aB['realB']/aB['baseB']:.2f}x")
    print(f"  verdict A: real-INR {'BEATS' if aA['realA']<aA['baseA'] else 'loses to'} baseline")
    print(f"  verdict B: real-INR {'BEATS' if aB['realB']<aB['baseB'] else 'loses to'} baseline")
    print("  (preliminary -- 3 videos, leave-one-out; needs more data for a paper claim)")
    print("=" * 72, flush=True)

    out = {"folds": folds, "aggregate": {"A": aA, "B": aB}}
    jpath = os.path.join(FIG_DIR, "step1_lovo.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"wrote {jpath}", flush=True)


if __name__ == "__main__":
    main()
