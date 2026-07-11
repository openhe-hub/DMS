"""Build stacked hand-trajectory windows from the extracted poses.

Output: outputs/hand_pilot/windows/windows_span{span}.npz with the arrays from
`hand_traj.make_hand_windows` concatenated over all clips/sides, plus a
per-clip yield table on stdout (the memo's data-accounting appendix).
"""
import argparse
import glob
import os

import numpy as np

import _paths as P
from dispose_siren.hand_traj import (load_poses, make_hand_windows,
                                     concat_windows, hand_canon, HAND_ORDER)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses_dir", default=P.POSES_DIR)
    ap.add_argument("--span", type=int, default=32)
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--conf_thr", type=float, default=0.3)
    ap.add_argument("--min_good_frac", type=float, default=0.8)
    ap.add_argument("--min_bone_px", type=float, default=8.0)
    ap.add_argument("--max_canon_amp", type=float, default=12.0,
                    help="drop windows whose canonical |coord| exceeds this "
                         "(hand-units; detection jumps, not real motion)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.poses_dir, "*.npz")))
    if not files:
        raise SystemExit(f"no poses under {args.poses_dir}; run 30_ first")

    parts, table = [], []
    for fp in files:
        clip = os.path.splitext(os.path.basename(fp))[0]
        poses = load_poses(fp)
        w = make_hand_windows(poses, clip, span=args.span, step=args.step,
                              conf_thr=args.conf_thr,
                              min_good_frac=args.min_good_frac,
                              min_bone_px=args.min_bone_px)
        n_side = {s: 0 for s in HAND_ORDER}
        if w is not None:
            parts.append(w)
            for i, s in enumerate(HAND_ORDER):
                n_side[s] = int((w["side"] == i).sum())
        table.append((clip, int(poses["meta"][0]["T"]), n_side))

    W = concat_windows(parts)
    if W is None:
        raise SystemExit("0 windows survived gating; loosen thresholds")

    # canon smoke: raises on degenerate scale, checks round-trip
    traj_n, _, _, mu, sc = hand_canon(W["traj"], W["conf"])
    back = traj_n * sc + mu
    rt = float(np.abs(back - W["traj"]).max())
    assert rt < 1e-9, f"canon round-trip failed: {rt}"

    # amplitude gate: canonical coords beyond max_canon_amp hand-units within
    # ~1s are detection jumps (the P0 blow-up source), not real motion
    amax = np.abs(traj_n).reshape(len(traj_n), -1).max(axis=1)
    keep = amax <= args.max_canon_amp
    n_amp_drop = int((~keep).sum())
    if n_amp_drop:
        W = {k: np.asarray(W[k])[keep] for k in W}
        print(f"amplitude gate: dropped {n_amp_drop} windows with "
              f"|canon|>{args.max_canon_amp} (max was {amax.max():.1f})")

    out = os.path.join(P.WINDOWS_DIR, f"windows_span{args.span}.npz")
    np.savez_compressed(out, **W)

    print(f"{'clip':>14} {'T':>5}  " + "  ".join(f"{s:>5}" for s in HAND_ORDER))
    for clip, T, n_side in table:
        print(f"{clip:>14} {T:>5}  "
              + "  ".join(f"{n_side[s]:>5}" for s in HAND_ORDER))
    n_clips_used = len({c for c in W['clip']})
    print(f"\ntotal {len(W['traj'])} windows from {n_clips_used}/{len(files)} "
          f"clips (span={args.span} step={args.step}); canon round-trip OK")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
