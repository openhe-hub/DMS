"""Jitter metric for OmniHands trajectories dumped by the OMNIHAND_SMOOTH=savgol patch.

Reads demo_out_smooth/<vname>/traj.npz (raw vs smoothed verts + cam translations)
and reports mean second-difference (acceleration) magnitude per frame — the
standard jitter proxy: lower = steadier, real motion is preserved by low-order
Savitzky-Golay so large drops mean removed noise, not removed motion.

Usage: python jitter_metric.py <traj.npz> [<traj.npz> ...]
"""
import sys

import numpy as np


def accel(x):
    """Mean L2 norm of the discrete second difference along time."""
    a = x[2:] - 2 * x[1:-1] + x[:-2]
    return float(np.linalg.norm(a, axis=-1).mean())


def report(path):
    d = np.load(path)
    name = path.split("/")[-2]
    print(f"\n== {name} ==")
    print(f"{'signal':<18}{'raw':>12}{'smoothed':>12}{'reduction':>12}")
    for label, raw_key, sm_key in [
        ("verts R (mm)", "raw_vr", "sm_vr"),
        ("verts L (mm)", "raw_vl", "sm_vl"),
        ("cam_t R (mm)", "raw_cr", "sm_cr"),
        ("cam_t L (mm)", "raw_cl", "sm_cl"),
    ]:
        r = accel(d[raw_key] * 1000)  # meters -> mm
        s = accel(d[sm_key] * 1000)
        print(f"{label:<18}{r:>12.3f}{s:>12.3f}{100 * (1 - s / r):>11.1f}%")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
