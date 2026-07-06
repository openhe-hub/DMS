"""R012 -- degradation-curve figures from step2_pilot_metrics.json.

Fig 1: PSNR_mid vs stride per system (the C1 money plot -- discrete baseline
       should fall, continuous control should stay near the orig:1 ceiling).
Fig 2: LPIPS_all vs stride. Fig 3: warp_error vs stride.
Values averaged over cases; per-case values plotted faint.
"""
import json
import os

import numpy as np

import _paths  # noqa: F401
from _paths import FIG_DIR

SYSTEMS = ["linear", "spline", "siren", "rife"]
COLORS = {"linear": "tab:blue", "spline": "tab:green", "siren": "tab:red",
          "rife": "tab:purple", "orig": "k"}


def collect(res, metric):
    """-> {system: {stride: [per-case values]}} plus orig:1 ceiling values."""
    out, ceil = {}, []
    for case, systems in res.items():
        for name, r in systems.items():
            if r.get(metric) is None:
                continue
            m, s = name.rsplit("_s", 1)
            s = int(s)
            if m == "orig" and s == 1:
                ceil.append(r[metric])
                continue
            out.setdefault(m, {}).setdefault(s, []).append(r[metric])
    return out, ceil


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = json.load(open(os.path.join(FIG_DIR, "step2_pilot_metrics.json")))
    specs = [("psnr_mid", "PSNR on in-between frames (dB) ^", False),
             ("lpips_all", "LPIPS v", True),
             ("warp", "output warp error v", True)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (metric, label, lower_better) in zip(axes, specs):
        data, ceil = collect(res, metric)
        for m in SYSTEMS:
            if m not in data:
                continue
            ss = sorted(data[m])
            mean = [np.mean(data[m][s]) for s in ss]
            ax.plot(ss, mean, "o-", color=COLORS[m], label=m)
            for s in ss:
                ax.plot([s] * len(data[m][s]), data[m][s], ".",
                        color=COLORS[m], alpha=0.3)
        if ceil:
            ax.axhline(np.mean(ceil), color="k", ls="--", lw=1,
                       label="orig full-fps ceiling")
        ax.set_xlabel("pose stride s")
        ax.set_title(label)
        ax.set_xticks([4, 8])
        ax.legend(fontsize=8)
    fig.suptitle("Pilot: low-fps driving -- continuous control vs discrete + post-hoc RIFE")
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "step2_pilot_curves.png")
    fig.savefig(p, dpi=130)
    print("wrote", p)


if __name__ == "__main__":
    main()
