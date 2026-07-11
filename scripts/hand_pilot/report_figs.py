"""Decision-memo figures + readout from the pilot runs.

Consumes: outputs/hand_pilot/scaling.json (34), gate_b/summary.json (31),
fig/p0_history.json (33). Produces fig/scaling.png, fig/gap_inpaint.png and a
printed verdict block for docs/experiments/hand_pilot.md.

Reminder baked into the plot: losing to spline at ~85 train clips is the
EXPECTED data-wall outcome; the judgment is the slope and the extrapolated
spline-crossing size vs asl50k.
"""
import argparse
import json
import os

import numpy as np

import _paths as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaling", default=os.path.join(P.OUT, "scaling.json"))
    args = ap.parse_args()

    res = json.load(open(args.scaling))
    sizes = sorted({r["size"] for r in res})
    agg = {}
    for s in sizes:
        rr = [r for r in res if r["size"] == s]
        agg[s] = dict(
            learned=[r["held_holdout"]["learned"] for r in rr],
            train_learned=[r["train_holdout"]["learned"] for r in rr],
        )
    spline = float(np.mean([r["held_holdout"]["spline"] for r in res]))
    linear = float(np.mean([r["held_holdout"]["linear"] for r in res]))
    gauss = float(np.mean([r["held_holdout"]["gauss+lin"] for r in res]))

    x = np.array(sizes, float)
    y = np.array([np.mean(agg[s]["learned"]) for s in sizes])
    yerr = np.array([np.std(agg[s]["learned"]) for s in sizes])
    # power-law fit log(err) = a + b log(N); crossing with spline line
    b, a = np.polyfit(np.log(x), np.log(y), 1)
    n_cross = float(np.exp((np.log(spline) - a) / b)) if b < 0 else float("inf")

    make_scaling_fig(x, y, yerr, agg, spline, linear, gauss, a, b, n_cross)
    make_gap_fig(res, sizes)

    verdict = {
        "sizes": sizes,
        "held_learned_mean": dict(zip(map(str, sizes), y.tolist())),
        "spline": spline, "linear": linear, "gauss+lin": gauss,
        "powerlaw_slope": float(b),
        "spline_crossing_clips": n_cross,
        "asl50k_justified": bool(b < -0.05 and n_cross < 50_000),
    }
    json.dump(verdict, open(os.path.join(P.OUT, "decision.json"), "w"),
              indent=1)
    print("\n===== SCALING READOUT =====")
    print(f"  held-out learned MSE by size: "
          + "  ".join(f"{s}:{np.mean(agg[s]['learned']):.5f}" for s in sizes))
    print(f"  baselines: spline={spline:.5f} linear={linear:.5f} "
          f"gauss={gauss:.5f}")
    print(f"  power-law slope b={b:.3f}; extrapolated spline crossing at "
          f"~{n_cross:,.0f} clips" if np.isfinite(n_cross) else
          f"  slope b={b:.3f} (non-decreasing -- scaling will not help)")
    print(f"  asl50k justified: {verdict['asl50k_justified']} "
          f"(pre-registered: slope<-0.05 AND crossing < 50k)")


def make_scaling_fig(x, y, yerr, agg, spline, linear, gauss, a, b, n_cross):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.errorbar(x, y, yerr, marker="o", label="learned (held-out clips)")
    tr = [np.mean(agg[s]["train_learned"]) for s in sorted(agg)]
    ax.plot(x, tr, "o--", alpha=0.5, label="learned (train clips)")
    xs = np.geomspace(x[0], 5e4, 100)
    ax.plot(xs, np.exp(a + b * np.log(xs)), ":", alpha=0.7,
            label=f"power-law fit (slope {b:.2f})")
    for v, n in ((spline, "spline"), (linear, "linear"), (gauss, "gauss+lin")):
        ax.axhline(v, ls="--", lw=1, alpha=0.7)
        ax.text(x[0], v, f" {n}", va="bottom", fontsize=8)
    if np.isfinite(n_cross) and n_cross < 5e5:
        ax.axvline(n_cross, color="g", ls=":", alpha=0.7)
        ax.text(n_cross, ax.get_ylim()[1], f" cross ~{n_cross:,.0f}",
                va="top", fontsize=8, color="g")
    ax.axvline(5e4, color="k", ls=":", alpha=0.4)
    ax.text(5e4, ax.get_ylim()[0], " asl50k", fontsize=8, rotation=90)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training clips")
    ax.set_ylabel("held-out masked MSE (canonical$^2$)")
    ax.set_title("Scaling: losing to spline at 85 clips is EXPECTED;\n"
                 "the judgment is the slope", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(P.FIG_DIR, "scaling.png")
    fig.savefig(out, dpi=140)
    print(f"fig -> {out}")


def make_gap_fig(res, sizes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    big = [r for r in res if r["size"] == sizes[-1]]
    if not big or "by_gap_len" not in big[0].get("held_gap", {}):
        return
    buckets = sorted({int(L) for r in big for L in r["held_gap"]["by_gap_len"]})
    methods = ("linear", "spline", "gauss+lin", "learned")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    w = 0.2
    for mi, m in enumerate(methods):
        vals = []
        for L in buckets:
            vs = [r["held_gap"]["by_gap_len"][str(L)][m]["mse"]
                  for r in big if str(L) in r["held_gap"]["by_gap_len"]]
            vals.append(np.mean(vs) if vs else np.nan)
        ax.bar(np.arange(len(buckets)) + (mi - 1.5) * w, vals, w, label=m)
    ax.set_xticks(range(len(buckets)), [str(b) for b in buckets])
    ax.set_xlabel("gap length (frames)")
    ax.set_ylabel("masked MSE (canonical$^2$)")
    ax.set_yscale("log")
    ax.set_title(f"Gap inpainting, {sizes[-1]}-clip model (the actual use case)",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(P.FIG_DIR, "gap_inpaint.png")
    fig.savefig(out, dpi=140)
    print(f"fig -> {out}")


if __name__ == "__main__":
    main()
