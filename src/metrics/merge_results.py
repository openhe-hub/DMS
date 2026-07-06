"""Merge the per-model per_video.csv files produced on jubail (dispose) and
jubail2 (mimic) into a combined table, a per-model aggregate, and a paired
(dispose - mimic) delta. Pure stdlib; runs locally, no GPU.

    python src/metrics/merge_results.py \
        --inputs outputs/metrics_hard27k/per_video_dispose.csv \
                 outputs/metrics_hard27k/per_video_mimic.csv \
        --out-dir outputs/metrics_hard27k
"""
import argparse
import csv
import os
import math

NUM_COLS = ["mean_hand_conf", "hand_good_rate", "body_det_rate",
            "body_pck", "body_nme", "hand_pck", "hand_nme"]


def _mean_std(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
    return m, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for path in args.inputs:
        with open(path) as fh:
            for r in csv.DictReader(fh):
                for k, v in r.items():
                    if k not in ("id", "model"):
                        try:
                            r[k] = float(v)
                        except (ValueError, TypeError):
                            r[k] = None
                rows.append(r)

    combined = os.path.join(args.out_dir, "per_video.csv")
    with open(combined, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # aggregate per model
    agg = os.path.join(args.out_dir, "aggregate.csv")
    with open(agg, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "n"] + [f"{c}_mean" for c in NUM_COLS]
                   + [f"{c}_std" for c in NUM_COLS])
        for model in ("dispose", "mimic"):
            mr = [r for r in rows if r["model"] == model]
            if not mr:
                continue
            ms, ss = [], []
            for c in NUM_COLS:
                m, s = _mean_std([r.get(c) for r in mr])
                ms.append(round(m, 4)); ss.append(round(s, 4))
            w.writerow([model, len(mr)] + ms + ss)

    # paired delta (dispose - mimic) per id
    by_id = {}
    for r in rows:
        by_id.setdefault(r["id"], {})[r["model"]] = r
    paired = os.path.join(args.out_dir, "paired_delta.csv")
    delta_cols = ["mean_hand_conf", "hand_good_rate", "hand_pck", "hand_nme",
                  "body_pck", "body_nme"]
    wins = {c: 0 for c in delta_cols}
    npair = 0
    with open(paired, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id"] + [f"d_{c}" for c in delta_cols])
        for vid, md in sorted(by_id.items()):
            if "dispose" in md and "mimic" in md:
                npair += 1
                deltas = []
                for c in delta_cols:
                    dv = md["dispose"].get(c)
                    mv = md["mimic"].get(c)
                    if dv is None or mv is None:
                        deltas.append("")
                        continue
                    d = dv - mv
                    deltas.append(round(d, 4))
                    # nme lower is better -> win if delta<0; others higher better
                    better = d < 0 if c.endswith("nme") else d > 0
                    if better:
                        wins[c] += 1
                w.writerow([vid] + deltas)

    print(f"WROTE {combined}\n      {agg}\n      {paired}")
    print(f"\nPaired videos: {npair}")
    print("DisPose wins (per id):")
    for c in delta_cols:
        print(f"  {c:16s} {wins[c]}/{npair}")


if __name__ == "__main__":
    main()
