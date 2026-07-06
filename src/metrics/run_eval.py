"""Orchestrate metric 1 (hand confidence) + metric 2 (motion fidelity) over the
DisPose and MimicMotion sign-language outputs, writing small CSVs.

Run from the repo root (DWPose weights are referenced relatively):

    python src/metrics/run_eval.py \
        --source-dir assets/example_data/sign_videos/hard27k_orig \
        --dispose-dirs outputs/20260705_test_sign_hard27k outputs/20260705_test_sign_hard27k_b2 \
                       outputs/20260705_test_sign_hard27k_c1 ... \
        --mimic-dirs   eval_inputs/mimic/... \
        --out-dir      outputs/metrics_hard27k --cache-dir outputs/metrics_hard27k/pose_cache

Source poses are extracted once and cached, then reused across both models.
"""
import argparse
import csv
import glob
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)   # sibling metric modules
sys.path.insert(0, ROOT)   # mimicmotion package + repo-relative weights

import numpy as np
import pose_extract as PE
from hand_confidence import hand_confidence_metrics
from motion_fidelity import motion_fidelity_metrics


def find_id(filename, id_set):
    """The known 10-char id that appears in the filename (handles _to_<id>_,
    <id>_hiya, <id>.mp4 naming)."""
    base = os.path.basename(filename)
    hits = [i for i in id_set if i in base]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:  # pick the longest match to be safe
        return max(hits, key=len)
    return None


def cached_poses(video_path, cache_dir, stride):
    if cache_dir:
        key = hashlib.md5((os.path.abspath(video_path) + f"|s{stride}").encode()).hexdigest()[:16]
        cpath = os.path.join(cache_dir, key + ".npz")
        if os.path.exists(cpath):
            return PE.load_poses(cpath)
        poses = PE.extract_video_poses(video_path, sample_stride=stride)
        PE.save_poses(cpath, poses)
        return poses
    return PE.extract_video_poses(video_path, sample_stride=stride)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--dispose-dirs", nargs="+", default=[])
    ap.add_argument("--mimic-dirs", nargs="+", default=[])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--dispose-offset", type=int, default=0)
    ap.add_argument("--mimic-offset", type=int, default=1)
    ap.add_argument("--det-thr", type=float, default=0.3)
    ap.add_argument("--pck-at", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    sources = {}
    for f in glob.glob(os.path.join(args.source_dir, "*.mp4")):
        sources[os.path.splitext(os.path.basename(f))[0]] = f
    id_set = set(sources)
    print(f"[src] {len(sources)} source videos", flush=True)

    jobs = []  # (model, offset, id, gen_path)
    for d in args.dispose_dirs:
        for f in glob.glob(os.path.join(d, "*.mp4")):
            vid = find_id(f, id_set)
            if vid:
                jobs.append(("dispose", args.dispose_offset, vid, f))
    for d in args.mimic_dirs:
        for f in glob.glob(os.path.join(d, "*.mp4")):
            vid = find_id(f, id_set)
            if vid:
                jobs.append(("mimic", args.mimic_offset, vid, f))
    print(f"[jobs] {len(jobs)} generated videos matched to sources", flush=True)

    src_cache = {}   # id -> poses (extract once)
    rows = []
    for n, (model, offset, vid, gen_path) in enumerate(jobs, 1):
        if vid not in src_cache:
            src_cache[vid] = cached_poses(sources[vid], args.cache_dir, args.stride)
        src = src_cache[vid]
        gen = cached_poses(gen_path, args.cache_dir, args.stride)

        m1 = hand_confidence_metrics(gen, det_thr=args.det_thr)
        m2 = motion_fidelity_metrics(gen, src, gen_offset=offset,
                                     thr=args.det_thr, pck_at=args.pck_at)
        rows.append(dict(
            id=vid, model=model, n_frames=m1["n_frames"],
            body_det_rate=round(m1["body_det_rate"], 4),
            mean_hand_conf=round(m1["mean_hand_conf"], 4),
            hand_good_rate=round(m1["hand_good_rate"], 4),
            body_pck=round(m2["body_pck"], 4), body_nme=round(m2["body_nme"], 5),
            hand_pck=round(m2["hand_pck"], 4), hand_nme=round(m2["hand_nme"], 5),
            hand_samples=m2["hand_samples"],
        ))
        print(f"[{n}/{len(jobs)}] {model:7s} {vid} "
              f"hand_conf={m1['mean_hand_conf']:.3f} hand_pck={m2['hand_pck']:.3f}",
              flush=True)

    # per-video CSV
    fields = ["id", "model", "n_frames", "body_det_rate", "mean_hand_conf",
              "hand_good_rate", "body_pck", "body_nme", "hand_pck", "hand_nme",
              "hand_samples"]
    per_video = os.path.join(args.out_dir, "per_video.csv")
    with open(per_video, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # aggregate per model (mean +/- std across videos)
    agg_fields = ["mean_hand_conf", "hand_good_rate", "body_det_rate",
                  "body_pck", "body_nme", "hand_pck", "hand_nme"]
    agg_path = os.path.join(args.out_dir, "aggregate.csv")
    with open(agg_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "n"] + [f"{m}_mean" for m in agg_fields]
                   + [f"{m}_std" for m in agg_fields])
        for model in ("dispose", "mimic"):
            mr = [r for r in rows if r["model"] == model]
            if not mr:
                continue
            means, stds = [], []
            for m in agg_fields:
                vals = np.array([r[m] for r in mr], float)
                vals = vals[np.isfinite(vals)]
                means.append(round(float(vals.mean()), 4) if len(vals) else float("nan"))
                stds.append(round(float(vals.std()), 4) if len(vals) else float("nan"))
            w.writerow([model, len(mr)] + means + stds)

    # paired delta (dispose - mimic) per id, for the head-to-head story
    paired_path = os.path.join(args.out_dir, "paired_delta.csv")
    by_id = {}
    for r in rows:
        by_id.setdefault(r["id"], {})[r["model"]] = r
    with open(paired_path, "w", newline="") as fh:
        w = csv.writer(fh)
        cols = ["mean_hand_conf", "hand_good_rate", "hand_pck", "hand_nme",
                "body_pck", "body_nme"]
        w.writerow(["id"] + [f"d_{c}" for c in cols])
        for vid, md in sorted(by_id.items()):
            if "dispose" in md and "mimic" in md:
                w.writerow([vid] + [round(md["dispose"][c] - md["mimic"][c], 4)
                                    for c in cols])

    print(f"\nWROTE {per_video}\n      {agg_path}\n      {paired_path}", flush=True)


if __name__ == "__main__":
    main()
