"""Compute FVD(model_outputs, source_videos) for one model. Run from repo root.

    python src/metrics/run_fvd.py --model dispose --i3d /path/i3d_torchscript.pt \
        --source-dir assets/example_data/sign_videos/hard27k_orig \
        --gen-dirs outputs/20260705_test_sign_hard27k ... \
        --out outputs/metrics_hard27k/fvd_dispose.json

Saves the I3D feature arrays (.npy) next to --out so the two models' features can
be re-scored locally if needed. Lower FVD = better.
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import numpy as np
import torch
import fvd as FV


def find_id(filename, id_set):
    base = os.path.basename(filename)
    hits = [i for i in id_set if i in base]
    return max(hits, key=len) if hits else None


def collect(files, i3d, device, stride):
    feats = []
    for n, f in enumerate(files, 1):
        try:
            feats.append(FV.video_features(f, i3d, device, stride=stride))
        except Exception as e:
            print(f"  skip {os.path.basename(f)}: {type(e).__name__} {e}", flush=True)
            continue
        if n % 20 == 0:
            print(f"  {n}/{len(files)} videos", flush=True)
    return np.concatenate(feats, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--i3d", required=True)
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--gen-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip-stride", type=int, default=16)
    ap.add_argument("--bootstrap", type=int, default=50)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    i3d = FV.load_i3d(args.i3d, device)
    print(f"[i3d] loaded on {device}", flush=True)

    src_files = sorted(glob.glob(os.path.join(args.source_dir, "*.mp4")))
    id_set = {os.path.splitext(os.path.basename(f))[0] for f in src_files}
    gen_files = []
    for d in args.gen_dirs:
        for f in glob.glob(os.path.join(d, "*.mp4")):
            if find_id(f, id_set):
                gen_files.append(f)
    print(f"[files] {len(gen_files)} gen, {len(src_files)} source", flush=True)

    print("[extract] source features", flush=True)
    src_feats = collect(src_files, i3d, device, args.clip_stride)
    print("[extract] gen features", flush=True)
    gen_feats = collect(gen_files, i3d, device, args.clip_stride)
    print(f"[feats] gen {gen_feats.shape} src {src_feats.shape}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    base = os.path.splitext(args.out)[0]
    np.save(base + "_genfeat.npy", gen_feats)
    np.save(base + "_srcfeat.npy", src_feats)

    fvd = FV.frechet_distance(gen_feats, src_feats)

    # bootstrap CI over resampled clips
    boots = []
    rng = np.random.RandomState(12345)
    for _ in range(args.bootstrap):
        gi = rng.randint(0, len(gen_feats), len(gen_feats))
        si = rng.randint(0, len(src_feats), len(src_feats))
        boots.append(FV.frechet_distance(gen_feats[gi], src_feats[si]))
    boots = np.array(boots)

    result = dict(
        model=args.model, fvd=round(fvd, 3),
        fvd_boot_mean=round(float(boots.mean()), 3),
        fvd_boot_std=round(float(boots.std()), 3),
        fvd_ci95=[round(float(np.percentile(boots, 2.5)), 3),
                  round(float(np.percentile(boots, 97.5)), 3)],
        n_gen_clips=int(len(gen_feats)), n_src_clips=int(len(src_feats)),
        n_gen_videos=len(gen_files), n_src_videos=len(src_files),
        clip_stride=args.clip_stride,
    )
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print("FVD RESULT:", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
