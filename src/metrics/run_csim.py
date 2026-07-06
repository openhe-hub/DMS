"""Compute CSIM for one model's generated videos against the reference avatar.
Run from the repo root (buffalo_l is cached under ~/.insightface).

    python src/metrics/run_csim.py --model dispose \
        --ref-image assets/example_data/sign_videos/refs/test2.jpg \
        --gen-dirs outputs/20260705_test_sign_hard27k ... \
        --source-dir assets/example_data/sign_videos/hard27k_orig \
        --out outputs/metrics_hard27k/csim_dispose.csv
"""
import argparse
import csv
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import csim as CS


def find_id(filename, id_set):
    base = os.path.basename(filename)
    hits = [i for i in id_set if i in base]
    if not hits:
        return None
    return max(hits, key=len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref-image", required=True)
    ap.add_argument("--gen-dirs", nargs="+", required=True)
    ap.add_argument("--source-dir", required=True, help="only used to map filenames->ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-frames", type=int, default=12)
    args = ap.parse_args()

    id_set = {os.path.splitext(os.path.basename(f))[0]
              for f in glob.glob(os.path.join(args.source_dir, "*.mp4"))}
    ref_emb = CS.ref_embedding(args.ref_image)
    print(f"[ref] embedded {args.ref_image}", flush=True)

    files = []
    for d in args.gen_dirs:
        for f in glob.glob(os.path.join(d, "*.mp4")):
            vid = find_id(f, id_set)
            if vid:
                files.append((vid, f))
    print(f"[jobs] {len(files)} {args.model} videos", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fields = ["id", "model", "n_sampled", "face_det_rate", "csim_mean",
              "csim_min", "csim_std", "csim_all"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for n, (vid, path) in enumerate(files, 1):
            r = CS.video_csim(path, ref_emb, n_frames=args.n_frames)
            row = dict(id=vid, model=args.model,
                       **{k: round(v, 4) for k, v in r.items()})
            w.writerow(row)
            fh.flush()
            print(f"[{n}/{len(files)}] {vid} csim={r['csim_mean']:.3f} "
                  f"min={r['csim_min']:.3f} facedet={r['face_det_rate']:.2f}", flush=True)
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
