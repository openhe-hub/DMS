"""Best-of-N seed selection by DWPose hand confidence (GT-free reranking).

Reads per-video mean_hand_conf for the original siren run and each reroll,
picks the argmax per clip, and assembles outputs/sign_siren_best/ (symlinks)
for the final metrics pass. Selection criterion needs NO ground truth --
this is deployable test-time reranking, disclosed in the table footnote.
"""
import argparse
import csv
import glob
import os
import re

import _paths as P


def per_video(path):
    return {r["id"]: float(r["mean_hand_conf"])
            for r in csv.DictReader(open(path))}


def find_mp4(dirs_glob, clip):
    for mp4 in glob.glob(os.path.join(dirs_glob, "*.mp4")):
        m = re.search(r"_to_([a-z0-9]+)_CFG", os.path.basename(mp4))
        if m and m.group(1) == clip:
            return os.path.abspath(mp4)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_csv",
                    default=os.path.join(P.REPO, "outputs/metrics_siren/per_video.csv"))
    ap.add_argument("--orig_glob",
                    default=os.path.join(P.REPO, "outputs/sign_siren_full/*"))
    ap.add_argument("--reroll", action="append", nargs=2,
                    metavar=("CSV", "GLOB"), default=[],
                    help="per_video.csv + mp4 dir glob of one reroll")
    ap.add_argument("--out_dir",
                    default=os.path.join(P.REPO, "outputs/sign_siren_best/best"))
    args = ap.parse_args()

    cands = [(per_video(args.orig_csv), args.orig_glob, "orig")]
    for csv_path, g in args.reroll:
        cands.append((per_video(csv_path), g, os.path.basename(csv_path)))

    os.makedirs(args.out_dir, exist_ok=True)
    picked = {"orig": 0}
    for clip in sorted(cands[0][0]):
        best = max(((pv.get(clip, -1), g, tag) for pv, g, tag in cands))
        conf, g, tag = best
        src = find_mp4(g, clip)
        assert src, f"no mp4 for {clip} in {g}"
        dst = os.path.join(args.out_dir, os.path.basename(src))
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)
        picked[tag] = picked.get(tag, 0) + 1
        if tag != "orig":
            print(f"  {clip}: reroll win ({tag}) conf {cands[0][0][clip]:.3f} "
                  f"-> {conf:.3f}")
    print(f"selection: {picked} -> {args.out_dir} "
          f"({len(os.listdir(args.out_dir))} videos)")


if __name__ == "__main__":
    main()
