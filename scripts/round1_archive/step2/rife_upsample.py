"""R011 -- RIFE post-hoc baseline (the killer anti-claim).

Takes the ORIGINAL DisPose output generated on the coarse grid (orig_s{s}.pt,
one frame per observed pose) and temporally upsamples it x s with RIFE
(v4.x IFNet, arbitrary-timestep inference), producing rife_s{s}.pt on the SAME
full grid as the continuous-control systems. If this matches linear/spline/
siren, control-side continuity is unnecessary.
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

import _paths  # noqa: F401
from _paths import OUT

RIFE_DIR = "/scratch/zl6890/zhewen/tools/Practical-RIFE"


def load_rife(rife_dir):
    sys.path.insert(0, rife_dir)
    cwd = os.getcwd()
    os.chdir(rife_dir)                        # train_log path is relative
    from train_log.RIFE_HDv3 import Model
    model = Model(-1)
    model.load_model("train_log", -1)
    model.eval()
    os.chdir(cwd)
    return model


def upsample(model, frames_u8, s, device):
    """(n,3,H,W) uint8 -> ((n-1)*s+1,3,H,W) uint8 via timestep interpolation."""
    n, _, H, W = frames_u8.shape
    ph = ((H - 1) // 32 + 1) * 32
    pw = ((W - 1) // 32 + 1) * 32
    pad = (0, pw - W, 0, ph - H)
    out = []
    with torch.no_grad():
        for i in range(n - 1):
            I0 = F.pad(frames_u8[i:i + 1].to(device).float() / 255., pad)
            I1 = F.pad(frames_u8[i + 1:i + 2].to(device).float() / 255., pad)
            out.append(frames_u8[i])
            for j in range(1, s):
                mid = model.inference(I0, I1, timestep=j / s)
                mid = (mid[0, :, :H, :W].clamp(0, 1) * 255).byte().cpu()
                out.append(mid)
        out.append(frames_u8[-1])
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, required=True)
    ap.add_argument("--strides", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--rife_dir", default=RIFE_DIR)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    case_dir = os.path.join(OUT, "pilot", f"case{args.case}")
    model = load_rife(args.rife_dir)
    print(f"RIFE loaded (v{model.version}) device={device}", flush=True)

    for s in args.strides:
        src = os.path.join(case_dir, f"orig_s{s}.pt")
        if not os.path.exists(src):
            print(f"missing {src}, skip", flush=True)
            continue
        frames = torch.load(src)
        up = upsample(model, frames, s, device)
        dst = os.path.join(case_dir, f"rife_s{s}.pt")
        torch.save(up, dst)
        print(f"rife x{s}: {tuple(frames.shape)} -> {tuple(up.shape)} -> {dst}",
              flush=True)


if __name__ == "__main__":
    main()
