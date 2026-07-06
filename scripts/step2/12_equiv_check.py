"""R001 gate -- stride=1 equivalence (integration identity).

GATE A (blocking): starting from ONE shared detection pass, the interp@stride=1
path (subsample nothing -> reconstruct -> redraw -> rebuild control) must be
bit-identical to feeding the detections straight through:
    detections -> build_control        ==  detections -> interp@1 -> build_control
    draw(detected dicts)               ==  draw(interp@1 dicts)
If this holds, pilot system differences are method differences by construction
(all pilot systems share one detection pass inside 11_lowfps_generate).

GATE B (informational): detect_video_full vs the original get_video_pose --
run as two INDEPENDENT detection passes, so any mismatch here measures ONNX-GPU
detector nondeterminism, not integration error. Reported per visible/invisible
keypoint; large diffs concentrated on invisible (subset=-1) keypoints are
expected garbage-coordinate jitter.
"""
import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

import _paths  # noqa: F401
from _paths import REPO, OUT

import inference_ctrl as IC
from lib_lowfps import detect_video_full, draw_sequence, interp_pose_dicts, person0
from importlib import import_module

lowfps_gen = import_module("11_lowfps_generate")


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO, "configs/test.yaml"))
    ap.add_argument("--case", type=int, default=1)
    args = ap.parse_args()

    infer_config = OmegaConf.load(args.config)
    task = infer_config.test_case[args.case]
    video = task.ref_video_path
    ref_png = lowfps_gen.get_self_ref(video, os.path.join(OUT, "refs"))
    image_pixels, ref_img, h_t, w_t = IC.load_ref_image(ref_png, task.resolution)

    frames, detected, ref_pose = detect_video_full(
        video, image_pixels, sample_stride=task.sample_stride)
    img_1x3hw = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))
    npersons = max(d["bodies"]["candidate"].shape[0] // 18 for d in detected)
    detected = person0(detected)   # control path uses person 0 only (as DisPose does)

    # ---------------- GATE A: integration identity on the SHARED detections
    print("GATE A (blocking): interp@1 integration identity", flush=True)
    ok = True
    check("single person (info)", npersons == 1,
          f"max persons/frame={npersons}; person-0 truncation applied")

    bp_direct = [ref_pose["bodies"]] + [d["bodies"] for d in detected]
    lo = IC.build_control(img_1x3hw, ref_img, bp_direct, None, h_t, w_t, lite=True)
    drawn_direct = draw_sequence(detected, h_t, w_t)

    for method in ("linear", "spline"):
        dicts, t_eval, _ = interp_pose_dicts(detected, 1, method)
        ok &= check(f"{method}@1 grid covers all frames", len(dicts) == len(detected))
        ci = np.stack([d["bodies"]["candidate"] for d in dicts])
        cn = np.stack([d["bodies"]["candidate"][:18] for d in detected])
        dm = float(np.abs(ci - cn).max())
        ok &= check(f"{method}@1 candidates identical", dm == 0.0, f"maxdiff={dm:.2e}")

        drawn_i = draw_sequence(dicts, h_t, w_t)
        dd = int(np.abs(drawn_i.astype(int) - drawn_direct.astype(int)).max())
        ok &= check(f"{method}@1 drawn skeletons identical", dd == 0, f"maxdiff={dd}")

        bp_i = [ref_pose["bodies"]] + [d["bodies"] for d in dicts]
        li = IC.build_control(img_1x3hw, ref_img, bp_i, None, h_t, w_t, lite=True)
        for k in ("traj_flow", "mask", "mask_384", "controlnet_image"):
            dm = float((lo[k].float() - li[k].float()).abs().max())
            ok &= check(f"{method}@1 control {k}", dm == 0.0, f"maxdiff={dm:.2e}")

        # sparse_flow itself is NOT bit-reproducible even on identical inputs:
        # pose2track parks invisible keypoints at (0,0) and sample_optical_flow
        # scatter-writes with duplicate indices (write order is a race). So we
        # gate on the DETERMINISTIC inputs to get_sparse_flow instead, and only
        # report the raced output diff for information.
        from mimicmotion.utils.utils import pose2track
        po, pso = pose2track(bp_direct, h_t, w_t)
        pi, psi = pose2track(bp_i, h_t, w_t)
        dm = float(np.abs(po - pi).max()) + float(np.abs(pso - psi).max())
        ok &= check(f"{method}@1 pose2track (sparse-flow input)", dm == 0.0,
                    f"maxdiff={dm:.2e}")
        for k in ("sparse_flow", "sparse_flow_384"):
            dm = float((lo[k].float() - li[k].float()).abs().max())
            print(f"  [info] {method}@1 {k} raced-scatter diff={dm:.2e}", flush=True)

    # ---------------- GATE B: detector-rerun consistency (informational)
    print("\nGATE B (informational): independent re-detection vs get_video_pose",
          flush=True)
    from mimicmotion.dwpose.preprocess import get_video_pose
    video_pose, body_point, face_point = get_video_pose(
        video, image_pixels, sample_stride=task.sample_stride)
    n = min(len(body_point), len(detected))
    vis_diff, inv_diff, flips = [], [], 0
    for b, d in zip(body_point[:n], detected[:n]):
        c1, c2 = b["candidate"][:18], d["bodies"]["candidate"][:18]
        s1 = np.asarray(b["subset"])[0][:18] != -1
        s2 = np.asarray(d["bodies"]["subset"])[0][:18] != -1
        flips += int((s1 != s2).sum())
        both = s1 & s2
        if both.any():
            vis_diff.append(np.abs(c1[both] - c2[both]).max())
        if (~both).any():
            inv_diff.append(np.abs(c1[~both] - c2[~both]).max() if (~both).any() else 0)
    vmax = max(vis_diff) if vis_diff else 0.0
    imax = max(inv_diff) if inv_diff else 0.0
    print(f"  visible-kp maxdiff={vmax:.2e}  invisible-kp maxdiff={imax:.2e}  "
          f"visibility flips={flips}/{n*18}", flush=True)
    print("  (nonzero here = ONNX-GPU detector nondeterminism across independent "
          "runs; harmless for the pilot, which shares ONE detection pass)", flush=True)

    print("\nEQUIVALENCE GATE A:", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
