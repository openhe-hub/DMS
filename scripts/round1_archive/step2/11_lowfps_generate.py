"""R011 pilot -- self-driven low-fps generation.

Self-driven protocol: reference image = FIRST FRAME of the driving video, so
the original video frames on the sampling grid are pixel-level ground truth.
Systems only differ in how the control signal is built from every `stride`-th
detection:

  orig    : DisPose unchanged on the coarse grid -> short low-fps video
            (stride=1 -> the full-fps upper bound / original DisPose)
  linear / spline / siren : reconstruct keypoints on the FULL grid, redraw
            skeletons, rebuild motion field -> full-fps video

Everything shares one DWPose pass, one DIFT pass, one CMP model, one diffusion
seed -> exactly paired.

Outputs per case -> outputs/step2/pilot/case{K}/:
  gt.pt (T,3,H,W uint8), {method}_s{stride}.pt + .mp4, meta.json
"""
import argparse
import json
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

import _paths  # noqa: F401
from _paths import OUT, REPO

import inference_ctrl as IC
from lib_lowfps import (apply_cmp, compute_dift, detect_video_full,
                        draw_sequence, frames_to_target, interp_pose_dicts,
                        person0)


def get_self_ref(video_path, refs_dir):
    """Extract + cache the first frame of the video as the reference image."""
    import decord
    os.makedirs(refs_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(video_path))[0]
    p = os.path.join(refs_dir, f"{name}_ref.png")
    if not os.path.exists(p):
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        Image.fromarray(vr[0].asnumpy()).save(p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO, "configs/test.yaml"))
    ap.add_argument("--case", type=int, required=True)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="method:stride pairs, e.g. orig:1 orig:4 linear:4 spline:8 siren:4")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--siren_w0", type=float, default=3.0)
    ap.add_argument("--siren_lam", type=float, default=0.0)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()
    device = IC.device

    infer_config = OmegaConf.load(args.config)
    task = infer_config.test_case[args.case]
    video = task.ref_video_path
    case_dir = os.path.join(OUT, "pilot", f"case{args.case}")
    os.makedirs(case_dir, exist_ok=True)

    ref_png = get_self_ref(video, os.path.join(OUT, "refs"))
    image_pixels, ref_img, h_t, w_t = IC.load_ref_image(ref_png, task.resolution)
    print(f"case={args.case} video={video} self-ref={ref_png} target={h_t}x{w_t}",
          flush=True)

    # ---- one shared detection pass (+ GT dump), CACHED on disk so job
    # restarts (e.g. OOM resume with --skip_existing) reuse the exact same
    # detections -- the ONNX-GPU detector is nondeterministic across runs and
    # would otherwise break the paired-comparison guarantee.
    import pickle
    gt_path = os.path.join(case_dir, "gt.pt")
    det_path = os.path.join(case_dir, "detections.pkl")
    if os.path.exists(det_path) and os.path.exists(gt_path):
        with open(det_path, "rb") as f:
            detected, ref_pose = pickle.load(f)
        print(f"loaded cached detections ({det_path})", flush=True)
    else:
        frames, detected, ref_pose = detect_video_full(
            video, image_pixels, sample_stride=task.sample_stride,
            max_frames=args.max_frames)
        detected = person0(detected)
        if not os.path.exists(gt_path):
            torch.save(frames_to_target(frames, h_t, w_t), gt_path)
        with open(det_path, "wb") as f:
            pickle.dump((detected, ref_pose), f)
    T = len(detected)
    from mimicmotion.dwpose.util import draw_pose
    image_pose = np.array(draw_pose(ref_pose, h_t, w_t))
    img_1x3hw = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))
    print(f"detected T={T} frames", flush=True)

    # ---- shared heavy modules
    dift_feats = compute_dift(img_1x3hw, infer_config.dift_model_path, device)
    from mimicmotion.modules.cmp_model import CMP
    cmp = CMP('./mimicmotion/modules/cmp/experiments/semiauto_annot/'
              'resnet50_vip+mpii_liteflow/config.yaml', 42000).to(device)
    cmp.requires_grad_(False)
    from mimicmotion.utils.loader import create_ctrl_pipeline
    pipeline = create_ctrl_pipeline(infer_config, device)

    meta = {"video": video, "T": T, "h": h_t, "w": w_t, "fps": int(task.fps),
            "seed": int(task.seed), "runs": {}}
    mpath = os.path.join(case_dir, "meta.json")
    if os.path.exists(mpath):
        meta = json.load(open(mpath))

    for run in args.runs:
        method, stride = run.split(":")
        stride = int(stride)
        tag = f"{method}_s{stride}"
        fpt = os.path.join(case_dir, f"{tag}.pt")
        if args.skip_existing and os.path.exists(fpt):
            print(f"skip existing {tag}", flush=True)
            continue
        print(f"\n=== {tag} ===", flush=True)

        if method == "orig":
            dicts = detected[::stride]
            t_eval = np.arange(0, T, stride)
        else:
            cfg = {"w0": args.siren_w0, "lam": args.siren_lam, "steps": 800}
            dicts, t_eval, _ = interp_pose_dicts(detected, stride, method,
                                                 siren_cfg=cfg, verbose=True)
        pose_imgs = draw_sequence(dicts, h_t, w_t)
        pose_pixels = np.concatenate([image_pose[None], pose_imgs])
        body_point_list = [ref_pose["bodies"]] + [d["bodies"] for d in dicts]

        lite = IC.build_control(img_1x3hw, ref_img, body_point_list,
                                infer_config.dift_model_path, h_t, w_t, lite=True)
        controlnet_flow = apply_cmp(cmp, lite, device)

        pose_t = torch.from_numpy(pose_pixels.copy()) / 127.5 - 1
        img_t = torch.from_numpy(img_1x3hw) / 127.5 - 1
        out = IC.run_pipeline(pipeline, img_t, pose_t, controlnet_flow,
                              lite["controlnet_image"], body_point_list,
                              dift_feats, lite["traj_flow"], device, task)
        torch.save(out, fpt)
        fps_eff = task.fps / stride if method == "orig" else task.fps
        IC.save_to_mp4(out, os.path.join(case_dir, f"{tag}.mp4"), fps=fps_eff)
        meta["runs"][tag] = {"frames": int(out.shape[0]), "fps": float(fps_eff),
                             "grid": [int(t) for t in t_eval]}
        json.dump(meta, open(mpath, "w"), indent=2)
        print(f"saved {tag}: {tuple(out.shape)}", flush=True)

        # 40GB cards barely fit one full-fps run; scrub between runs
        import gc
        del out, controlnet_flow, lite, pose_t, img_t, pose_pixels, pose_imgs
        gc.collect()
        torch.cuda.empty_cache()

    print("\nALL RUNS DONE", flush=True)


if __name__ == "__main__":
    main()
