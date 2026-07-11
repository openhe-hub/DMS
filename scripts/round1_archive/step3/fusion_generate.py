"""R101/R102 -- self-driven low-fps generation WITH sampling-time latent fusion.

Same protocol/pairing as step2 R011 (shared cached detections, shared DIFT,
same seed). Control = linear keypoint interpolation (R012: interp method is
irrelevant at video level); the new ingredient is the FusionCallback that
softly pulls mid-frame latents toward flow-warped keyframe latents at every
denoising step.

Reads shared caches from outputs/step2/pilot/case{K}/ (gt.pt, detections.pkl),
writes fusion outputs to outputs/step3/pilot/case{K}/.

Run tags: fusion_s{stride}            (frozen config)
          fusion_s{stride}_{devtag}   (--dev_tag set, R101 sweeps)
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch
from omegaconf import OmegaConf

import _paths  # noqa: F401
from _paths import OUT, REPO, STEP2_PILOT

import inference_ctrl as IC
from lib_lowfps import apply_cmp, compute_dift, draw_sequence, interp_pose_dicts
from lib_fusion import (FusionCallback, X0FusionScheduler, build_fusion_plan,
                        run_pipeline_cb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO, "configs/test.yaml"))
    ap.add_argument("--case", type=int, required=True)
    ap.add_argument("--strides", type=int, nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--win_lo", type=float, default=0.3)
    ap.add_argument("--win_hi", type=float, default=0.9)
    ap.add_argument("--sigma", type=float, default=6.0,
                    help="Gaussian splat sigma in latent pixels")
    ap.add_argument("--space", choices=["latent", "x0"], default="x0",
                    help="fuse noisy latents (round-1, failed) or x0 preds")
    ap.add_argument("--dev_tag", default=None,
                    help="suffix for R101 sweep runs, e.g. a30w0309")
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()
    device = IC.device

    infer_config = OmegaConf.load(args.config)
    task = infer_config.test_case[args.case]
    s2_dir = os.path.join(STEP2_PILOT, f"case{args.case}")
    case_dir = os.path.join(OUT, "pilot", f"case{args.case}")
    os.makedirs(case_dir, exist_ok=True)

    # ---- shared caches from step2 (MUST exist: pairing depends on them)
    det_path = os.path.join(s2_dir, "detections.pkl")
    assert os.path.exists(det_path), f"missing step2 cache {det_path}"
    with open(det_path, "rb") as f:
        detected, ref_pose = pickle.load(f)
    T = len(detected)

    name = os.path.splitext(os.path.basename(task.ref_video_path))[0]
    ref_png = os.path.join(os.path.dirname(STEP2_PILOT), "refs",
                           f"{name}_ref.png")
    assert os.path.exists(ref_png), f"missing step2 self-ref {ref_png}"
    image_pixels, ref_img, h_t, w_t = IC.load_ref_image(ref_png, task.resolution)
    h_lat, w_lat = h_t // 8, w_t // 8
    print(f"case={args.case} T={T} target={h_t}x{w_t} latent={h_lat}x{w_lat}",
          flush=True)

    from mimicmotion.dwpose.util import draw_pose
    image_pose = np.array(draw_pose(ref_pose, h_t, w_t))
    img_1x3hw = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))

    dift_feats = compute_dift(img_1x3hw, infer_config.dift_model_path, device)
    from mimicmotion.modules.cmp_model import CMP
    cmp = CMP('./mimicmotion/modules/cmp/experiments/semiauto_annot/'
              'resnet50_vip+mpii_liteflow/config.yaml', 42000).to(device)
    cmp.requires_grad_(False)
    from mimicmotion.utils.loader import create_ctrl_pipeline
    pipeline = create_ctrl_pipeline(infer_config, device)

    meta_path = os.path.join(case_dir, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else \
        {"video": task.ref_video_path, "T": T, "runs": {}}

    for stride in args.strides:
        tag = f"fusion_s{stride}"
        if args.dev_tag:
            tag += f"_{args.dev_tag}"
        fpt = os.path.join(case_dir, f"{tag}.pt")
        if args.skip_existing and os.path.exists(fpt):
            print(f"skip existing {tag}", flush=True)
            continue
        print(f"\n=== {tag} (alpha={args.alpha} win=[{args.win_lo},"
              f"{args.win_hi}] sigma={args.sigma}) ===", flush=True)

        dicts, t_eval, _ = interp_pose_dicts(detected, stride, "linear")
        pose_imgs = draw_sequence(dicts, h_t, w_t)
        pose_pixels = np.concatenate([image_pose[None], pose_imgs])
        body_point_list = [ref_pose["bodies"]] + [d["bodies"] for d in dicts]

        lite = IC.build_control(img_1x3hw, ref_img, body_point_list,
                                infer_config.dift_model_path, h_t, w_t,
                                lite=True)
        controlnet_flow = apply_cmp(cmp, lite, device)

        cand = np.stack([d["bodies"]["candidate"][:18] for d in dicts])
        sub = np.stack([np.asarray(d["bodies"]["subset"])[0][:18]
                        for d in dicts])
        obs_idx = np.arange(0, T, stride)
        plan = build_fusion_plan(cand, sub, obs_idx, h_lat, w_lat,
                                 sigma=args.sigma)
        print(f"fusion plan: {len(plan)} mid frames, "
              f"{task.num_inference_steps} steps, space={args.space}",
              flush=True)

        pose_t = torch.from_numpy(pose_pixels.copy()) / 127.5 - 1
        img_t = torch.from_numpy(img_1x3hw) / 127.5 - 1
        if args.space == "x0":
            orig_sched = pipeline.scheduler
            cb = X0FusionScheduler(orig_sched, plan, task.num_inference_steps,
                                   alpha=args.alpha,
                                   win=(args.win_lo, args.win_hi))
            pipeline.scheduler = cb
            try:
                out = run_pipeline_cb(pipeline, img_t, pose_t, controlnet_flow,
                                      lite["controlnet_image"],
                                      body_point_list, dift_feats,
                                      lite["traj_flow"], device, task,
                                      callback=None)
            finally:
                pipeline.scheduler = orig_sched
        else:
            cb = FusionCallback(plan, task.num_inference_steps,
                                alpha=args.alpha,
                                win=(args.win_lo, args.win_hi))
            out = run_pipeline_cb(pipeline, img_t, pose_t, controlnet_flow,
                                  lite["controlnet_image"], body_point_list,
                                  dift_feats, lite["traj_flow"], device, task,
                                  callback=cb)
        assert cb.calls > 0, "fusion never fired"
        torch.save(out, fpt)
        IC.save_to_mp4(out, os.path.join(case_dir, f"{tag}.mp4"),
                       fps=task.fps)
        meta["runs"][tag] = {"frames": int(out.shape[0]),
                             "alpha": args.alpha,
                             "win": [args.win_lo, args.win_hi],
                             "sigma": args.sigma, "space": args.space,
                             "cb_calls": int(cb.calls),
                             "grid": [int(t) for t in t_eval]}
        json.dump(meta, open(meta_path, "w"), indent=2)
        print(f"saved {tag}: {tuple(out.shape)} cb_calls={cb.calls}",
              flush=True)

        import gc
        del out, controlnet_flow, lite, pose_t, img_t, pose_pixels, pose_imgs
        gc.collect()
        torch.cuda.empty_cache()

    print("\nALL RUNS DONE", flush=True)


if __name__ == "__main__":
    main()
