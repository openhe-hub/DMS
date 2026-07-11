"""Step 1 / Probe A (training-free) -- does motion-field trajectory quality even
propagate to the generated video?

Generate the SAME clip (fixed diffusion seed) while injecting increasing keypoint
jitter into ONLY the motion-field branch (CMP flow + traj_flow + point_adapter);
the drawn skeleton pose_pixels stays clean. If the outputs barely change as jitter
grows, the diffusion absorbs trajectory differences and improving the pose-level
motion field (the SIREN idea) cannot help the video -> the whole pose-level route
is gated off here, before spending data/training.
"""
import argparse
import os

import torch
from omegaconf import OmegaConf

import _paths  # noqa: F401
from _paths import OUT, REPO

import inference_ctrl as IC
from mimicmotion.utils.loader import create_ctrl_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO, "configs/test.yaml"))
    ap.add_argument("--case", type=int, default=1, help="test_case index (default video2)")
    ap.add_argument("--noises", type=float, nargs="+", default=[0, 4, 8, 16, 32])
    ap.add_argument("--max_frames", type=int, default=40)
    args = ap.parse_args()
    device = IC.device  # torch.device, not str (pipeline's randn_tensor needs .type)

    infer_config = OmegaConf.load(args.config)
    pipeline = create_ctrl_pipeline(infer_config, device)
    task = infer_config.test_case[args.case]
    probe_dir = os.path.join(OUT, "video_probe")
    os.makedirs(probe_dir, exist_ok=True)
    print(f"case={args.case} video={task.ref_video_path} noises={args.noises} "
          f"max_frames={args.max_frames} seed={task.seed}", flush=True)

    for sigma in args.noises:
        print(f"\n=== generate sigma={sigma} ===", flush=True)
        pose_pixels, image_pixels, cf, ci, point_list, dift, tf = IC.preprocess(
            task.ref_video_path, task.ref_image_path, infer_config.dift_model_path,
            resolution=task.resolution, sample_stride=task.sample_stride,
            kp_noise=float(sigma), kp_noise_seed=0, max_frames=args.max_frames)
        frames = IC.run_pipeline(pipeline, image_pixels, pose_pixels, cf, ci,
                                 point_list, dift, tf, device, task)
        torch.save(frames, os.path.join(probe_dir, f"frames_sigma{int(sigma)}.pt"))
        IC.save_to_mp4(frames, os.path.join(probe_dir, f"video_sigma{int(sigma)}.mp4"),
                       fps=task.fps)
        print(f"   saved frames_sigma{int(sigma)}.pt  shape={tuple(frames.shape)}", flush=True)


if __name__ == "__main__":
    main()
