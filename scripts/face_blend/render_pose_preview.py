"""Render pose-condition previews for the head/face blend-ratio sweep.

DWPose is extracted ONCE for the ref image + driving video; each
(head_blend_ratio, face_blend_ratio) combo then replays the rescale/graft/
blend/draw chain (pure numpy+cv2, seconds per combo) -- no SVD generation.
Output: one skeleton mp4 per combo + a labeled grid mp4 (source video panel
first) for eyeballing how much expression comes through before spending
34min/clip on real generation.

Run on a GPU node (onnxruntime-gpu for DWPose). Usage:
    python scripts/face_blend/render_pose_preview.py \
        --video assets/example_data/sign_videos/5ok8y3eheq8_7-1-rgb_front_8s.mp4 \
        --ref assets/example_data/sign_videos/refs/test2.jpg \
        --out outputs/face_blend/preview
"""
import argparse
import copy
import os
import sys

import cv2
import decord
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.getcwd())

from mimicmotion.dwpose.util import draw_pose  # noqa: E402
from mimicmotion.dwpose.graft import (  # noqa: E402
    graft_pose_v2, blend_head_pose_only, blend_face_expression)

HEAD_RATIOS = [0.15, 0.3]
FACE_RATIOS = [0.0, 0.2, 0.4, 0.6, 0.8]
CELL_W = 288  # grid cell width, px


def replay_combo(detected_poses, ref_pose, a, b, h, w, head_r, face_r):
    """Replay the get_video_pose per-frame chain for one ratio combo."""
    frames = []
    for pose in detected_poses:
        dp = copy.deepcopy(pose)
        dp['bodies']['candidate'] = dp['bodies']['candidate'] * a + b
        dp['faces'] = dp['faces'] * a + b
        dp['hands'] = dp['hands'] * a + b
        if dp['bodies']['candidate'].shape[0] == 18:
            video_t = copy.deepcopy(dp)
            dp = graft_pose_v2(ref_pose, dp)
            dp = blend_head_pose_only(dp, video_t, blend_ratio=head_r)
            dp = blend_face_expression(dp, video_t, ref_pose, blend_ratio=face_r)
        # draw_pose returns CHW RGB; back to HWC for cv2
        frames.append(np.array(draw_pose(dp, h, w)).transpose(1, 2, 0))
    return frames  # list of RGB uint8 (h, w, 3)


def write_mp4(path, frames_bgr, fps):
    hh, ww = frames_bgr[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (ww, hh))
    for f in frames_bgr:
        vw.write(f)
    vw.release()


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sample_stride', type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # import here so --help works without GPU
    from mimicmotion.dwpose.dwpose_detector import dwpose_detector as dwprocessor

    ref_bgr = cv2.imread(args.ref)
    ref_image = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    h, w = ref_image.shape[:2]

    # --- mirror get_video_pose: ref pose + rescale fit (preprocess.py) ---
    ref_pose = dwprocessor(ref_image)
    ref_keypoint_id = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    ref_keypoint_id = [i for i in ref_keypoint_id
                       if len(ref_pose['bodies']['subset']) > 0
                       and ref_pose['bodies']['subset'][0][i] >= .0]
    ref_body = ref_pose['bodies']['candidate'][ref_keypoint_id]

    vr = decord.VideoReader(args.video, ctx=decord.cpu(0))
    stride = args.sample_stride * max(1, int(vr.get_avg_fps() / 24))
    idxs = list(range(0, len(vr), stride))
    frames = vr.get_batch(idxs).asnumpy()
    fps = vr.get_avg_fps() / stride
    detected_poses = [dwprocessor(frm) for frm in tqdm(frames, desc='DWPose')]
    dwprocessor.release_memory()

    detected_bodies = np.stack(
        [p['bodies']['candidate'] for p in detected_poses
         if p['bodies']['candidate'].shape[0] == 18])[:, ref_keypoint_id]
    ay, by = np.polyfit(detected_bodies[:, :, 1].flatten(),
                        np.tile(ref_body[:, 1], len(detected_bodies)), 1)
    fh, fw = frames[0].shape[:2]
    ax = ay / (fh / fw / h * w)
    bx = np.mean(np.tile(ref_body[:, 0], len(detected_bodies))
                 - detected_bodies[:, :, 0].flatten() * ax)
    a, b = np.array([ax, ay]), np.array([bx, by])

    clip = os.path.splitext(os.path.basename(args.video))[0][:12]
    cell_h = int(round(h / w * CELL_W))

    # source panel (driver video), resized to cell size
    grid_cells = []
    src_cells = [label(cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2BGR),
                                  (CELL_W, cell_h)), 'source')
                 for f in frames]
    grid_cells.append(src_cells)

    for head_r in HEAD_RATIOS:
        for face_r in FACE_RATIOS:
            tag = f'h{head_r:g}_f{face_r:g}'
            print(f'combo {tag} ...', flush=True)
            rgb = replay_combo(detected_poses, ref_pose, a, b, h, w,
                               head_r, face_r)
            bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in rgb]
            write_mp4(os.path.join(args.out, f'{clip}_{tag}.mp4'), bgr, fps)
            grid_cells.append([label(cv2.resize(f, (CELL_W, cell_h)), tag)
                               for f in bgr])

    # grid: 11 panels -> 4 cols x 3 rows (pad with black)
    n = len(grid_cells)
    cols, rows = 4, int(np.ceil(n / 4))
    blank = np.zeros((cell_h, CELL_W, 3), np.uint8)
    T = len(frames)
    grid_frames = []
    for t in range(T):
        cells = [grid_cells[i][t] if i < n else blank for i in range(cols * rows)]
        rows_img = [np.hstack(cells[r * cols:(r + 1) * cols]) for r in range(rows)]
        grid_frames.append(np.vstack(rows_img))
    write_mp4(os.path.join(args.out, f'{clip}_blend_grid.mp4'), grid_frames, fps)
    print(f'done: {n - 1} combos + grid -> {args.out}')


if __name__ == '__main__':
    main()
