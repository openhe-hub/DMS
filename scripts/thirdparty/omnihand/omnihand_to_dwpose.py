"""Convert OmniHands trajectories (traj.npz) to DWPose-format 2D hand keypoints.

Bridges the OmniHands 3D recovery back into the DisPose pose chain: regresses
21 MANO joints per hand (OpenPose/COCO-WholeBody ordering, identical to what
DWPose emits for slots 92-133), perspective-projects them with the exact
camera the OmniHands renderer used (verified visually against the mesh
overlays), and writes the hand_override npz format consumed by
mimicmotion/dwpose/preprocess.py via `hand_recon_dir`:

  hands       [T,2,21,2]  normalized (x/W, y/H); [:,0]=left, [:,1]=right
  hands_score [T,2,21]    constant RECON_CONF (0.61, same convention as the
                          SIREN arm: marks "model-recovered" frames)
  covered     [T,2]       True where OmniHands produced a hand

Joint regression mirrors hands_4d/models/mano_wrapper.py exactly:
16 J_regressor joints + 5 fingertip vertices, reordered by mano_to_openpose.
The right-hand J_regressor is valid for the left hand too — left verts are a
mirrored right-hand mesh (same topology) and the regressor is linear.

Run on jubail in the `omhand` env (needs torch/smplx/cv2; CPU is fine):

  python omnihand_to_dwpose.py \
      --traj demo_out_smooth/<vname>/traj.npz --video <video.mp4> \
      --mano _DATA/data/mano --out kps_out/<vname>.npz \
      --overlay kps_out/kps_<vname>.mp4
"""
import argparse
import os

import cv2
import numpy as np

RECON_CONF = 0.61
MANO_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
              (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
# match the mesh-overlay videos: left purple, right cyan (BGR)
LEFT_COLOR, RIGHT_COLOR = (255, 61, 148), (255, 229, 102)


def load_regressor(mano_dir):
    """J_regressor [16,778] + fingertip vertex ids, same as mano_wrapper.py."""
    import torch  # noqa: F401  (smplx needs it)
    import smplx
    from smplx.vertex_ids import vertex_ids

    layer = smplx.MANOLayer(model_path=mano_dir, is_rhand=True, use_pca=False)
    j_reg = layer.J_regressor.detach().cpu().numpy()
    tips = np.array(list(vertex_ids['mano'].values()), dtype=np.int64)
    return j_reg, tips


def verts_to_joints2d(verts, cam_t, j_reg, tips, focal, cx, cy):
    """[T,778,3] verts + [T,3] cam_t -> [T,21,2] pixel keypoints (OpenPose order)."""
    joints = np.einsum('jv,tvc->tjc', j_reg, verts)          # [T,16,3]
    joints = np.concatenate([joints, verts[:, tips]], axis=1)  # [T,21,3]
    joints = joints[:, MANO_TO_OPENPOSE]
    p = joints + cam_t[:, None, :]
    return np.stack([focal * p[..., 0] / p[..., 2] + cx,
                     focal * p[..., 1] / p[..., 2] + cy], axis=-1)


def draw_hand(img, kps, color):
    for a, b in HAND_EDGES:
        cv2.line(img, tuple(np.round(kps[a]).astype(int)),
                 tuple(np.round(kps[b]).astype(int)), color, 2, cv2.LINE_AA)
    for p in kps:
        cv2.circle(img, tuple(np.round(p).astype(int)), 3, (255, 255, 255), -1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traj', required=True)
    ap.add_argument('--video', required=True)
    ap.add_argument('--mano', required=True, help='dir containing MANO_RIGHT.pkl')
    ap.add_argument('--out', required=True)
    ap.add_argument('--overlay', default=None, help='optional overlay mp4 path')
    ap.add_argument('--which', default='sm', choices=['sm', 'raw'],
                    help='use smoothed (sm) or raw trajectories')
    ap.add_argument('--focal', type=float, default=5000.0, help='EXTRA.FOCAL_LENGTH')
    ap.add_argument('--image-size', type=float, default=256.0, help='MODEL.IMAGE_SIZE')
    args = ap.parse_args()

    d = np.load(args.traj)
    w = args.which
    verts_r, verts_l = d[f'{w}_vr'], d[f'{w}_vl']
    cam_r, cam_l = d[f'{w}_cr'], d[f'{w}_cl']
    T = len(verts_r)

    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if abs(nframes - T) > 1:
        print(f'WARNING: video has {nframes} frames but traj has {T}')

    # same intrinsics as the OmniHands renderer (run_demo.py / renderer.py)
    focal = args.focal / args.image_size * max(W, H)
    j_reg, tips = load_regressor(args.mano)
    kps_r = verts_to_joints2d(verts_r, cam_r, j_reg, tips, focal, W / 2.0, H / 2.0)
    kps_l = verts_to_joints2d(verts_l, cam_l, j_reg, tips, focal, W / 2.0, H / 2.0)

    hands = np.stack([kps_l, kps_r], axis=1)  # [T,2,21,2], 0=left 1=right (DWPose slot order)
    in_frame = ((hands[..., 0] >= 0) & (hands[..., 0] < W)
                & (hands[..., 1] >= 0) & (hands[..., 1] < H)).mean()
    print(f'{os.path.basename(args.traj)}: T={T} size={W}x{H} focal={focal:.1f} '
          f'in-frame={100 * in_frame:.1f}%')

    hands_norm = hands / np.array([W, H], dtype=np.float64)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez_compressed(args.out,
                        hands=hands_norm.astype(np.float32),
                        hands_score=np.full((T, 2, 21), RECON_CONF, dtype=np.float32),
                        covered=np.ones((T, 2), dtype=bool),
                        hands_px=hands.astype(np.float32),
                        meta=np.array([W, H, focal, fps], dtype=np.float64))

    if args.overlay:
        os.makedirs(os.path.dirname(args.overlay) or '.', exist_ok=True)
        vw = cv2.VideoWriter(args.overlay, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
        for i in range(T):
            ok, frame = cap.read()
            if not ok:
                break
            draw_hand(frame, hands[i, 0], LEFT_COLOR)
            draw_hand(frame, hands[i, 1], RIGHT_COLOR)
            vw.write(frame)
        vw.release()
        print(f'overlay -> {args.overlay}')
    cap.release()


if __name__ == '__main__':
    main()
