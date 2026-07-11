"""Real keypoint-trajectory extraction via DisPose's own DWPose.

Produces, per video, a dense (K=18, T, 2) pixel-coordinate trajectory plus a
(K, T) visibility mask, using the exact detector + rescale + pose2track path
DisPose uses at inference. Only imported on the cluster (needs mimicmotion +
onnxruntime + DWPose onnx weights).
"""
import numpy as np


def extract_dense_trajectory(video_path, ref_image_path, sample_stride=1):
    """Returns:
        points : (18, T, 2) float pixel coords (rescaled to ref-image frame)
        vis    : (18, T)    bool visibility (DWPose subset != -1)
        meta   : dict(h, w, T, video, stride)
    sample_stride=1 keeps the densest available sampling (effective stride is
    further multiplied by fps/24 inside get_video_pose) -> used for high-fps
    pseudo-GT; the N-frame DisPose window is sub-sampled later in eval.
    """
    from torchvision.datasets.folder import pil_loader   # heavy deps -> lazy
    from mimicmotion.dwpose.preprocess import get_video_pose
    from mimicmotion.utils.utils import pose2track

    ref = np.array(pil_loader(ref_image_path))            # (h,w,3) RGB
    h, w, _ = ref.shape
    _, body_point, _ = get_video_pose(video_path, ref, sample_stride=sample_stride)
    track, subset = pose2track(body_point, h, w)          # (18,T,2), (18,T,1)
    points = np.asarray(track, dtype=np.float32)          # (18,T,2)
    vis = (np.asarray(subset)[:, :, 0] != -1)             # (18,T)
    meta = dict(h=int(h), w=int(w), T=int(points.shape[1]),
                video=str(video_path), stride=int(sample_stride))
    return points, vis, meta


def save_npz(path, points, vis, meta):
    np.savez(path, points=points, vis=vis, meta=np.array([meta], dtype=object))


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return d["points"], d["vis"], d["meta"][0]
