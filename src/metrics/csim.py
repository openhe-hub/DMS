"""CSIM -- identity similarity of generated faces to the reference avatar.

ArcFace (insightface buffalo_l: det_10g detector + w600k_r50 recogniser) embeds
the largest face in each sampled frame; CSIM is the cosine similarity to the
reference image's embedding. Both models are conditioned on the same reference
(test2.jpg, md5-identical on both clusters), so this measures how faithfully /
stably each model preserves that identity through the diffusion.
"""
import numpy as np

# torch/onnxruntime before decord (OpenMP init order, see pose_extract).
import onnxruntime  # noqa: F401
import decord
from insightface.app import FaceAnalysis

_APP = None


def _app(ctx_id=0, det_size=320):
    global _APP
    if _APP is None:
        a = FaceAnalysis(name="buffalo_l",
                         allowed_modules=["detection", "recognition"],
                         providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        # faces are large + centred here, so 320 detects fine and is ~4x cheaper
        a.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
        _APP = a
    return _APP


def _largest_face_emb(bgr):
    faces = _app().get(bgr)
    if not faces:
        return None
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    return f.normed_embedding


def ref_embedding(ref_path):
    import cv2
    bgr = cv2.imread(ref_path)
    emb = _largest_face_emb(bgr)
    if emb is None:
        raise RuntimeError(f"no face in reference image {ref_path}")
    return emb


def video_csim(video_path, ref_emb, n_frames=12):
    """Identity is stable within a clip, so a fixed set of evenly-spaced frames
    (default 12) gives the same CSIM estimate as dense sampling at a fraction of
    the face-detection cost."""
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    n = len(vr)
    idxs = sorted(set(np.linspace(0, n - 1, min(n_frames, n)).astype(int).tolist()))
    frames = vr.get_batch(idxs).asnumpy()  # RGB
    sims, n_face = [], 0
    for rgb in frames:
        emb = _largest_face_emb(rgb[:, :, ::-1])  # -> BGR for insightface
        if emb is None:
            continue
        n_face += 1
        sims.append(float(np.dot(emb, ref_emb)))
    n = len(idxs)
    if not sims:
        return dict(n_sampled=n, face_det_rate=0.0, csim_mean=0.0,
                    csim_min=0.0, csim_std=0.0, csim_all=0.0)
    sims = np.array(sims)
    return dict(
        n_sampled=n,
        face_det_rate=n_face / n,
        csim_mean=float(sims.mean()),      # over face-detected frames
        csim_min=float(sims.min()),        # worst-frame identity
        csim_std=float(sims.std()),        # identity stability
        csim_all=float(sims.sum() / n),    # undetected frames count as 0
    )
