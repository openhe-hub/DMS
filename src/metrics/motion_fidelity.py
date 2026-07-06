"""Metric 2 -- control adherence via body/hand-decomposed pose fidelity.

We compare the generated pose against the driving source pose, frame-aligned.
Because the source signer and the avatar are different people at different scale
and position, each skeleton is first mapped into a subject-invariant frame:

    body : translate to the neck, scale by shoulder width  -> arm/torso placement
    hand : translate to the wrist, scale by wrist->mid-MCP  -> finger articulation

Then per-keypoint PCK@0.2 and NME are computed over keypoints confidently
detected in *both* poses. Reporting body and hand separately is the point: it
shows whether the DisPose advantage concentrates on the (hardest) hands.

Note (honesty): DisPose is pose-conditioned, so higher fidelity to the source
pose is partly by design. Frame this as *control adherence*, not a neutral
comparison.
"""
import numpy as np

NECK, RSHO, LSHO = 1, 2, 5   # OpenPose-18 indices
WRIST, MID_MCP = 0, 9        # 21-point hand indices


def _norm_body(body, score, thr):
    if score[NECK] < thr or score[RSHO] < thr or score[LSHO] < thr:
        return None
    scale = np.linalg.norm(body[RSHO] - body[LSHO])
    if not np.isfinite(scale) or scale < 1e-6:
        return None
    return (body - body[NECK]) / scale


def _norm_hand(hand, score, thr):
    if score[WRIST] < thr or score[MID_MCP] < thr:
        return None
    scale = np.linalg.norm(hand[WRIST] - hand[MID_MCP])
    if not np.isfinite(scale) or scale < 1e-6:
        return None
    return (hand - hand[WRIST]) / scale


def _pck_nme(gen, src, gen_s, src_s, thr, pck_at):
    valid = (gen_s > thr) & (src_s > thr)
    if valid.sum() == 0:
        return None
    d = np.linalg.norm(gen[valid] - src[valid], axis=1)
    return float((d < pck_at).mean()), float(d.mean()), int(valid.sum())


def motion_fidelity_metrics(gen, src, gen_offset=0, thr=0.3, pck_at=0.2):
    """gen, src: dense pose dicts from pose_extract.

    gen_offset drops leading generated frames before index-aligning with the
    source (MimicMotion pads one reference frame at the front -> offset 1;
    DisPose is frame-aligned -> offset 0).
    """
    n = min(len(gen["detected"]) - gen_offset, len(src["detected"]))

    body_pck, body_nme, hand_pck, hand_nme = [], [], [], []
    for i in range(n):
        gi = i + gen_offset
        if not gen["detected"][gi] or not src["detected"][i]:
            continue
        gb = _norm_body(gen["body"][gi], gen["body_score"][gi], thr)
        sb = _norm_body(src["body"][i], src["body_score"][i], thr)
        if gb is not None and sb is not None:
            r = _pck_nme(gb, sb, gen["body_score"][gi], src["body_score"][i], thr, pck_at)
            if r:
                body_pck.append(r[0]); body_nme.append(r[1])
        for k in range(2):
            gh = _norm_hand(gen["hands"][gi, k], gen["hands_score"][gi, k], thr)
            sh = _norm_hand(src["hands"][i, k], src["hands_score"][i, k], thr)
            if gh is not None and sh is not None:
                r = _pck_nme(gh, sh, gen["hands_score"][gi, k],
                             src["hands_score"][i, k], thr, pck_at)
                if r:
                    hand_pck.append(r[0]); hand_nme.append(r[1])

    def _m(x):
        return float(np.mean(x)) if x else float("nan")

    return dict(
        body_pck=_m(body_pck), body_nme=_m(body_nme), body_frames=len(body_pck),
        hand_pck=_m(hand_pck), hand_nme=_m(hand_nme), hand_samples=len(hand_pck),
        aligned_frames=int(n),
    )
