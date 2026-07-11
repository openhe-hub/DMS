"""Blocking regression for the hand_flow diffs (no Gate A generation before PASS).

Checks:
(a) K-generalized pose2track/points_to_flows with n_points=18 are bit-identical
    to the ORIGINAL hardcoded-18 implementations (frozen verbatim below).
(b) hand_flow with all hand keypoints invisible (conf below threshold) is
    inert: first-18 rows of pose2track match the body-only run exactly and the
    hand rows scatter zero flow.
(c) hand_flow with confident hands is LIVE at the control-tensor level
    (traj_flow changes; reported magnitude vs body for the saturation risk).
(d) smooth_hands: sigma=0 identity; conf==1 matches a plain Gaussian.

--mode synthetic (default): pure numpy, runs anywhere.
--mode real: shared-detection build_control(lite=True) comparison on a config
  case (GPU DWPose; run on jubail). The original scripts/step2/equiv_check.py
  must also still PASS -- run it separately (same sbatch).
"""
import argparse
import sys

import numpy as np

import _paths as P  # noqa: F401


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    return bool(ok)


# ---- frozen pre-diff reference implementations (verbatim copies) ----
def _orig_pose2track(points_list, height, width):
    track_points = np.zeros((18, len(points_list), 2))
    track_points_subsets = np.zeros((18, len(points_list), 1))
    for f in range(len(points_list)):
        candidates, subsets, scores = points_list[f]['candidate'], points_list[f]['subset'][0], points_list[f]['score']
        for i in range(18):
            if subsets[i] == -1:
                track_points_subsets[i][f] = -1
            else:
                track_points[i][f][0] = max(min(candidates[i][0] * width, width - 1), 0)
                track_points[i][f][1] = max(min(candidates[i][1] * height, height - 1), 0)
                track_points_subsets[i][f] = i
    return track_points, track_points_subsets


def _orig_points_to_flows(points_list, model_length, height, width):
    track_points, track_points_subsets = _orig_pose2track(points_list, height, width)
    input_drag = np.zeros((model_length - 1, height, width, 2))
    for splited_track, points_subset in zip(track_points, track_points_subsets):
        if len(splited_track) == 1:
            displacement_point = tuple([splited_track[0][0] + 1, splited_track[0][1] + 1])
            splited_track = tuple([splited_track[0], displacement_point])
        if len(splited_track) < model_length:
            splited_track = splited_track + [splited_track[-1]] * (model_length - len(splited_track))
        for i in range(model_length - 1):
            if points_subset[i] != -1:
                start_point = splited_track[i]
                end_point = splited_track[i + 1]
                input_drag[i][int(start_point[1])][int(start_point[0])][0] = end_point[0] - start_point[0]
                input_drag[i][int(start_point[1])][int(start_point[0])][1] = end_point[1] - start_point[1]
    return input_drag


def synth_body_list(rng, f=6, nums=1):
    out = []
    for _ in range(f):
        cand = rng.rand(nums * 18, 2)
        sub = np.arange(18, dtype=float)[None].repeat(nums, 0)
        sub[rng.rand(*sub.shape) < 0.2] = -1
        out.append(dict(candidate=cand, subset=sub, score=rng.rand(nums, 18)))
    return out


def synth_hand_list(rng, f=6, conf=0.9):
    return [dict(hands=rng.rand(2, 21, 2),
                 hands_score=np.full((2, 21), conf)) for _ in range(f)]


def run_synthetic():
    from mimicmotion.utils.utils import pose2track, points_to_flows
    from mimicmotion.dwpose.hand_control import (append_hands_to_point_list,
                                                 smooth_hands)
    rng = np.random.RandomState(0)
    H = W = 64
    ok = True

    # (a) frozen-reference equivalence, single- and multi-person lists
    for nums in (1, 2):
        bp = synth_body_list(rng, f=8, nums=nums)
        po, pso = _orig_pose2track(bp, H, W)
        pn, psn = pose2track(bp, H, W, n_points=18)
        d = np.abs(po - pn).max() + np.abs(pso - psn).max()
        ok &= check(f"(a) pose2track n_points=18 == original (nums={nums})",
                    d == 0.0, f"maxdiff={d:.2e}")
        fo = _orig_points_to_flows(bp, len(bp), H, W)
        fn = points_to_flows(bp, len(bp), H, W, n_points=18)
        d = np.abs(fo - fn).max()
        ok &= check(f"(a) points_to_flows n_points=18 == original (nums={nums})",
                    d == 0.0, f"maxdiff={d:.2e}")

    # (b) invisible hands are inert
    bp = synth_body_list(rng, f=8, nums=2)  # multi-person: person-0 isolation too
    hp_dead = synth_hand_list(rng, f=8, conf=0.05)
    fl, n_pts = append_hands_to_point_list(bp, hp_dead, conf_thr=0.3)
    ok &= check("(b) n_points math", n_pts == 18 + 42, f"n_points={n_pts}")
    po, pso = pose2track(bp, H, W, n_points=18)
    pa, psa = pose2track(fl, H, W, n_points=n_pts)
    d = np.abs(po - pa[:18]).max() + np.abs(pso - psa[:18]).max()
    ok &= check("(b) body rows unchanged under augmentation", d == 0.0,
                f"maxdiff={d:.2e}")
    ok &= check("(b) dead hand rows all invisible", (psa[18:] == -1).all())
    fo = points_to_flows(bp, len(bp), H, W, n_points=18)
    fa = points_to_flows(fl, len(fl), H, W, n_points=n_pts)
    d = np.abs(fo - fa).max()
    ok &= check("(b) traj_flow identical with dead hands", d == 0.0,
                f"maxdiff={d:.2e}")

    # (c) confident hands are live; tips variant; magnitude report
    hp_live = synth_hand_list(rng, f=8, conf=0.9)
    fl_live, n_live = append_hands_to_point_list(bp, hp_live, conf_thr=0.3)
    fa_live = points_to_flows(fl_live, len(fl_live), H, W, n_points=n_live)
    delta = np.abs(fa_live - fo)
    ok &= check("(c) traj_flow responds to confident hands", delta.max() > 0.0,
                f"maxdelta={delta.max():.3f}")
    _, psl = pose2track(fl_live, H, W, n_points=n_live)
    ok &= check("(c) live hand rows visible", (psl[18:] != -1).all())
    fl_tips, n_tips = append_hands_to_point_list(bp, hp_live, conf_thr=0.3,
                                                 kp_subset="tips")
    ok &= check("(c) tips variant n_points", n_tips == 18 + 12,
                f"n_points={n_tips}")
    body_mag = np.abs(fo[fo != 0]) if (fo != 0).any() else np.zeros(1)
    hand_mag = np.abs(fa_live - fo)
    hand_mag = hand_mag[hand_mag != 0]
    print(f"  [info] nonzero-flow magnitude: body median={np.median(body_mag):.2f} "
          f"hand median={np.median(hand_mag) if len(hand_mag) else 0:.2f} "
          f"(pre-blur px displacement; saturation watch)")

    # (d) smooth_hands invariants
    hp = synth_hand_list(rng, f=20, conf=1.0)
    same = smooth_hands(hp, sigma=0.0)
    ok &= check("(d) sigma=0 identity", same is hp)
    sm = smooth_hands(hp, sigma=1.5)
    Hst = np.stack([e["hands"] for e in hp])
    r = max(1, int(round(3 * 1.5)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / 1.5) ** 2)
    T = len(hp)
    num = np.zeros_like(Hst)
    den = np.zeros(T)
    for j, kw in enumerate(k):
        off = j - r
        src = slice(max(0, -off), T - max(0, off))
        dst = slice(max(0, off), T - max(0, -off))
        num[dst] += kw * Hst[src]
        den[dst] += kw
    ref = num / den[:, None, None, None]
    d = max(np.abs(sm[t]["hands"] - ref[t]).max() for t in range(T))
    ok &= check("(d) conf==1 matches plain Gaussian", d < 1e-9, f"maxdiff={d:.2e}")
    d = max(np.abs(sm[t]["hands_score"] - hp[t]["hands_score"]).max()
            for t in range(T))
    ok &= check("(d) scores passed through", d == 0.0)
    return ok


def run_real(config, case):
    """Shared-detection comparison through build_control(lite=True) on jubail."""
    import torch  # noqa: F401
    from omegaconf import OmegaConf
    import inference_ctrl as IC
    from lib_lowfps import detect_video_full, person0
    from mimicmotion.dwpose.hand_control import (append_hands_to_point_list,
                                                 person0_hands)

    infer_config = OmegaConf.load(config)
    task = infer_config.test_case[case]
    image_pixels, ref_img, h_t, w_t = IC.load_ref_image(
        task.ref_image_path, task.resolution)
    frames, detected, ref_pose = detect_video_full(
        task.ref_video_path, image_pixels, sample_stride=task.sample_stride)
    hand_list = [dict(zip(("hands", "hands_score"), person0_hands(d)))
                 for d in detected]
    detected = person0(detected)
    img = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))
    bp = [ref_pose["bodies"]] + [d["bodies"] for d in detected]
    ref_h, ref_s = person0_hands(ref_pose)
    hl = [dict(hands=ref_h, hands_score=ref_s)] + hand_list

    ok = True
    base = IC.build_control(img, ref_img, bp, None, h_t, w_t, lite=True)
    base18 = IC.build_control(img, ref_img, bp, None, h_t, w_t, lite=True,
                              n_points=18)
    for k in ("traj_flow", "mask", "mask_384", "controlnet_image"):
        d = float((base[k].float() - base18[k].float()).abs().max())
        ok &= check(f"(real) default vs n_points=18 {k}", d == 0.0,
                    f"maxdiff={d:.2e}")

    dead = [dict(hands=e["hands"], hands_score=np.zeros_like(e["hands_score"]))
            for e in hl]
    fl_dead, n_pts = append_hands_to_point_list(bp, dead)
    aug_dead = IC.build_control(img, ref_img, fl_dead, None, h_t, w_t,
                                lite=True, n_points=n_pts)
    for k in ("traj_flow", "controlnet_image"):
        d = float((base[k].float() - aug_dead[k].float()).abs().max())
        ok &= check(f"(real) dead-hands inert {k}", d == 0.0, f"maxdiff={d:.2e}")

    fl_live, n_pts = append_hands_to_point_list(bp, hl)
    aug = IC.build_control(img, ref_img, fl_live, None, h_t, w_t, lite=True,
                           n_points=n_pts)
    d = float((base["traj_flow"].float() - aug["traj_flow"].float()).abs().max())
    ok &= check("(real) live hands change traj_flow", d > 0.0, f"maxdelta={d:.3f}")
    tb = base["traj_flow"].float().abs()
    ta = aug["traj_flow"].float().abs()
    print(f"  [info] blurred traj_flow abs: body-only max={tb.max():.4f} "
          f"mean={tb.mean():.6f} | +hands max={ta.max():.4f} mean={ta.mean():.6f}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("synthetic", "real"), default="synthetic")
    ap.add_argument("--config", default="configs/test_sign_hard27k.yaml")
    ap.add_argument("--case", type=int, default=0)
    args = ap.parse_args()

    print(f"equiv_check_hands mode={args.mode}", flush=True)
    ok = run_synthetic()
    if args.mode == "real":
        print("\n-- real-detection checks --", flush=True)
        ok &= run_real(args.config, args.case)

    print("\nHAND_FLOW REGRESSION:", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
