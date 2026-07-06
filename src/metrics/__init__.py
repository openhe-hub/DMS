"""Quantitative metrics for MimicMotion vs DisPose sign-language comparison.

All metrics reuse the *same* DWPose whole-body estimator used during inference
(`mimicmotion.dwpose`), so pose is measured with one consistent detector across
the source video and both models' outputs.

Metric 1 (hand_confidence): structural quality of generated hands, measured by
    DWPose per-keypoint confidence + hand detectability. No cross-subject
    alignment needed -> directly reflects "does the hand look like a hand".

Metric 2 (motion_fidelity): control adherence, measured as body- vs hand-
    decomposed PCK/NME between the generated pose and the driving source pose,
    each normalized into a subject-invariant frame (body: neck-centred /
    shoulder-width; hand: wrist-centred / wrist-MCP bone). The body/hand split
    is the story: whether DisPose's advantage concentrates on the hands.
"""
