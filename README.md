# DisPose × MimicMotion × SIREN

A research fork for controllable human video generation that combines three components:

1. **MimicMotion** — the video-diffusion animation backbone ([repo](https://github.com/Tencent/MimicMotion)).
2. **DisPose** — training-free pose-guidance disentanglement layered on that backbone, turning a sparse skeleton into motion-field guidance + keypoint correspondence ([paper](https://arxiv.org/abs/2412.09349), arXiv:2412.09349).
3. **SIREN / INR motion representation** — an exploration that grafts continuous, periodic-activation implicit neural representations ([SIREN paper](https://arxiv.org/abs/2006.09661), arXiv:2006.09661) onto DisPose's pose-trajectory / motion-field control signal. Code in `src/dispose_siren/` and `scripts/step{1,2,3}/`; motivation and pre-registered findings in `docs/idea/` and `docs/experiments/`.

Downstream evaluations live under `docs/experiments/` and are interchangeable — e.g. a sign- and hand-fidelity stress test (`docs/experiments/baseline/`, with `qualitative.md` + `quantitative.md`) on ASL50K is one such experiment and can be swapped for other downstream datasets.

> **Primary references:** DisPose (Li et al., ICLR 2025) and SIREN (Sitzmann et al., NeurIPS 2020) — full BibTeX under [Citation](#-citation).

## Repository layout

| path | contents |
|---|---|
| `mimicmotion/` | animation backbone + DisPose control + DWPose (incl. the graft pose-retarget module) |
| `src/dispose_siren/` | SIREN / INR motion-representation exploration |
| `src/metrics/` | quantitative evaluation library (pose fidelity, hand confidence, CSIM, FVD) |
| `scripts/step{1,2,3}/`, `scripts/slurm/` | experiment pipelines + cluster launchers |
| `docs/idea/`, `docs/experiments/` | motivation, novelty checks, and experiment records |
| `docs/dispose_upstream_readme.md` | **base DisPose setup, model weights, and inference usage** |

For installing the environment, downloading model weights, and running inference, see the
base method's instructions in [`docs/dispose_upstream_readme.md`](docs/dispose_upstream_readme.md).

## 🔍 Citation

Primary references — DisPose (Li et al., ICLR 2025) and SIREN (Sitzmann et al., NeurIPS 2020):

```
@inproceedings{
li2025dispose,
title={DisPose: Disentangling Pose Guidance for Controllable Human Image Animation},
author={Hongxiang Li and Yaowei Li and Yuhang Yang and Junjie Cao and Zhihong Zhu and Xuxin Cheng and Long Chen},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=AumOa10MKG}
}

@inproceedings{sitzmann2020siren,
title={Implicit Neural Representations with Periodic Activation Functions},
author={Vincent Sitzmann and Julien N. P. Martel and Alexander W. Bergman and David B. Lindell and Gordon Wetzstein},
booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
year={2020},
url={https://arxiv.org/abs/2006.09661}
}
```
