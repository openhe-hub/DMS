# Quantitative-metric slurm jobs (sign-language DisPose vs MimicMotion)

Each metric runs as two independent jobs, one per model, on the cluster where
that model's outputs live. The detector weights (DWPose, I3D, ArcFace buffalo_l)
are byte-identical on both clusters (md5-verified), so separate jobs are
equivalent to one shared estimator; only the small CSV/JSON results are pulled
back and merged with `src/metrics/merge_results.py`.

| metric | DisPose side (jubail, `zl6890`) | MimicMotion side (jubail2, `yf23`) |
|---|---|---|
| hand-conf + motion-fidelity | `metrics_eval_dispose.slurm` | `metrics_eval_mimic.slurm` |
| CSIM (identity) | `csim_dispose.slurm` | `csim_mimic.slurm` |
| FVD (naturalness) | `fvd_dispose.slurm` | `fvd_mimic.slurm` |

Prereqs on each side:
- DWPose weights under the repo's `pretrained_weights/DWPose/` (dispose) /
  `models/DWPose/` (mimic fork).
- CSIM: `insightface` installed + `buffalo_l` pre-cached in `~/.insightface`
  (download on a login node; compute nodes are offline).
- FVD: StyleGAN-V `i3d_torchscript.pt` under `<scratch>/fvd_models/`.

The `*_dispose.slurm` scripts reference `/scratch/zl6890/...`; the `*_mimic.slurm`
scripts reference `/scratch/yf23/chatsign-175/MimicMotion/...`. Adjust the paths
if reproducing under different accounts.

See `docs/experiments/baseline/sign_cmp_quantitative.md` for the results.
