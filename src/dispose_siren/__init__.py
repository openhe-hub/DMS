"""dispose_siren: learned continuous (FiLM-SIREN) keypoint-trajectory denoiser
and evaluation against finite-difference / fd+Gaussian baselines.

Step-1 goal (synthetic -> real DWPose transfer):
  Does an amortized, *scale-invariant* learned INR prior recover a cleaner
  keypoint motion field than DisPose's fd+Gaussian on REAL DWPose trajectories?

No ground-truth velocity exists for real video, so we use two honest proxies:
  - held-out frame reconstruction (neutral)
  - high-fps finite-diff pseudo-GT velocity (favors fd-like methods; reported with caveat)
"""

N_FRAMES = 16          # DisPose per-window keypoint sampling
DENSE_T = 200          # dense evaluation grid resolution

__all__ = ["N_FRAMES", "DENSE_T"]
