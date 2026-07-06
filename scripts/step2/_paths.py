"""Shared path setup for step2 (video-level) scripts."""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

OUT = os.path.join(REPO, "outputs", "step2")
FIG_DIR = os.path.join(OUT, "fig")
STEP1_OUT = os.path.join(REPO, "outputs", "step1")
TRAJ_DIR = os.path.join(STEP1_OUT, "traj")          # reuse step1 trajectories
CKPT_DIR = os.path.join(STEP1_OUT, "ckpt")
for d in (OUT, FIG_DIR):
    os.makedirs(d, exist_ok=True)
