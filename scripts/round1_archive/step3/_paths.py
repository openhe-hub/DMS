"""Shared path setup for step3 (sampling-time latent fusion) scripts."""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(REPO, "src")
STEP2_SCRIPTS = os.path.join(REPO, "scripts", "round1_archive", "step2")
for p in (SRC, REPO, STEP2_SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = os.path.join(REPO, "outputs", "step3")
FIG_DIR = os.path.join(OUT, "fig")
STEP2_OUT = os.path.join(REPO, "outputs", "step2")
STEP2_PILOT = os.path.join(STEP2_OUT, "pilot")      # gt.pt / detections.pkl / R011 runs
for d in (OUT, FIG_DIR):
    os.makedirs(d, exist_ok=True)
