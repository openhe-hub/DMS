"""Shared path setup for hand_pilot (hand control channel + NIAF-SIREN) scripts."""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO, "src")
STEP2_SCRIPTS = os.path.join(REPO, "scripts", "step2")
for p in (SRC, REPO, STEP2_SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

OUT = os.path.join(REPO, "outputs", "hand_pilot")
POSES_DIR = os.path.join(OUT, "poses")
GATE_B_DIR = os.path.join(OUT, "gate_b")
WINDOWS_DIR = os.path.join(OUT, "windows")
CKPT_DIR = os.path.join(OUT, "ckpt")
GATE_A_DIR = os.path.join(OUT, "gate_a")
FIG_DIR = os.path.join(OUT, "fig")
for d in (OUT, POSES_DIR, GATE_B_DIR, WINDOWS_DIR, CKPT_DIR, GATE_A_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)
