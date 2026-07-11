"""Shared path setup for step1 scripts: put repo-root/src on sys.path."""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if REPO not in sys.path:
    sys.path.insert(0, REPO)   # so `import mimicmotion...` resolves on the cluster

OUT = os.path.join(REPO, "outputs", "step1")
TRAJ_DIR = os.path.join(OUT, "traj")
CKPT_DIR = os.path.join(OUT, "ckpt")
FIG_DIR = os.path.join(OUT, "fig")
for d in (TRAJ_DIR, CKPT_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)
