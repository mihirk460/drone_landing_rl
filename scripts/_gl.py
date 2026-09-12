"""Pick a headless OpenGL backend before mujoco is imported. Import this first in every script."""
import os
import sys

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")   # override with MUJOCO_GL=osmesa if EGL is unavailable
