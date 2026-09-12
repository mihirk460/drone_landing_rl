"""Smoke-test the install: load the scene, render the onboard camera, detect the H, hover for 2 s.

    python scripts/check_setup.py          # writes outputs/camera_sample.png and outputs/chase_sample.png
"""
import _gl  # noqa: F401

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from drone_landing.env import DroneLandingEnv
from drone_landing.vision import find_h_centroid


def main():
    out = Path(__file__).resolve().parents[1] / "outputs"
    out.mkdir(exist_ok=True)
    env = DroneLandingEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    print(f"drone pos {np.round(obs[:3], 2)}  pad centre {np.round(env.pad_center, 2)}")
    print(f"pad visible: {info['pad_visible']}   camera estimate of pad rel. position: {np.round(obs[16:19], 2)}"
          f"   true: {np.round(env.pad_center - obs[:3], 2)}")
    print(f"imu: {env.get_imu()}")

    cam = env.get_camera_image().copy()
    det = find_h_centroid(cam)
    if det:
        h, w = cam.shape[:2]
        cv2.drawMarker(cam, (int((det[0] + 1) / 2 * (w - 1)), int((1 - det[1]) / 2 * (h - 1))), (255, 0, 0), cv2.MARKER_CROSS, 12, 1)
    cv2.imwrite(str(out / "camera_sample.png"), cv2.cvtColor(cv2.resize(cam, (384, 384), interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "chase_sample.png"), cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR))

    t = time.time()
    for _ in range(100):
        obs, r, term, trunc, info = env.step(np.zeros(4))   # 0 % == hover
    print(f"after 2 s of hover: pos {np.round(obs[:3], 3)}  speed {info['speed']:.3f} m/s  outcome {info['outcome']}")
    print(f"env speed with camera: {100 / (time.time() - t):.0f} steps/s")
    print(f"wrote {out / 'camera_sample.png'} and {out / 'chase_sample.png'}")
    env.close()


if __name__ == "__main__":
    main()
