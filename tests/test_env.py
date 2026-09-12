import os
import sys

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from drone_landing.env import DroneLandingEnv


def test_gym_api():
    check_env(DroneLandingEnv(vision_mode="ground_truth"), skip_render_check=True)


def test_hover_is_stable():
    env = DroneLandingEnv(vision_mode="ground_truth")
    obs, _ = env.reset(seed=0)
    start = obs[:3].copy()
    for _ in range(100):
        obs, _, term, trunc, _ = env.step(np.zeros(4))
        assert not term and not trunc
    assert np.linalg.norm(obs[:3] - start) < 0.05


def test_pad_visible_from_all_starts():
    env = DroneLandingEnv(vision_mode="ground_truth")
    for seed in range(50):
        obs, info = env.reset(seed=seed)
        assert info["pad_visible"]
        assert np.linalg.norm(obs[16:19] - (env.pad_center - obs[:3])) < 1e-6


def test_camera_detection_matches_ground_truth():
    try:
        env = DroneLandingEnv(vision_mode="camera")
        obs, info = env.reset(seed=3)
    except Exception as e:  # no OpenGL on this machine
        pytest.skip(f"camera rendering unavailable: {e}")
    assert info["pad_visible"]
    assert env.get_camera_image().shape == (96, 96, 3)
    assert np.linalg.norm(obs[16:19] - (env.pad_center - obs[:3])) < 0.15


def test_free_fall_crashes():
    env = DroneLandingEnv(vision_mode="ground_truth")
    env.reset(seed=0)
    for _ in range(env.max_steps):
        _, _, term, _, info = env.step(-np.ones(4))
        if term:
            break
    assert info["outcome"] == "crashed"
