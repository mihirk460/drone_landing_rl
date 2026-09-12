"""Reward function. Kept separate so it is easy to tune or extend."""
import numpy as np

from drone_landing.config import RewardConfig


def step_reward(cfg: RewardConfig, s: dict) -> tuple[float, dict]:
    """s is the state dict built by DroneLandingEnv._state(). Returns (reward, per-term breakdown)."""
    terms = {
        "progress": cfg.w_progress * (s["prev_dist"] - s["dist"]),
        "xy": -cfg.w_xy * s["xy_dist"],
        "z": -cfg.w_z * s["height"],
        "vel": -cfg.w_vel * s["speed"],
        "tilt": -cfg.w_tilt * s["tilt"],
        "action": -cfg.w_action * float(np.mean(s["action"] ** 2)),
        "lost": -cfg.w_lost * (0.0 if s["pad_visible"] else 1.0),
    }
    return float(sum(terms.values())), terms


def terminal_reward(cfg: RewardConfig, outcome: str, s: dict) -> float:
    if outcome == "landed":
        softness = 1.0 - 0.5 * min(s["speed"] / 0.5, 1.0)   # 1.0 for a gentle touchdown, 0.5 at the limit
        return cfg.R_landed * softness
    if outcome == "crashed":
        return cfg.R_crashed
    if outcome == "out_of_bounds":
        return cfg.R_out_of_bounds
    return 0.0
