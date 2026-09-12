"""Evaluate a trained policy.

    python scripts/evaluate.py --model models/v1/best_model.zip --episodes 20
    python scripts/evaluate.py --model models/v1/best_model.zip --video outputs/landing.mp4
    python scripts/evaluate.py --model models/v1/best_model.zip --render      # interactive MuJoCo viewer
"""
import _gl  # noqa: F401

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from stable_baselines3 import PPO

from drone_landing.env import DroneLandingEnv
from drone_landing.vision import find_h_centroid


def overlay_camera(frame, cam, visible):
    """Paste the onboard camera (with detected centroid) into the corner of the chase-cam frame."""
    cam = cam.copy()
    det = find_h_centroid(cam)
    if det is not None:
        h, w = cam.shape[:2]
        cx = int((det[0] + 1) / 2 * (w - 1)); cy = int((1 - det[1]) / 2 * (h - 1))
        cv2.drawMarker(cam, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 10, 1)
    cam = cv2.resize(cam, (160, 160), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(cam, (0, 0), (159, 159), (0, 255, 0) if visible else (255, 0, 0), 2)
    frame[10:170, frame.shape[1] - 170 : frame.shape[1] - 10] = cam
    return frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--vision", choices=["camera", "ground_truth"], default="camera")
    p.add_argument("--video", type=Path, default=None, help="write an mp4 of the episodes")
    p.add_argument("--render", action="store_true", help="open the interactive viewer (needs a display)")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of taking the mean")
    args = p.parse_args()

    render_mode = "human" if args.render else ("rgb_array" if args.video else None)
    env = DroneLandingEnv(vision_mode=args.vision, render_mode=render_mode)
    policy = PPO.load(args.model, device="cpu")
    writer = None
    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), 50, (640, 480))

    outcomes, returns, lengths = Counter(), [], []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        ret, done = 0.0, False
        while not done:
            action, _ = policy.predict(obs, deterministic=not args.stochastic)
            obs, r, term, trunc, info = env.step(action)
            ret += r
            done = term or trunc
            if writer is not None:
                frame = env.render()
                if env.get_camera_image() is not None:
                    frame = overlay_camera(frame, env.get_camera_image(), info["pad_visible"])
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if args.render:
                env.render(); time.sleep(env.dt)
        outcomes[info["outcome"]] += 1
        returns.append(ret); lengths.append(env._step_count)
        print(f"ep {ep:3d}  outcome={info['outcome']:<14} return={ret:8.2f}  steps={env._step_count:4d}  "
              f"final xy_err={info['xy_dist']:.2f} m  speed={info['speed']:.2f} m/s  tilt={info['tilt_deg']:.1f} deg")

    print("\noutcomes:", dict(outcomes))
    print(f"mean return {np.mean(returns):.2f} +- {np.std(returns):.2f}   mean length {np.mean(lengths):.0f} steps")
    if writer is not None:
        writer.release(); print(f"wrote {args.video}")
    env.close()


if __name__ == "__main__":
    main()
