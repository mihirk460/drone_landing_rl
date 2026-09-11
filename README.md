# drone_landing_rl

Reinforcement-learning quadrotor that lands on an "H" landing pad in MuJoCo, using a downward-facing
wide-angle camera to find the pad and ground-truth pose from the simulator. Version 1: xy alignment and a
slow, safe touchdown. Orientation (yaw) matching is deferred to v2.

## What is in the box

| Path | Purpose |
|---|---|
| `assets/drone_landing.xml` | MuJoCo scene: 1 kg X-quadrotor (4 thrust motors, IMU, down camera), 1.5 m "H" pad, ground |
| `assets/h_pad.png` | Pad texture (regenerate with `scripts/make_pad_texture.py`) |
| `drone_landing/env.py` | Gymnasium env `DroneLandingEnv` (action / observation / termination) |
| `drone_landing/vision.py` | H detection (threshold + largest dark blob) and pixel -> ground ray-cast |
| `drone_landing/rewards.py` | Reward function (step terms + terminal bonuses) |
| `drone_landing/config.py` | Every tunable number: env, reward weights, PPO hyper-parameters |
| `scripts/train.py` | PPO training (Stable-Baselines3), checkpoints, periodic eval, TensorBoard |
| `scripts/evaluate.py` | Run a saved policy, print outcomes, optional mp4 or interactive viewer |
| `scripts/check_setup.py` | Smoke test the install and dump a camera frame with the detected centroid |
| `tests/` | pytest sanity checks (gym API, hover stability, pad visibility, detection accuracy) |
| `models/` | Where trained weights go (see below) |

## Environment interface

**Action** `Box(-1, 1, (4,))`: throttle per motor, -1 = -100 % (motor off), 0 = hover, +1 = +100 % (max).
The env maps this linearly to thrust in Newtons; max thrust is 2x hover (thrust-to-weight 2.0).
Motor order: m1 front-right, m2 back-left, m3 front-left, m4 back-right (X configuration, CCW/CCW/CW/CW).
Info also reports `motor_percent` in -100..+100.

**Observation** (19 floats):

| Index | Content | Source |
|---|---|---|
| 0:3 | drone xyz, world | MuJoCo ground truth |
| 3:7 | orientation quaternion (w,x,y,z) | MuJoCo ground truth |
| 7:10 | linear velocity, world | MuJoCo ground truth |
| 10:13 | angular velocity, body | gyro sensor |
| 13 | pad visible (0/1) | camera |
| 14:16 | H centroid in normalised image coords (u right, v up) | camera |
| 16:19 | estimated pad centre relative to drone, world frame | camera centroid ray-cast onto ground plane |

The IMU (accelerometer + gyro at the body centre) is in the model and readable via `env.get_imu()`; the
accelerometer is not in the v1 observation.

**Camera**: `down_cam`, 96x96 RGB, 120 deg vertical FOV, mounted under the core, looking straight down.
`env.get_camera_image()` returns the last frame. The H is found by thresholding near-black pixels and
taking the largest connected blob; the centroid is ray-cast through the camera model onto the pad plane to
get a metric xyz estimate (error ~3 cm from 3 m up).

**Episode**: pad centre sampled in [-1, 1]^2 m, drone spawns level and hovering 2.5-3.5 m above the pad and
0.5-1.5 m to the side (pad always in view). Control at 50 Hz, max 12 s.
Ends with one of: `landed` (touched the pad within 0.35 m of centre, < 0.5 m/s, < 20 deg tilt),
`crashed` (any other touchdown, or tilt > 70 deg), `out_of_bounds` (> 4 m from pad or > 6 m up), `timeout`.

**Vision modes** (`vision_mode` in `EnvConfig`, `--vision` on the scripts):
- `camera` (default): render the onboard camera every step and detect the H. This is the real task.
- `ground_truth`: same observation layout, but the centroid is computed by projecting the true pad centre
  through the same camera model. No OpenGL, ~50x faster. Use it to debug reward / hyper-parameters
  quickly, then train the final policy in `camera` mode.

## Reward

Per step (weights in `RewardConfig`):

```
+ w_progress * (previous distance to pad - current distance)      approach
- w_xy * horizontal distance      - w_z * height above pad
- w_vel * speed                   - w_tilt * tilt angle
- w_action * mean(action^2)       - w_lost * (pad not visible)
```
Terminal: `landed` +100 scaled by touchdown softness (100 gentle, 50 at the speed limit), `crashed` -50,
`out_of_bounds` -50. Every term is logged per step in `info["reward_terms"]` so you can see what dominates.

## Algorithm

PPO (Stable-Baselines3, MLP policy 2x64). Small enough to train on a laptop CPU: the network is ~10k
parameters and memory use is dominated by the 8 parallel MuJoCo instances (< 2 GB total). No GPU needed.

## Install

```bash
git clone https://github.com/mihirk460/drone_landing_rl.git
cd drone_landing_rl
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: CPU-only torch, much smaller download
```

Headless rendering of the onboard camera:
- **Linux**: the scripts default to `MUJOCO_GL=egl` (works with any Mesa / NVIDIA driver, no display needed).
  If EGL fails, `export MUJOCO_GL=osmesa` (`apt install libosmesa6`). Note: OSMesa and PyTorch can crash
  each other with a segfault (LLVM symbol clash); if that happens use EGL, or train in `ground_truth` mode.
- **macOS / Windows**: leave `MUJOCO_GL` unset. Use `--vec dummy` on macOS if subprocess envs misbehave.

Verify:

```bash
python scripts/check_setup.py     # prints pose, detection, hover check; writes outputs/camera_sample.png
python -m pytest tests            # ~1 s
```

## Train locally

```bash
python scripts/train.py --run-name v1                     # 2M steps, 8 envs, camera mode
python scripts/train.py --run-name v1_gt --vision ground_truth   # fast debug run, no rendering
python scripts/train.py --run-name v1 --resume models/v1/final_model.zip --timesteps 1000000   # continue
tensorboard --logdir logs
```

Options: `--timesteps`, `--n-envs` (match your CPU core count), `--vec subproc|dummy`, `--seed`.
Hyper-parameters live in `TrainConfig`. Ctrl-C is safe: the current weights are saved on exit.

Expect roughly 150-300 env steps/s per core in camera mode on an integrated GPU (rendering bound) and
several thousand in ground-truth mode. 2M camera-mode steps on 8 cores is a few hours.

### Where the weights are

```
models/<run_name>/final_model.zip     weights at the end of training (or at Ctrl-C)
models/<run_name>/best_model.zip      best mean eval return seen during training
models/<run_name>/checkpoints/        snapshot every 100k steps (git-ignored)
models/<run_name>/config.json         exact env / reward / train config used
```

`final_model.zip` and `best_model.zip` are small (~100 kB) and are not git-ignored: commit the ones you want
to keep so they live in the repo, not just on one machine. If you also want them off-repo, copy them to
cloud storage; there is nothing else to back up.

## Evaluate

```bash
python scripts/evaluate.py --model models/v1/best_model.zip --episodes 20
python scripts/evaluate.py --model models/v1/best_model.zip --video outputs/landing.mp4   # chase cam + onboard inset
python scripts/evaluate.py --model models/v1/best_model.zip --render                      # interactive viewer (needs a display)
```

## Training on GitHub Actions

Not set up yet (pending decision). What it would look like: a manually triggered workflow that installs the
requirements, runs `train.py` with `MUJOCO_GL=egl` on `ubuntu-latest` (4 vCPU, 16 GB RAM, software
rendering), uploads `models/<run>/` as an artifact and optionally commits the weights back. Constraints:
a job is capped at 6 hours, the free runner is slower than a laptop, and there is no GPU. It works for
`ground_truth` mode runs and for short camera-mode runs, and you would need `--resume` across several jobs
for anything longer.

## Tuning

- Reward weights: `RewardConfig` in `drone_landing/config.py`. Add a term in `rewards.step_reward`.
- Start distribution, landing tolerances, camera size: `EnvConfig`.
- Drone mass / arm length / motor yaw torque: `assets/drone_landing.xml`. Hover thrust and ctrl range are
  derived from the model mass at runtime, so changing mass needs no code change.

## Roadmap

- v2: yaw alignment to the H (needs H orientation from the camera, e.g. principal axis of the blob), use
  the IMU / estimated pose instead of ground-truth xyz, wind and mass randomisation.
