# drone_landing_rl

A quadrotor drone learns to land on a helipad marked with a yellow "H", inside the MuJoCo physics
simulator. The drone has a wide-angle camera pointing down. The camera finds the H, and a
reinforcement-learning (RL) policy controls the four motors to fly over the pad and touch down slowly.

Version 1 goal: line up in x/y and land gently. Matching the drone's heading to the H comes in version 2.

## How it works, in one paragraph

Every 1/50 s the policy gets the drone's position, orientation and speed (read straight from MuJoCo),
plus where the H is in the camera image. It outputs a throttle for each of the 4 motors, from -100 % (off)
to +100 % (max). MuJoCo does all the physics. A reward tells the policy whether it is doing well
(closer to the pad, slow, level, pad still in view) and gives a big bonus for a gentle touchdown on the pad
or a big penalty for crashing. The PPO algorithm trains the policy from many thousands of such episodes.

## Files

| Path | What it is |
|---|---|
| `assets/drone_landing.xml` | The MuJoCo scene: drone, motors, IMU, camera, landing pad, ground |
| `assets/h_pad.png` | Picture of the H painted on the pad (remake with `scripts/make_pad_texture.py`) |
| `drone_landing/env.py` | The Gymnasium environment (`DroneLandingEnv`) |
| `drone_landing/vision.py` | Finds the yellow H in the camera image and converts it to a position |
| `drone_landing/rewards.py` | The reward function |
| `drone_landing/config.py` | All the numbers you might want to change (env, reward weights, PPO settings) |
| `scripts/train.py` | Train the policy |
| `scripts/evaluate.py` | Test a trained policy, make a video |
| `scripts/check_setup.py` | Quick check that everything is installed and working |
| `tests/` | Automated sanity tests (`python -m pytest tests`) |
| `models/` | Trained weights are saved here |

## The drone and the scene

- 1 kg quadrotor in an X layout, arms 0.15 m. Four motors, each producing thrust straight up from the
  rotor plus a small yaw torque (opposite pairs spin opposite ways).
- IMU (accelerometer + gyro) at the centre. Available through `env.get_imu()`; not used by v1 yet.
- Camera `down_cam` under the body, pointing straight down, 120° field of view, 96x96 pixels.
  `env.get_camera_image()` gives the latest frame.
- Landing pad: 1.5 m x 1.5 m, dark grey, with a yellow H. Ground is green. The pad is placed at a random
  spot (within 1 m of the origin) every episode, so the policy has to use the camera to find it.
- Each episode the drone starts level and hovering, 2.5-3.5 m above the pad and 0.5-1.5 m to the side.
  The pad is always inside the camera view at the start.

## What the policy sees and does

**Action**: 4 numbers in [-1, 1], one per motor. -1 = -100 % (motor off), 0 = hover, +1 = +100 % (max
thrust, which is 2x hover). Throttle maps linearly to thrust.
Motor order: m1 front-right, m2 back-left, m3 front-left, m4 back-right.

**Observation**: 19 numbers.

| Index | Meaning | Comes from |
|---|---|---|
| 0-2 | drone x, y, z in the world | MuJoCo (ground truth) |
| 3-6 | orientation quaternion (w, x, y, z) | MuJoCo (ground truth) |
| 7-9 | velocity in the world frame | MuJoCo (ground truth) |
| 10-12 | turn rate in the body frame | gyro |
| 13 | 1 if the H is visible in the camera, else 0 | camera |
| 14-15 | where the H is in the image (u right, v up, both in [-1, 1]) | camera |
| 16-18 | estimated position of the pad centre relative to the drone (x, y, z) | camera |

How the camera numbers are made: keep only yellow pixels, take the biggest blob, find its centre pixel.
Then shoot a ray from the camera through that pixel and see where it hits the ground plane. That gives the
pad centre in metres. From 3 m up the error is about 3 cm.

**Episode ends** when one of these happens:

| Outcome | Condition |
|---|---|
| `landed` | touched the pad within 0.35 m of its centre, moving slower than 0.5 m/s, tilted less than 20° |
| `crashed` | touched the pad or ground any other way, or tilted more than 70° |
| `out_of_bounds` | more than 4 m from the pad sideways, or higher than 6 m |
| `timeout` | 12 seconds passed (600 steps) |

**Two vision modes** (`--vision` on the scripts):

- `camera` (default): really render the camera every step and detect the H. This is the actual task.
- `ground_truth`: same 19 numbers, but the H position is computed from the true pad position instead of
  from pixels. No rendering, about 50x faster. Use it first to check that learning works at all.

## The reward

Every step the policy gets the sum of these (weights are in `RewardConfig` in `config.py`):

| Term | Formula | Weight | Why |
|---|---|---|---|
| progress | + (last distance to pad - distance now) | 5.0 | reward for getting closer |
| xy | - horizontal distance to pad centre (m) | 0.02 | stay over the pad |
| z | - height above pad (m) | 0.01 | come down |
| vel | - speed (m/s) | 0.005 | be slow |
| tilt | - tilt angle (rad) | 0.02 | stay level |
| action | - mean(action²) | 0.001 | don't thrash the motors |
| lost | - 1 if the H is not in the camera view | 0.05 | keep the pad in sight |

When the episode ends, one bonus is added:

| Outcome | Bonus |
|---|---|
| landed | +200 for a very gentle touchdown, down to +100 at the 0.5 m/s limit |
| crashed | -100 |
| out_of_bounds | -100 |
| timeout | 0 |

The weights are chosen so that: **land (about +200) > hover all episode (about -30) > crash (about -90)**.
If you raise the per-step penalties a lot, crashing early becomes cheaper than flying, and the policy will
learn to crash. Keep that ordering when you tune. Each term's value is in `info["reward_terms"]` every step,
so you can see what is driving the behaviour.

## The RL model (PPO)

Algorithm: PPO (Proximal Policy Optimization) from Stable-Baselines3, the standard choice for continuous
control with a small state vector. It is on-policy, stable, and has few knobs.

Architecture (`MlpPolicy`, two separate networks):

```
Actor  (policy):  19 inputs -> Dense 64, tanh -> Dense 64, tanh -> 4 outputs (mean motor throttle)
                  + 4 learned log-std values (Gaussian exploration noise)
Critic (value):   19 inputs -> Dense 64, tanh -> Dense 64, tanh -> 1 output (state value)
```

About 11k parameters in total. During training, actions are sampled from the Gaussian; at test time the
mean is used. Everything runs on CPU. Memory use is dominated by the parallel MuJoCo instances, well under
2 GB in total.

Training settings (`TrainConfig` in `config.py`):

| Setting | Value |
|---|---|
| total steps | 2,000,000 |
| parallel envs | 8 |
| rollout per env before each update | 1024 steps (8192 total) |
| minibatch / epochs per update | 256 / 10 |
| learning rate | 3e-4 |
| discount gamma / GAE lambda | 0.99 / 0.95 |
| clip range | 0.2 |
| checkpoint / eval every | 100k / 50k steps |

## Setup

```bash
git clone https://github.com/mihirk460/drone_landing_rl.git
cd drone_landing_rl
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional, Linux only, to avoid the multi-GB CUDA download:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`. On macOS the normal torch wheel is
already CPU-only.

Rendering the camera without a window:

- **macOS / Windows**: nothing to do.
- **Linux**: the scripts set `MUJOCO_GL=egl` automatically. If that errors, run `export MUJOCO_GL=osmesa`
  (`sudo apt install libosmesa6`). OSMesa and PyTorch can crash together (segfault); if so, go back to EGL
  or train with `--vision ground_truth`.

Check it works:

```bash
python scripts/check_setup.py      # prints positions, H detection, hover check; saves outputs/camera_sample.png
python -m pytest tests             # 5 tests, about 1 second
```

## Train

```bash
# 1. quick sanity run, no rendering (a few minutes). The reward should climb and "landed" should appear in the eval log.
python scripts/train.py --run-name test_gt --vision ground_truth --timesteps 500000

# 2. the real thing, with the camera
python scripts/train.py --run-name v1

# watch progress
tensorboard --logdir logs
```

Useful options:

| Option | Meaning |
|---|---|
| `--timesteps N` | total environment steps (default 2,000,000) |
| `--n-envs N` | parallel environments (default 8; use your number of CPU cores) |
| `--vision camera\|ground_truth` | see "Two vision modes" |
| `--vec subproc\|dummy` | `subproc` = one process per env (fast). Use `dummy` if subprocess rendering fails on your machine |
| `--resume models/v1/final_model.zip` | keep training an existing model |
| `--seed N` | random seed |

Ctrl-C is safe: the current weights are saved before the script exits.

### Where the trained model is saved

Yes, to a folder: `models/<run_name>/`.

```
models/v1/final_model.zip      weights when training finished (or when you pressed Ctrl-C)
models/v1/best_model.zip       weights with the best evaluation score seen during training  <- use this one
models/v1/checkpoints/         a snapshot every 100k steps (ignored by git)
models/v1/config.json          the exact settings this run used
```

`final_model.zip` and `best_model.zip` are about 100 kB each and are not git-ignored. Commit them so the
weights are stored in the repo and not only on your laptop.

### How long will it take?

Rough estimates for a MacBook Air (M-series, 16 GB), 8 parallel envs:

| Mode | Speed | 2M steps |
|---|---|---|
| `ground_truth` | several thousand steps/s | 10-20 minutes |
| `camera` | a few hundred to ~1500 steps/s (rendering is the bottleneck) | 30 minutes to 2 hours |

Whether 2M steps is *enough* is a separate question. Learning raw per-motor control from scratch is hard;
it is normal to need 5-10M steps before landings become reliable. If the eval reward is still improving at
the end, continue with `--resume`. Run the `ground_truth` sanity run first: if it does not learn to land
in a few million steps, more camera-mode training will not help and the reward or hyper-parameters need
attention.

## Test a trained model

```bash
python scripts/evaluate.py --model models/v1/best_model.zip --episodes 20
python scripts/evaluate.py --model models/v1/best_model.zip --video outputs/landing.mp4   # chase view + camera inset
python scripts/evaluate.py --model models/v1/best_model.zip --render                      # live 3D viewer (on macOS run with: mjpython scripts/evaluate.py ...)
```

It prints one line per episode (outcome, return, final position error, touchdown speed) and a summary.

## Things you may want to change

- Reward weights: `RewardConfig` in `drone_landing/config.py`. New term: add a line in `rewards.step_reward`.
- Start height/offset, landing tolerances, camera size: `EnvConfig` in `config.py`.
- Drone mass, arm length, yaw torque: `assets/drone_landing.xml`. Hover thrust is computed from the mass
  at run time, so no code changes are needed.
- Pad colour/shape: `scripts/make_pad_texture.py` (and the yellow threshold in `vision.find_h_centroid`).

## Next version

- Yaw alignment to the H (get the H's orientation from the blob's main axis).
- Use the IMU instead of ground-truth position.
- Randomise mass and add wind so the policy is more robust.
