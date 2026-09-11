"""All tunable numbers in one place. Edit here, not in env.py."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENE_XML = ROOT / "assets" / "drone_landing.xml"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"


@dataclass
class EnvConfig:
    # ---- simulation ----
    frame_skip: int = 10            # sim timestep 0.002 s * 10 = 50 Hz control
    max_episode_seconds: float = 12.0

    # ---- motors ----
    # Action from the policy is in [-1, 1] per motor == [-100 %, +100 %].
    # -100 % -> motor off, 0 % -> hover thrust, +100 % -> max thrust (2x hover).
    # Thrust-to-weight ratio of the vehicle is therefore 2.0.
    thrust_to_weight: float = 2.0

    # ---- camera ----
    cam_width: int = 96
    cam_height: int = 96
    cam_name: str = "down_cam"
    # "camera": render the onboard camera and detect the H with OpenCV (the real thing).
    # "ground_truth": skip rendering, project the true pad centre through the same
    #                 camera model. Same observation layout; ~10x faster; no OpenGL needed.
    vision_mode: str = "camera"
    dark_threshold: int = 80        # pixel value below which a pixel counts as "black H"
    min_blob_pixels: int = 3        # smaller detections are treated as "pad not visible"

    # ---- episode start ----
    start_height: tuple[float, float] = (2.5, 3.5)     # metres above the pad
    start_offset: tuple[float, float] = (0.5, 1.5)     # horizontal distance from pad, metres
    pad_xy_range: float = 1.0       # pad centre is sampled uniformly in [-r, r]^2 each episode
                                    # (set 0 to pin it at the origin; then the camera is redundant
                                    # because the ground-truth xyz already tells you where the pad is)

    # ---- termination ----
    bounds_xy: float = 4.0          # |x - pad_x| or |y - pad_y| beyond this -> out of bounds
    bounds_z: float = 6.0
    crash_tilt_deg: float = 70.0    # exceeding this tilt anywhere -> crash
    land_xy_tol: float = 0.35       # touchdown within this radius of pad centre counts as on-pad
    land_speed_max: float = 0.5     # m/s at touchdown, above this it's a hard landing (crash)
    land_tilt_deg: float = 20.0     # tilt at touchdown above this -> crash


@dataclass
class RewardConfig:
    """Per-step reward = sum of weighted terms; terminal bonuses added on episode end.

    Per-step terms (all <= 0 except progress):
        progress : (previous 3D distance to pad - current distance)  -> positive when approaching
        xy       : -horizontal distance to pad centre
        z        : -height above pad
        vel      : -speed
        tilt     : -tilt angle (rad)
        action   : -mean(action^2)  (0 == hover, so this penalises aggressive throttle)
        lost     : -1 when the pad is not visible in the camera
    Terminal:
        landed   : +R_landed * softness  (softness in [0.5, 1], higher for slower touchdown)
        crashed  : R_crashed
        oob      : R_out_of_bounds
    """
    w_progress: float = 1.0
    w_xy: float = 0.10
    w_z: float = 0.03
    w_vel: float = 0.01
    w_tilt: float = 0.05
    w_action: float = 0.002
    w_lost: float = 0.10

    R_landed: float = 100.0
    R_crashed: float = -50.0
    R_out_of_bounds: float = -50.0


@dataclass
class TrainConfig:
    algo: str = "PPO"
    total_timesteps: int = 2_000_000
    n_envs: int = 8
    n_steps: int = 1024             # rollout length per env -> 8192 samples per update
    batch_size: int = 256
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    net_arch: list = field(default_factory=lambda: [64, 64])
    checkpoint_every: int = 100_000  # steps (per env count is handled in train.py)
    eval_every: int = 50_000
    eval_episodes: int = 10
    seed: int = 0
