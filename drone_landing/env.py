"""Gymnasium environment: land a quadrotor on the H pad using ground-truth pose + onboard camera.

Action  (4,)  float in [-1, 1]  : per-motor throttle, -1 = -100 % (off), 0 = hover, +1 = +100 % (max).
Observation (19,) float32:
    [0:3]   drone xyz in world frame (ground truth from MuJoCo)
    [3:7]   drone orientation quaternion (w, x, y, z)
    [7:10]  drone linear velocity, world frame
    [10:13] drone angular velocity, body frame (gyro)
    [13]    pad visible in camera (1.0 / 0.0)
    [14:16] H centroid in normalised image coords (u right, v up), 0 if not visible
    [16:19] estimated pad-centre position relative to the drone, world frame (from the camera
            centroid ray-cast onto the ground plane), 0 if not visible
"""
from __future__ import annotations

import math
from dataclasses import replace

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from drone_landing.config import SCENE_XML, EnvConfig, RewardConfig
from drone_landing.rewards import step_reward, terminal_reward
from drone_landing.vision import find_h_centroid, ground_to_pixel, pixel_to_ground

OBS_DIM = 19


class DroneLandingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(self, env_cfg: EnvConfig | None = None, reward_cfg: RewardConfig | None = None,
                 render_mode: str | None = None, **overrides):
        super().__init__()
        self.cfg = replace(env_cfg or EnvConfig(), **overrides)
        self.rcfg = reward_cfg or RewardConfig()
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.drone_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "drone")
        self.pad_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pad")
        self.pad_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pad_geom")
        self.ground_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, self.cfg.cam_name)
        self.drone_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == self.drone_body}
        self.pad_top_z = float(m.geom_pos[self.pad_geom][2] + m.geom_size[self.pad_geom][2])
        self.sensor_adr = {m.sensor(i).name: (m.sensor_adr[i], m.sensor_dim[i]) for i in range(m.nsensor)}

        # Motors: thrust in Newtons; ctrl range set so that action 0 == hover.
        self.mass = float(m.body_subtreemass[self.drone_body])
        g = float(-m.opt.gravity[2])
        self.hover_thrust = self.mass * g / 4.0
        self.max_thrust = self.cfg.thrust_to_weight * self.hover_thrust
        m.actuator_ctrlrange[:] = [0.0, self.max_thrust]

        self.dt = float(m.opt.timestep * self.cfg.frame_skip)
        self.max_steps = int(self.cfg.max_episode_seconds / self.dt)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)

        self._renderer = None        # onboard camera (lazy, only in camera mode)
        self._ext_renderer = None    # chase camera for render()
        self._viewer = None
        self._last_frame = None
        self._step_count = 0
        self._prev_dist = 0.0
        self._action = np.zeros(4, dtype=np.float32)

    # ------------------------------------------------------------------ helpers
    @property
    def pad_center(self) -> np.ndarray:
        return self.model.body_pos[self.pad_body] + np.array([0.0, 0.0, self.pad_top_z])

    def _sensor(self, name: str) -> np.ndarray:
        adr, dim = self.sensor_adr[name]
        return self.data.sensordata[adr : adr + dim].copy()

    def get_imu(self) -> dict:
        """Raw IMU readings (body frame). Not used in the v1 observation."""
        return {"accel": self._sensor("imu_acc"), "gyro": self._sensor("imu_gyro")}

    def get_camera_image(self) -> np.ndarray | None:
        """Last rendered onboard frame (H x W x 3, RGB uint8). None in ground_truth mode."""
        return self._last_frame

    def _render_onboard(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.cfg.cam_height, self.cfg.cam_width)
        self._renderer.update_scene(self.data, camera=self.cam_id)
        return self._renderer.render()

    def _detect_pad(self) -> tuple[bool, float, float, np.ndarray]:
        """Returns (visible, u, v, pad_rel_xyz_estimate)."""
        cam_pos = self.data.cam_xpos[self.cam_id]
        cam_mat = self.data.cam_xmat[self.cam_id].reshape(3, 3)
        fovy = float(self.model.cam_fovy[self.cam_id])
        aspect = self.cfg.cam_width / self.cfg.cam_height

        if self.cfg.vision_mode == "camera":
            self._last_frame = self._render_onboard()
            det = find_h_centroid(self._last_frame, self.cfg.min_blob_pixels)
            uv = None if det is None else det[:2]
        elif self.cfg.vision_mode == "ground_truth":
            uv = ground_to_pixel(self.pad_center, cam_pos, cam_mat, fovy, aspect)
        else:
            raise ValueError(f"unknown vision_mode {self.cfg.vision_mode!r}")

        if uv is None:
            return False, 0.0, 0.0, np.zeros(3)
        hit = pixel_to_ground(uv[0], uv[1], cam_pos, cam_mat, fovy, aspect, self.pad_top_z)
        if hit is None:
            return False, 0.0, 0.0, np.zeros(3)
        return True, uv[0], uv[1], hit - self.data.xpos[self.drone_body]

    def _state(self) -> dict:
        d = self.data
        pos = d.xpos[self.drone_body].copy()
        quat = d.xquat[self.drone_body].copy()
        rot = d.xmat[self.drone_body].reshape(3, 3)
        linvel = d.qvel[0:3].copy()   # free joint: world-frame linear velocity
        angvel = self._sensor("imu_gyro")
        rel = pos - self.pad_center
        visible, u, v, pad_est = self._detect_pad()
        return {
            "pos": pos, "quat": quat, "linvel": linvel, "angvel": angvel,
            "xy_dist": float(np.linalg.norm(rel[:2])),
            "height": float(rel[2]),
            "dist": float(np.linalg.norm(rel)),
            "prev_dist": self._prev_dist,
            "speed": float(np.linalg.norm(linvel)),
            "tilt": float(math.acos(np.clip(rot[2, 2], -1.0, 1.0))),
            "pad_visible": visible, "u": u, "v": v, "pad_est": pad_est,
            "action": self._action,
        }

    def _obs(self, s: dict) -> np.ndarray:
        return np.concatenate([
            s["pos"], s["quat"], s["linvel"], s["angvel"],
            [1.0 if s["pad_visible"] else 0.0], [s["u"], s["v"]], s["pad_est"],
        ]).astype(np.float32)

    def _touching(self) -> tuple[bool, bool]:
        """(touching pad, touching ground)"""
        pad = ground = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 in self.drone_geoms or g2 in self.drone_geoms:
                other = g2 if g1 in self.drone_geoms else g1
                if other == self.pad_geom:
                    pad = True
                elif other == self.ground_geom:
                    ground = True
        return pad, ground

    def _outcome(self, s: dict) -> str | None:
        cfg = self.cfg
        rel = s["pos"] - self.pad_center
        if abs(rel[0]) > cfg.bounds_xy or abs(rel[1]) > cfg.bounds_xy or s["pos"][2] > cfg.bounds_z:
            return "out_of_bounds"
        if math.degrees(s["tilt"]) > cfg.crash_tilt_deg:
            return "crashed"
        on_pad, on_ground = self._touching()
        if on_pad or on_ground:
            gentle = (s["speed"] <= cfg.land_speed_max and math.degrees(s["tilt"]) <= cfg.land_tilt_deg)
            if on_pad and not on_ground and s["xy_dist"] <= cfg.land_xy_tol and gentle:
                return "landed"
            return "crashed"
        return None

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        cfg, rng = self.cfg, self.np_random
        mujoco.mj_resetData(self.model, self.data)

        # random pad position, then spawn the drone above and to the side of it
        pad_xy = rng.uniform(-cfg.pad_xy_range, cfg.pad_xy_range, size=2)
        self.model.body_pos[self.pad_body][:2] = pad_xy
        r = rng.uniform(*cfg.start_offset)
        th = rng.uniform(0.0, 2.0 * math.pi)
        z = self.pad_top_z + rng.uniform(*cfg.start_height)
        self.data.qpos[0:3] = [pad_xy[0] + r * math.cos(th), pad_xy[1] + r * math.sin(th), z]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]   # level, yaw 0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.hover_thrust           # start "stably": hovering
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._action = np.zeros(4, dtype=np.float32)
        self._prev_dist = float(np.linalg.norm(self.data.xpos[self.drone_body] - self.pad_center))
        s = self._state()
        return self._obs(s), self._info(s, None)

    def step(self, action):
        self._action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.data.ctrl[:] = (self._action + 1.0) * 0.5 * self.max_thrust
        for _ in range(self.cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        s = self._state()
        reward, terms = step_reward(self.rcfg, s)
        outcome = self._outcome(s)
        terminated = outcome is not None
        truncated = (not terminated) and self._step_count >= self.max_steps
        if terminated:
            reward += terminal_reward(self.rcfg, outcome, s)
        elif truncated:
            outcome = "timeout"
        self._prev_dist = s["dist"]

        info = self._info(s, outcome)
        info["reward_terms"] = terms
        return self._obs(s), float(reward), terminated, truncated, info

    def _info(self, s: dict, outcome: str | None) -> dict:
        return {
            "outcome": outcome,
            "xy_dist": s["xy_dist"], "height": s["height"], "speed": s["speed"],
            "tilt_deg": math.degrees(s["tilt"]), "pad_visible": s["pad_visible"],
            "motor_percent": (self._action * 100.0).tolist(),
            "imu": self.get_imu(),
        }

    def render(self):
        if self.render_mode == "rgb_array":
            if self._ext_renderer is None:
                self._ext_renderer = mujoco.Renderer(self.model, 480, 640)
            self._ext_renderer.update_scene(self.data, camera="chase")
            return self._ext_renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer
                self._viewer = viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        return None

    def close(self):
        for r in (self._renderer, self._ext_renderer):
            if r is not None:
                r.close()
        self._renderer = self._ext_renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
