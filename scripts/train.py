"""Train the landing policy with PPO (Stable-Baselines3).

    python scripts/train.py --run-name v1 --timesteps 2000000 --n-envs 8
    python scripts/train.py --run-name v1_gt --vision ground_truth      # no rendering, ~10x faster

Weights land in models/<run_name>/ (see models/README.md). TensorBoard logs in logs/<run_name>/.
"""
import _gl  # noqa: F401  (must be first)

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from drone_landing.config import LOGS_DIR, MODELS_DIR, EnvConfig, RewardConfig, TrainConfig
from drone_landing.env import DroneLandingEnv


def parse_args():
    t = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default="v1")
    p.add_argument("--timesteps", type=int, default=t.total_timesteps)
    p.add_argument("--n-envs", type=int, default=t.n_envs)
    p.add_argument("--vision", choices=["camera", "ground_truth"], default=EnvConfig().vision_mode)
    p.add_argument("--vec", choices=["subproc", "dummy"], default="subproc",
                   help="subproc = one process per env (faster); dummy = single process (use on macOS / if GL breaks)")
    p.add_argument("--seed", type=int, default=t.seed)
    p.add_argument("--resume", type=Path, default=None, help="path to a .zip to continue training from")
    return p.parse_args()


def main():
    args = parse_args()
    tcfg = TrainConfig(total_timesteps=args.timesteps, n_envs=args.n_envs, seed=args.seed)
    ecfg = EnvConfig(vision_mode=args.vision)
    rcfg = RewardConfig()

    run_dir = MODELS_DIR / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(
        {"env": dataclasses.asdict(ecfg), "reward": dataclasses.asdict(rcfg), "train": dataclasses.asdict(tcfg)},
        indent=2))

    vec_cls = SubprocVecEnv if args.vec == "subproc" else DummyVecEnv
    env_kwargs = {"env_cfg": ecfg, "reward_cfg": rcfg}
    env = make_vec_env(DroneLandingEnv, n_envs=tcfg.n_envs, seed=tcfg.seed, env_kwargs=env_kwargs, vec_env_cls=vec_cls)
    eval_env = make_vec_env(DroneLandingEnv, n_envs=1, seed=tcfg.seed + 1000, env_kwargs=env_kwargs, vec_env_cls=DummyVecEnv)

    if args.resume:
        model = PPO.load(args.resume, env=env, tensorboard_log=str(LOGS_DIR))
        print(f"resumed from {args.resume}")
    else:
        model = PPO(
            "MlpPolicy", env, seed=tcfg.seed, verbose=1, tensorboard_log=str(LOGS_DIR),
            n_steps=tcfg.n_steps, batch_size=tcfg.batch_size, n_epochs=tcfg.n_epochs,
            learning_rate=tcfg.learning_rate, gamma=tcfg.gamma, gae_lambda=tcfg.gae_lambda,
            clip_range=tcfg.clip_range, ent_coef=tcfg.ent_coef,
            policy_kwargs={"net_arch": {"pi": tcfg.net_arch, "vf": tcfg.net_arch}},
        )

    callbacks = [
        CheckpointCallback(save_freq=max(tcfg.checkpoint_every // tcfg.n_envs, 1), save_path=str(ckpt_dir), name_prefix="ppo"),
        EvalCallback(eval_env, best_model_save_path=str(run_dir), log_path=str(run_dir / "eval"),
                     eval_freq=max(tcfg.eval_every // tcfg.n_envs, 1), n_eval_episodes=tcfg.eval_episodes,
                     deterministic=True, render=False),
    ]
    try:
        model.learn(total_timesteps=tcfg.total_timesteps, callback=callbacks, tb_log_name=args.run_name,
                    reset_num_timesteps=args.resume is None)
    finally:
        # always keep the weights, even on Ctrl-C
        model.save(run_dir / "final_model.zip")
        print(f"saved {run_dir / 'final_model.zip'}")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
