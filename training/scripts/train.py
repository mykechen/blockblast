#!/usr/bin/env python3
"""Block Blast DQN training script.

Usage:
    cd training
    uv run python scripts/train.py
    uv run python scripts/train.py --config configs/default.yaml --steps 100000
"""

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.dqn import DQNTrainer, C51Trainer
from agent.model import get_device
from env.block_blast_env import BlockBlastEnv


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train Block Blast DQN agent")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--steps", type=int, default=None, help="Override total training steps")
    parser.add_argument("--checkpoint", default=None, help="Resume from checkpoint path")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--log-dir", default="logs", help="Parent directory for TensorBoard runs")
    parser.add_argument("--run-name", default=None, help="Run name suffix (defaults to unix timestamp)")
    args = parser.parse_args()

    config = load_config(args.config)
    tc = config["training"]
    if args.steps:
        tc["total_steps"] = args.steps

    device = get_device()
    print(f"Device: {device}")

    env = BlockBlastEnv(config_path=args.config)
    TrainerClass = C51Trainer if config.get("algorithm") == "c51" else DQNTrainer
    trainer = TrainerClass(env, config, device)

    start_step = 0
    epsilon = tc["epsilon_start"]
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = trainer.load_checkpoint(args.checkpoint)
        start_step = ckpt["step"]
        epsilon = ckpt["epsilon"]
        print(f"Resumed from step {start_step}")

    run_name = args.run_name or str(int(time.time()))
    log_dir = Path(args.log_dir) / f"run_{run_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    print(f"Logging to {log_dir}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints to {ckpt_dir}")

    total_steps = tc["total_steps"]
    batch_size = tc["batch_size"]
    min_replay = tc["min_replay_size"]
    train_freq = tc["train_freq"]
    target_freq = tc["target_update_freq"]
    eps_start = tc["epsilon_start"]
    eps_end = tc["epsilon_end"]
    eps_decay = tc["epsilon_decay_steps"]
    n_step = int(tc.get("n_step", 1))
    gamma = tc["gamma"]
    nstep_buffer: deque = deque()

    def _emit_nstep(buf: deque, force_done: bool = False):
        """Aggregate the fronted transitions in buf into one n-step transition and push."""
        if not buf:
            return
        s0, a0, _, _, _, _ = buf[0]
        R = 0.0
        for i, t in enumerate(buf):
            R += (gamma ** i) * t[2]
        _, _, _, s_last, done_last, mask_last = buf[-1]
        done_flag = bool(done_last or force_done)
        trainer.buffer.push(s0, a0, R, s_last, done_flag, mask_last)

    episode_rewards = deque(maxlen=100)
    episode_scores = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    best_scores: list[tuple[float, str]] = []

    obs, info = env.reset()
    episode_reward = 0.0
    episode_length = 0
    episode_count = 0

    step = start_step
    t_start = time.time()

    print(f"Training for {total_steps} steps...")

    while step < total_steps:
        epsilon = max(eps_end, eps_start - (step / eps_decay) * (eps_start - eps_end))

        state_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask = info["action_mask"]
        action = trainer.select_action(state_t, mask, epsilon)

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        next_mask = next_info["action_mask"]

        nstep_buffer.append((obs, action, float(reward), next_obs, done, next_mask))
        if len(nstep_buffer) >= n_step:
            _emit_nstep(nstep_buffer)
            nstep_buffer.popleft()

        episode_reward += reward
        episode_length += 1

        if step % train_freq == 0 and len(trainer.buffer) >= min_replay:
            beta = min(1.0, 0.4 + (step / total_steps) * 0.6)
            loss = trainer.train_step(batch_size, beta)

            if step % 1000 == 0:
                writer.add_scalar("train/loss", loss, step)
                writer.add_scalar("train/epsilon", epsilon, step)
                writer.add_scalar("train/buffer_size", len(trainer.buffer), step)
                writer.add_scalar("train/mean_q", trainer.last_mean_q, step)

        if step % target_freq == 0 and step > 0:
            trainer.update_target()

        if done:
            while nstep_buffer:
                _emit_nstep(nstep_buffer, force_done=True)
                nstep_buffer.popleft()

            game_score = env._state.score if env._state else 0
            episode_rewards.append(episode_reward)
            episode_scores.append(game_score)
            episode_lengths.append(episode_length)
            episode_count += 1

            writer.add_scalar("episode/reward", episode_reward, step)
            writer.add_scalar("episode/score", game_score, step)
            writer.add_scalar("episode/length", episode_length, step)

            if episode_count % 10 == 0:
                elapsed = time.time() - t_start
                sps = step / max(elapsed, 1)
                mean_score = np.mean(episode_scores) if episode_scores else 0
                mean_reward = np.mean(episode_rewards) if episode_rewards else 0
                print(
                    f"Step {step:>8d}/{total_steps} | "
                    f"Ep {episode_count:>5d} | "
                    f"Score {mean_score:>6.0f} | "
                    f"Reward {mean_reward:>7.1f} | "
                    f"Eps {epsilon:.3f} | "
                    f"SPS {sps:.0f}"
                )

            obs, info = env.reset()
            episode_reward = 0.0
            episode_length = 0
        else:
            obs, info = next_obs, next_info

        if step > 0 and step % 50_000 == 0:
            mean_score = float(np.mean(episode_scores)) if episode_scores else 0.0
            ckpt_path = str(ckpt_dir / f"checkpoint_{step}.pt")
            trainer.save_checkpoint(ckpt_path, step, epsilon, list(episode_scores))
            print(f"  Saved checkpoint: {ckpt_path} (mean_score={mean_score:.0f})")

            best_scores.append((mean_score, ckpt_path))
            best_scores.sort(key=lambda x: x[0], reverse=True)
            while len(best_scores) > 5:
                _, old_path = best_scores.pop()
                if os.path.exists(old_path):
                    os.remove(old_path)

        step += 1

    final_path = str(ckpt_dir / "final_model.pt")
    trainer.save_checkpoint(final_path, step, epsilon, list(episode_scores))
    print(f"\nTraining complete. Final model saved to {final_path}")
    print(f"Total episodes: {episode_count}")
    if episode_scores:
        print(f"Final mean score (last 100): {np.mean(episode_scores):.0f}")

    writer.close()


if __name__ == "__main__":
    main()
