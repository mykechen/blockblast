#!/usr/bin/env python3
"""Evaluate a trained Block Blast DQN checkpoint with a deterministic (greedy) policy.

Usage:
    cd training
    uv run python scripts/eval.py --checkpoint checkpoints/final_model.pt
    uv run python scripts/eval.py --checkpoint checkpoints/final_model.pt --episodes 200 --epsilon 0.0
    uv run python scripts/eval.py --sweep                      # evaluate every checkpoint in checkpoints/
    uv run python scripts/eval.py --sweep --episodes 50
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.dqn import DQNTrainer
from agent.model import get_device
from env.block_blast_env import BlockBlastEnv


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate(
    trainer: DQNTrainer,
    env: BlockBlastEnv,
    episodes: int,
    epsilon: float,
    device: torch.device,
    seed_base: int = 1_000_000,
) -> dict:
    scores: list[int] = []
    rewards: list[float] = []
    lengths: list[int] = []

    trainer.policy_net.eval()
    for i in range(episodes):
        obs, info = env.reset(seed=seed_base + i)
        ep_reward = 0.0
        ep_length = 0
        done = False

        while not done:
            state_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = trainer.select_action(state_t, info["action_mask"], epsilon)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_length += 1
            done = terminated or truncated

        game_score = env._state.score if env._state else 0
        scores.append(int(game_score))
        rewards.append(float(ep_reward))
        lengths.append(ep_length)

    scores_arr = np.array(scores)
    lengths_arr = np.array(lengths)
    rewards_arr = np.array(rewards)

    return {
        "episodes": episodes,
        "epsilon": epsilon,
        "score_mean": float(scores_arr.mean()),
        "score_median": float(np.median(scores_arr)),
        "score_std": float(scores_arr.std()),
        "score_min": int(scores_arr.min()),
        "score_max": int(scores_arr.max()),
        "score_p25": float(np.percentile(scores_arr, 25)),
        "score_p75": float(np.percentile(scores_arr, 75)),
        "length_mean": float(lengths_arr.mean()),
        "length_max": int(lengths_arr.max()),
        "reward_mean": float(rewards_arr.mean()),
        "scores": scores,
    }


def print_result(label: str, r: dict):
    print(
        f"{label:<35} "
        f"n={r['episodes']:<4d} "
        f"ε={r['epsilon']:.2f}  "
        f"score mean={r['score_mean']:>7.1f}  "
        f"med={r['score_median']:>5.0f}  "
        f"p25={r['score_p25']:>5.0f}  "
        f"p75={r['score_p75']:>6.0f}  "
        f"max={r['score_max']:>6d}  "
        f"min={r['score_min']:>4d}  "
        f"std={r['score_std']:>6.1f}  "
        f"len={r['length_mean']:>5.1f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN checkpoint")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/final_model.pt",
        help="Path to checkpoint (ignored if --sweep is set)",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Exploration rate during eval (default 0.0 = greedy)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Evaluate every .pt file in checkpoints/, ranked by mean score",
    )
    parser.add_argument(
        "--checkpoints-dir",
        default="checkpoints",
        help="Directory to sweep (used with --sweep)",
    )
    parser.add_argument(
        "--random-baseline",
        action="store_true",
        help="Also run a random (uniform over valid actions) baseline",
    )
    parser.add_argument("--seed-base", type=int, default=1_000_000)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    print(f"Device: {device}")
    print(f"Episodes per run: {args.episodes}  |  seed_base={args.seed_base}\n")

    env = BlockBlastEnv(config_path=args.config)
    trainer = DQNTrainer(env, config, device)

    if args.random_baseline:
        t0 = time.time()
        baseline = evaluate(
            trainer, env, args.episodes, epsilon=1.0, device=device, seed_base=args.seed_base
        )
        print_result(f"RANDOM baseline (ε=1.0)", baseline)
        print(f"  elapsed: {time.time() - t0:.1f}s\n")

    if args.sweep:
        pattern = os.path.join(args.checkpoints_dir, "*.pt")
        paths = sorted(glob.glob(pattern))
        if not paths:
            print(f"No checkpoints found in {args.checkpoints_dir}/")
            sys.exit(1)

        results = []
        for path in paths:
            trainer.load_checkpoint(path)
            t0 = time.time()
            r = evaluate(
                trainer,
                env,
                args.episodes,
                args.epsilon,
                device,
                seed_base=args.seed_base,
            )
            results.append((path, r))
            print_result(Path(path).name, r)
            print(f"  elapsed: {time.time() - t0:.1f}s")

        print("\n=== Ranked by mean score ===")
        for path, r in sorted(results, key=lambda x: x[1]["score_mean"], reverse=True):
            print_result(Path(path).name, r)

        best_path, best_r = max(results, key=lambda x: x[1]["score_mean"])
        print(f"\nBest checkpoint: {best_path}  (mean={best_r['score_mean']:.1f})")
    else:
        if not os.path.exists(args.checkpoint):
            print(f"Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
        trainer.load_checkpoint(args.checkpoint)
        t0 = time.time()
        r = evaluate(
            trainer,
            env,
            args.episodes,
            args.epsilon,
            device,
            seed_base=args.seed_base,
        )
        print_result(Path(args.checkpoint).name, r)
        print(f"  elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
