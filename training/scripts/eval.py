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
import copy
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.dqn import DQNTrainer, C51Trainer
from agent.model import get_device
from env.block_blast_env import BlockBlastEnv


def _snapshot(env: BlockBlastEnv) -> dict:
    return {
        "state": copy.deepcopy(env._state),
        "rng_state": env._rng.bit_generator.state if env._rng else None,
        "combo_streak": env._combo_streak,
    }


def _restore(env: BlockBlastEnv, snap: dict) -> None:
    env._state = copy.deepcopy(snap["state"])
    if snap["rng_state"] is not None and env._rng is not None:
        env._rng.bit_generator.state = snap["rng_state"]
    env._combo_streak = snap["combo_streak"]


def _pick_candidates(
    trainer: DQNTrainer,
    obs: np.ndarray,
    action_mask: np.ndarray,
    device: torch.device,
    k: int,
) -> np.ndarray:
    valid = np.where(action_mask)[0]
    if len(valid) <= k:
        return valid
    state_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q = trainer.get_q_values(state_t).squeeze(0).cpu().numpy()
    q[~action_mask] = -np.inf
    return np.argsort(-q)[:k].astype(np.int64)


def _expand_and_score_1step(
    trainer: DQNTrainer,
    env: BlockBlastEnv,
    snap: dict,
    candidates: np.ndarray,
    device: torch.device,
    gamma: float,
) -> tuple[np.ndarray, list, list, list]:
    """Step env for each candidate from snap; return (values_1step, next_obs, next_masks, terms).

    values_1step[i] = reward_i + gamma * max_a' Q(s'_i, a') (0 if terminal)."""
    next_obs_list, next_masks_list, rewards_list, terms_list = [], [], [], []
    for a in candidates:
        _restore(env, snap)
        n_obs, r, term, trunc, n_info = env.step(int(a))
        next_obs_list.append(n_obs)
        next_masks_list.append(n_info["action_mask"])
        rewards_list.append(float(r))
        terms_list.append(bool(term or trunc))
    _restore(env, snap)

    states_batch = torch.tensor(np.array(next_obs_list), dtype=torch.float32, device=device)
    with torch.no_grad():
        q_batch = trainer.get_q_values(states_batch).cpu().numpy()
    masks_arr = np.array(next_masks_list, dtype=bool)
    q_batch[~masks_arr] = -np.inf
    max_q = q_batch.max(axis=1)
    max_q[~np.isfinite(max_q)] = 0.0

    rewards_arr = np.array(rewards_list, dtype=np.float32)
    terms_arr = np.array(terms_list, dtype=bool)
    bootstraps = np.where(terms_arr, 0.0, max_q).astype(np.float32)
    values = rewards_arr + gamma * bootstraps
    return values, next_obs_list, next_masks_list, terms_list


def lookahead_select_action(
    trainer: DQNTrainer,
    env: BlockBlastEnv,
    obs: np.ndarray,
    action_mask: np.ndarray,
    device: torch.device,
    gamma: float = 0.99,
    depth: int = 1,
    k: int = 8,
) -> int:
    """Pick action by lookahead search. depth=1: r+γ·maxQ(s'). depth=2: pruned top-k 2-step search."""
    valid = np.where(action_mask)[0]
    if len(valid) == 0:
        return 0
    if len(valid) == 1:
        return int(valid[0])

    snap = _snapshot(env)

    if depth == 1:
        values, *_ = _expand_and_score_1step(trainer, env, snap, valid, device, gamma)
        return int(valid[int(values.argmax())])

    # depth == 2: prune to top-k at root, then at each child, 1-step lookahead with sub-top-k
    root_candidates = _pick_candidates(trainer, obs, action_mask, device, k)
    child_rewards, child_next_obs, child_next_masks, child_terms = [], [], [], []
    for a in root_candidates:
        _restore(env, snap)
        n_obs, r, term, trunc, n_info = env.step(int(a))
        child_rewards.append(float(r))
        child_next_obs.append(n_obs)
        child_next_masks.append(n_info["action_mask"])
        child_terms.append(bool(term or trunc))
    _restore(env, snap)

    values = np.zeros(len(root_candidates), dtype=np.float32)
    for i, a in enumerate(root_candidates):
        if child_terms[i]:
            values[i] = child_rewards[i]
            continue
        sub_mask = child_next_masks[i]
        if not sub_mask.any():
            values[i] = child_rewards[i]
            continue
        sub_candidates = _pick_candidates(trainer, child_next_obs[i], sub_mask, device, k)
        _restore(env, snap)
        env.step(int(a))
        sub_snap = _snapshot(env)
        sub_values, *_ = _expand_and_score_1step(
            trainer, env, sub_snap, sub_candidates, device, gamma
        )
        v_next = float(sub_values.max())
        values[i] = child_rewards[i] + gamma * v_next

    _restore(env, snap)
    return int(root_candidates[int(values.argmax())])


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
    lookahead_depth: int = 0,
    lookahead_k: int = 8,
    gamma: float = 0.99,
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
            if lookahead_depth > 0:
                action = lookahead_select_action(
                    trainer, env, obs, info["action_mask"], device,
                    gamma=gamma, depth=lookahead_depth, k=lookahead_k,
                )
            else:
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
    parser.add_argument(
        "--lookahead",
        action="store_true",
        help="Shortcut for --lookahead-depth 1",
    )
    parser.add_argument(
        "--lookahead-depth",
        type=int,
        default=0,
        help="0=greedy, 1=1-step lookahead, 2=2-step pruned lookahead",
    )
    parser.add_argument(
        "--lookahead-k",
        type=int,
        default=8,
        help="Top-k pruning width for lookahead-depth >= 2",
    )
    parser.add_argument("--seed-base", type=int, default=1_000_000)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    gamma = float(config.get("training", {}).get("gamma", 0.99))
    lookahead_depth = max(args.lookahead_depth, 1 if args.lookahead else 0)
    print(f"Device: {device}  |  lookahead_depth: {lookahead_depth} (k={args.lookahead_k})  |  gamma: {gamma}")
    print(f"Episodes per run: {args.episodes}  |  seed_base={args.seed_base}\n")

    env = BlockBlastEnv(config_path=args.config)
    TrainerClass = C51Trainer if config.get("algorithm") == "c51" else DQNTrainer
    trainer = TrainerClass(env, config, device)

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
                lookahead_depth=lookahead_depth,
                lookahead_k=args.lookahead_k,
                gamma=gamma,
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
