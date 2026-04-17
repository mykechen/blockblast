#!/usr/bin/env python3
"""Search many seeds for demo-worthy high-scoring runs.

Evaluates a checkpoint on N seeds with greedy play, ranks by score, and saves
the top-K games (seed + action sequence) as JSON so they can be replayed later.

Usage:
    cd training
    uv run python scripts/demo_search.py --checkpoint checkpoints_tier2/checkpoint_1850000.pt
    uv run python scripts/demo_search.py --checkpoint <path> --seeds 10000 --top-k 20 --config configs/tier2.yaml
"""

import argparse
import json
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


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def play_one(trainer: DQNTrainer, env: BlockBlastEnv, seed: int, device: torch.device) -> dict:
    obs, info = env.reset(seed=seed)
    actions: list[int] = []
    done = False
    while not done:
        state_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action = trainer.select_action(state_t, info["action_mask"], epsilon=0.0)
        actions.append(int(action))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    score = env._state.score if env._state else 0
    return {"seed": seed, "score": int(score), "length": len(actions), "actions": actions}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="configs/tier2.yaml")
    p.add_argument("--seeds", type=int, default=10_000)
    p.add_argument("--seed-base", type=int, default=5_000_000)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--output", default=None, help="Output JSON path (default: demos/<checkpoint_name>.json)")
    p.add_argument("--progress-every", type=int, default=500)
    args = p.parse_args()

    config = load_config(args.config)
    device = get_device()
    env = BlockBlastEnv(config_path=args.config)
    TrainerClass = C51Trainer if config.get("algorithm") == "c51" else DQNTrainer
    trainer = TrainerClass(env, config, device)
    trainer.load_checkpoint(args.checkpoint)
    trainer.policy_net.eval()

    ckpt_name = Path(args.checkpoint).stem
    out_path = Path(args.output) if args.output else Path("demos") / f"{ckpt_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Searching {args.seeds} seeds from {args.seed_base}, keeping top {args.top_k}")
    print(f"Output: {out_path}\n")

    t0 = time.time()
    results: list[dict] = []
    for i in range(args.seeds):
        seed = args.seed_base + i
        r = play_one(trainer, env, seed, device)
        results.append(r)
        if (i + 1) % args.progress_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta_s = (args.seeds - i - 1) / max(rate, 1e-6)
            scores = np.array([x["score"] for x in results])
            top = np.sort(scores)[-5:][::-1]
            print(
                f"  [{i+1}/{args.seeds}] rate={rate:.1f} ep/s  elapsed={elapsed:.0f}s  "
                f"ETA={eta_s:.0f}s  mean={scores.mean():.0f}  max={scores.max()}  "
                f"top5={top.tolist()}"
            )

    results.sort(key=lambda x: x["score"], reverse=True)
    top_k = results[: args.top_k]

    scores = np.array([x["score"] for x in results])
    summary = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "seeds_searched": args.seeds,
        "seed_base": args.seed_base,
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "std": float(scores.std()),
        "max": int(scores.max()),
        "min": int(scores.min()),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
        "elapsed_seconds": time.time() - t0,
        "top_k": top_k,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Search complete in {summary['elapsed_seconds']:.0f}s ===")
    print(f"Mean: {summary['mean']:.0f}  Median: {summary['median']:.0f}  "
          f"p95: {summary['p95']:.0f}  p99: {summary['p99']:.0f}  Max: {summary['max']}")
    print(f"\nTop {args.top_k} games:")
    for rank, g in enumerate(top_k, 1):
        print(f"  {rank:>2d}. seed={g['seed']:>10d}  score={g['score']:>7d}  length={g['length']}")
    print(f"\nSaved to: {out_path}")
    print(f"Replay any game with: uv run python scripts/replay_demo.py --demo-file {out_path} --rank 1")


if __name__ == "__main__":
    main()
