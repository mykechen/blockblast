#!/usr/bin/env python3
"""Replay a saved demo game step-by-step, printing board state at each step.

Usage:
    cd training
    uv run python scripts/replay_demo.py --demo-file demos/checkpoint_1850000.json --rank 1
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.block_blast_env import BlockBlastEnv


def render_board(board) -> str:
    rows = []
    for row in board:
        rows.append("".join("#" if c else "." for c in row))
    return "\n".join(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demo-file", required=True)
    p.add_argument("--rank", type=int, default=1, help="Which ranked game to replay (1 = top)")
    p.add_argument("--seed", type=int, default=None, help="Or replay a specific seed")
    p.add_argument("--config", default=None)
    p.add_argument("--pause", type=float, default=0.0, help="Seconds between steps (for visual pacing)")
    p.add_argument("--no-render", action="store_true")
    args = p.parse_args()

    with open(args.demo_file) as f:
        demo = json.load(f)

    config_path = args.config or demo["config"]
    env = BlockBlastEnv(config_path=config_path)

    if args.seed is not None:
        game = next((g for g in demo["top_k"] if g["seed"] == args.seed), None)
        if not game:
            print(f"Seed {args.seed} not in top_k; re-running by seed instead.")
            game = {"seed": args.seed, "actions": None}
    else:
        game = demo["top_k"][args.rank - 1]

    print(f"Replaying seed={game['seed']}  expected_score={game.get('score', '?')}")
    obs, info = env.reset(seed=game["seed"])

    import time
    actions = game.get("actions") or []
    for step_idx, action in enumerate(actions):
        obs, reward, term, trunc, info = env.step(int(action))
        score = env._state.score if env._state else 0
        if not args.no_render:
            print(f"\n--- step {step_idx+1}  action={action}  reward={reward:+.1f}  score={score} ---")
            print(render_board(env._state.board))
        if args.pause > 0:
            time.sleep(args.pause)
        if term or trunc:
            break

    final_score = env._state.score if env._state else 0
    print(f"\nFinal score: {final_score}")


if __name__ == "__main__":
    main()
