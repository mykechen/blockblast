#!/usr/bin/env python3
"""FastAPI server that exposes the trained DQN agent for the Next.js game.

Usage:
    cd training
    uv run python scripts/serve.py
    uv run python scripts/serve.py --checkpoint checkpoints_tier2/checkpoint_1850000.pt --config configs/tier2.yaml
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.dqn import DQNTrainer, C51Trainer
from agent.model import get_device
from env.block_blast_env import BlockBlastEnv

BOARD_SIZE = 8
OBS_CHANNELS = 9


class PieceData(BaseModel):
    shape: list[list[bool]]
    id: str


class MoveRequest(BaseModel):
    board: list[list[bool]]
    pieces: list[PieceData | None]


class MoveResponse(BaseModel):
    pieceIndex: int
    row: int
    col: int
    action: int


def build_observation(board: list[list[bool]], pieces: list[PieceData | None]) -> np.ndarray:
    obs = np.zeros((OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    board_arr = np.array(board, dtype=np.float32)
    obs[0] = board_arr

    for i in range(3):
        piece = pieces[i] if i < len(pieces) else None
        if piece is not None:
            shape = np.array(piece.shape, dtype=np.float32)
            h, w = shape.shape
            obs[1 + i, :h, :w] = shape

    for i in range(3):
        piece = pieces[i] if i < len(pieces) else None
        if piece is not None:
            obs[4 + i, :, :] = 1.0

    row_fill = board_arr.sum(axis=1) / BOARD_SIZE
    col_fill = board_arr.sum(axis=0) / BOARD_SIZE
    obs[7] = row_fill[:, None]
    obs[8] = col_fill[None, :]

    return obs


def build_action_mask(board: list[list[bool]], pieces: list[PieceData | None]) -> np.ndarray:
    mask = np.zeros(192, dtype=bool)
    board_arr = np.array(board, dtype=bool)

    for piece_idx in range(3):
        piece = pieces[piece_idx] if piece_idx < len(pieces) else None
        if piece is None:
            continue
        shape = np.array(piece.shape, dtype=bool)
        ph, pw = shape.shape
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if row + ph > BOARD_SIZE or col + pw > BOARD_SIZE:
                    continue
                region = board_arr[row:row + ph, col:col + pw]
                if not np.any(region & shape):
                    mask[piece_idx * 64 + row * 8 + col] = True

    return mask


def create_app(checkpoint: str, config_path: str) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = get_device()
    env = BlockBlastEnv(config_path=config_path)
    TrainerClass = C51Trainer if config.get("algorithm") == "c51" else DQNTrainer
    trainer = TrainerClass(env, config, device)
    trainer.load_checkpoint(checkpoint)
    trainer.policy_net.eval()

    print(f"Loaded {checkpoint} on {device}")

    @app.post("/move", response_model=MoveResponse)
    def get_move(req: MoveRequest):
        obs = build_observation(req.board, req.pieces)
        mask = build_action_mask(req.board, req.pieces)

        if not mask.any():
            return MoveResponse(pieceIndex=0, row=0, col=0, action=0)

        state_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = trainer.select_action(state_t, mask, epsilon=0.0)

        piece_idx = action // 64
        row = (action % 64) // 8
        col = action % 8
        return MoveResponse(pieceIndex=piece_idx, row=row, col=col, action=action)

    @app.get("/health")
    def health():
        return {"status": "ok", "checkpoint": checkpoint, "device": str(device)}

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints_tier2/checkpoint_1850000.pt")
    parser.add_argument("--config", default="configs/tier2.yaml")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = create_app(args.checkpoint, args.config)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
