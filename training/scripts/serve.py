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
    explanation: str


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


def count_holes(board: np.ndarray) -> int:
    holes = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c]:
                continue
            filled = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= BOARD_SIZE or nc < 0 or nc >= BOARD_SIZE:
                    filled += 1
                elif board[nr, nc]:
                    filled += 1
            if filled >= 3:
                holes += 1
    return holes


def find_completed_lines(board: np.ndarray) -> tuple[list[int], list[int]]:
    rows = [r for r in range(BOARD_SIZE) if board[r].all()]
    cols = [c for c in range(BOARD_SIZE) if board[:, c].all()]
    return rows, cols


def generate_explanation(
    board: list[list[bool]], piece: PieceData, row: int, col: int,
) -> str:
    board_arr = np.array(board, dtype=bool)
    shape_arr = np.array(piece.shape, dtype=bool)
    ph, pw = shape_arr.shape

    new_board = board_arr.copy()
    for r in range(ph):
        for c in range(pw):
            if shape_arr[r, c]:
                new_board[row + r, col + c] = True

    cleared_rows, cleared_cols = find_completed_lines(new_board)
    lines_cleared = len(cleared_rows) + len(cleared_cols)

    for r in cleared_rows:
        new_board[r, :] = False
    for c in cleared_cols:
        new_board[:, c] = False

    holes_before = count_holes(board_arr)
    holes_after = count_holes(new_board)
    holes_delta = holes_after - holes_before

    occupied_before = int(board_arr.sum())
    occupied_after = int(new_board.sum())

    reasons = []
    if lines_cleared > 0:
        cells = lines_cleared * BOARD_SIZE
        reasons.append(f"Clears {lines_cleared} line{'s' if lines_cleared > 1 else ''} ({cells} cells)")
    if holes_delta < 0:
        reasons.append(f"Removes {-holes_delta} hole{'s' if -holes_delta > 1 else ''}")
    elif holes_delta == 0 and holes_before > 0:
        reasons.append("No new holes created")
    if lines_cleared == 0:
        near = []
        for r in range(BOARD_SIZE):
            filled = int(new_board[r].sum())
            if 6 <= filled < BOARD_SIZE:
                near.append(f"row {r+1} ({filled}/8)")
        for c in range(BOARD_SIZE):
            filled = int(new_board[:, c].sum())
            if 6 <= filled < BOARD_SIZE:
                near.append(f"col {c+1} ({filled}/8)")
        if near:
            reasons.append(f"Sets up {near[0]}")
    if occupied_after < occupied_before:
        reasons.append(f"Frees {occupied_before - occupied_after} cells")
    if not reasons:
        reasons.append("Best available placement")

    return " · ".join(reasons[:2])


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
            return MoveResponse(pieceIndex=0, row=0, col=0, action=0, explanation="No valid moves")

        state_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = trainer.select_action(state_t, mask, epsilon=0.0)

        piece_idx = action // 64
        row = (action % 64) // 8
        col = action % 8

        piece = req.pieces[piece_idx]
        explanation = generate_explanation(req.board, piece, row, col) if piece else "No piece"
        return MoveResponse(pieceIndex=piece_idx, row=row, col=col, action=action, explanation=explanation)

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
