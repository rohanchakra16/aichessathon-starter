#!/usr/bin/env python3
"""Train the shipped linear leaf evaluator on deterministic generated positions.

The teacher is repository-owned, readable material/activity code. Existing
engines are neither invoked nor shipped. The generated JSON records the entire
feature/training recipe required to reproduce the weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import chess
import numpy as np

SEED = 20260830
PIECE_VALUES = np.array([100.0, 320.0, 330.0, 500.0, 900.0, 0.0])


def features(board: chess.Board) -> list[float]:
    side = board.turn
    values: list[float] = []
    for piece_type in range(chess.PAWN, chess.KING + 1):
        values.append(
            float(len(board.pieces(piece_type, side)) - len(board.pieces(piece_type, not side)))
        )
    for piece_type in range(chess.PAWN, chess.KING + 1):
        activity = 0.0
        for colour, sign in ((side, 1.0), (not side, -1.0)):
            for square in board.pieces(piece_type, colour):
                file_distance = abs(chess.square_file(square) - 3.5)
                rank_distance = abs(chess.square_rank(square) - 3.5)
                activity += sign * (3.5 - (file_distance + rank_distance) / 2.0)
        values.append(activity)
    return values


def teacher(board: chess.Board, row: list[float]) -> float:
    material = float(np.dot(PIECE_VALUES, np.array(row[:6])))
    activity = float(sum(row[6:])) * 3.0
    mobility = float(board.legal_moves.count()) * 0.25
    return material + activity + mobility


def positions(count: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(SEED)
    rows: list[list[float]] = []
    labels: list[float] = []
    while len(rows) < count:
        board = chess.Board()
        plies = rng.randint(0, 100)
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if board.is_game_over(claim_draw=True):
            continue
        row = features(board)
        rows.append(row)
        labels.append(teacher(board, row))
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=6000)
    parser.add_argument("--output", type=Path, default=Path("weights/model.json"))
    args = parser.parse_args()
    x, y = positions(args.examples)
    design = np.column_stack((np.ones(len(x)), x))
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "model_kind": "linear_leaf_evaluator",
        "materially_drives": "all non-terminal search leaf evaluations",
        "training": {
            "method": "deterministic generated-position teacher distillation",
            "seed": SEED,
            "examples": args.examples,
            "script": "training/train_linear_evaluator.py",
            "script_sha256": script_hash,
            "external_engine_used": False
        },
        "features": [
            "pawn_count", "knight_count", "bishop_count", "rook_count",
            "queen_count", "king_count", "pawn_activity", "knight_activity",
            "bishop_activity", "rook_activity", "queen_activity", "king_activity"
        ],
        "bias": float(coefficients[0]),
        "weights": [float(value) for value in coefficients[1:]],
        "training_rmse": float(np.sqrt(np.mean((design @ coefficients - y) ** 2)))
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
