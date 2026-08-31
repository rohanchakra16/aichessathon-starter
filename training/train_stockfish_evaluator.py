#!/usr/bin/env python3
"""Train a fast tapered piece-square evaluator from offline engine labels.

The external engine is a development-only teacher. It is never imported by or
packaged with ``agent.py``. One thread and a fixed node count make dataset
generation repeatable for a recorded engine binary and invocation order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any

import chess
import chess.engine
import numpy as np

SEED = 20260831
PIECE_TYPES = tuple(range(chess.PAWN, chess.KING + 1))
SQUARE_FEATURES = len(PIECE_TYPES) * 64
CASTLING_FEATURES = 2
FEATURES = SQUARE_FEATURES * 2 + CASTLING_FEATURES
PHASE_VALUES = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
    chess.KING: 0,
}
MAX_PHASE = 24


def phase(board: chess.Board) -> float:
    current = sum(
        PHASE_VALUES[piece_type] * len(board.pieces(piece_type, colour))
        for piece_type in PIECE_TYPES
        for colour in chess.COLORS
    )
    return min(1.0, current / MAX_PHASE)


def features(board: chess.Board) -> np.ndarray:
    """Tapered piece-square and castling features from side-to-move view."""
    squares = np.zeros(SQUARE_FEATURES, dtype=np.float64)
    side = board.turn
    for colour in chess.COLORS:
        sign = 1.0 if colour == side else -1.0
        for piece_type in PIECE_TYPES:
            offset = (piece_type - 1) * 64
            for square in board.pieces(piece_type, colour):
                relative = square if colour == chess.WHITE else chess.square_mirror(square)
                squares[offset + relative] += sign
    current_phase = phase(board)
    castling = np.asarray(
        [
            float(board.has_kingside_castling_rights(side))
            - float(board.has_kingside_castling_rights(not side)),
            float(board.has_queenside_castling_rights(side))
            - float(board.has_queenside_castling_rights(not side)),
        ],
        dtype=np.float64,
    )
    return np.concatenate((squares * current_phase, squares * (1.0 - current_phase), castling))


def choose_move(board: chess.Board, rng: random.Random) -> chess.Move:
    moves = list(board.legal_moves)
    weights: list[float] = []
    for move in moves:
        weight = 1.0
        if board.is_capture(move):
            weight += 3.0
        if move.promotion:
            weight += 5.0
        if board.is_castling(move):
            weight += 1.5
        if board.gives_check(move):
            weight += 2.0
        weights.append(weight)
    return rng.choices(moves, weights=weights, k=1)[0]


def generated_positions(count: int) -> list[chess.Board]:
    rng = random.Random(SEED)
    positions: list[chess.Board] = []
    while len(positions) < count:
        board = chess.Board()
        target = rng.randint(8, 110)
        for ply in range(target):
            if board.is_game_over(claim_draw=True):
                break
            board.push(choose_move(board, rng))
            if ply >= 7 and rng.random() < 0.28 and not board.is_game_over(claim_draw=True):
                positions.append(board.copy(stack=False))
                if len(positions) == count:
                    break
    return positions


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def teacher_labels(
    positions: list[chess.Board], engine_path: Path, nodes: int, progress_every: int
) -> tuple[np.ndarray, str]:
    labels: list[float] = []
    dataset_digest = hashlib.sha256()
    with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
        engine.configure({"Threads": 1, "Hash": 64})
        for index, board in enumerate(positions, 1):
            information = engine.analyse(board, chess.engine.Limit(nodes=nodes))
            score = information["score"].pov(board.turn).score(mate_score=10_000)
            if score is None:
                raise RuntimeError(f"teacher produced no score for {board.fen()}")
            clipped = float(max(-2_000, min(2_000, score)))
            labels.append(clipped)
            dataset_digest.update(f"{board.fen()}\t{clipped:.1f}\n".encode())
            if progress_every and index % progress_every == 0:
                print(f"labelled {index}/{len(positions)}", flush=True)
    return np.asarray(labels, dtype=np.float64), dataset_digest.hexdigest()


def ridge_fit(design: np.ndarray, labels: np.ndarray, penalty: float) -> np.ndarray:
    augmented = np.column_stack((np.ones(len(design)), design))
    regularizer = np.eye(augmented.shape[1], dtype=np.float64) * penalty
    regularizer[0, 0] = 0.0
    return np.linalg.solve(augmented.T @ augmented + regularizer, augmented.T @ labels)


def root_mean_square_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def train(
    positions: list[chess.Board], labels: np.ndarray, penalty: float
) -> tuple[np.ndarray, dict[str, float | int]]:
    design = np.vstack([features(board) for board in positions])
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(design))
    validation_count = max(1, len(design) // 5)
    validation = indices[:validation_count]
    training = indices[validation_count:]
    coefficients = ridge_fit(design[training], labels[training], penalty)
    training_prediction = coefficients[0] + design[training] @ coefficients[1:]
    validation_prediction = coefficients[0] + design[validation] @ coefficients[1:]
    metrics: dict[str, float | int] = {
        "training_examples": len(training),
        "validation_examples": len(validation),
        "training_rmse": root_mean_square_error(labels[training], training_prediction),
        "validation_rmse": root_mean_square_error(labels[validation], validation_prediction),
    }
    return coefficients, metrics


def model_payload(
    coefficients: np.ndarray,
    metrics: dict[str, float | int],
    args: argparse.Namespace,
    engine_path: Path,
    dataset_sha256: str,
) -> dict[str, Any]:
    script_path = Path(__file__)
    return {
        "schema_version": 2,
        "model_kind": "stockfish_distilled_tapered_piece_square_evaluator",
        "materially_drives": "all non-terminal search leaf evaluations",
        "layout": {
            "midgame_piece_square": [0, SQUARE_FEATURES],
            "endgame_piece_square": [SQUARE_FEATURES, SQUARE_FEATURES * 2],
            "castling": [SQUARE_FEATURES * 2, FEATURES],
            "relative_square_orientation": "white=a1..h8; black vertically mirrored",
            "phase_max": MAX_PHASE,
        },
        "training": {
            "method": "deterministic generated positions labelled by offline Stockfish",
            "seed": SEED,
            "examples": args.examples,
            "nodes_per_position": args.nodes,
            "ridge_penalty": args.ridge_penalty,
            "script": str(script_path.relative_to(Path.cwd())),
            "script_sha256": file_sha256(script_path),
            "external_engine_used": True,
            "teacher_name": "Stockfish 18",
            "teacher_binary_sha256": file_sha256(engine_path),
            "dataset_sha256": dataset_sha256,
        },
        "bias": float(coefficients[0]),
        "weights": [float(value) for value in coefficients[1:]],
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=6000)
    parser.add_argument("--nodes", type=int, default=2500)
    parser.add_argument("--ridge-penalty", type=float, default=10.0)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--output", type=Path, default=Path("weights/model.json"))
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()
    if args.examples < 10:
        parser.error("--examples must be at least 10")
    if args.nodes < 1:
        parser.error("--nodes must be positive")
    discovered = shutil.which("stockfish") if args.engine is None else str(args.engine)
    if discovered is None:
        parser.error("Stockfish is required for offline labels; pass --engine")
    engine_path = Path(discovered).resolve()
    positions = generated_positions(args.examples)
    labels, dataset_sha256 = teacher_labels(
        positions, engine_path, args.nodes, args.progress_every
    )
    coefficients, metrics = train(positions, labels, args.ridge_penalty)
    payload = model_payload(coefficients, metrics, args, engine_path, dataset_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
