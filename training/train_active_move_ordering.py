#!/usr/bin/env python3
"""Train a cheap move-ordering policy from game-grouped MultiPV children."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import chess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_active_residual_evaluator import (  # noqa: E402
    ACTIVE_SPLIT_SEED,
    load_active_dataset,
    split_game_ids,
)
from training.train_stockfish_evaluator import file_sha256, root_mean_square_error  # noqa: E402

PIECE_SQUARES = 6 * 64
FROM_OFFSET = 0
TO_OFFSET = PIECE_SQUARES
CAPTURE_OFFSET = PIECE_SQUARES * 2
PROMOTION_OFFSET = CAPTURE_OFFSET + 1
CHECK_OFFSET = PROMOTION_OFFSET + 1
CASTLING_OFFSET = CHECK_OFFSET + 1
VICTIM_OFFSET = CASTLING_OFFSET + 1
FEATURES = VICTIM_OFFSET + 6


def move_between(parent_fen: str, child_fen: str) -> chess.Move:
    board = chess.Board(parent_fen)
    for move in board.legal_moves:
        board.push(move)
        matches = board.fen() == child_fen
        board.pop()
        if matches:
            return move
    raise ValueError(f"no legal move connects parent to child: {parent_fen} -> {child_fen}")


def move_features(board: chess.Board, move: chess.Move) -> np.ndarray:
    piece_type = board.piece_type_at(move.from_square)
    if piece_type is None:
        raise ValueError(f"move has no source piece: {move.uci()}")
    vector = np.zeros(FEATURES, dtype=np.float64)
    piece_offset = (piece_type - 1) * 64
    vector[FROM_OFFSET + piece_offset + move.from_square] = 1.0
    vector[TO_OFFSET + piece_offset + move.to_square] = 1.0
    victim = board.piece_type_at(move.to_square) or 0
    if board.is_en_passant(move):
        victim = chess.PAWN
    if victim:
        vector[CAPTURE_OFFSET] = 1.0
        vector[VICTIM_OFFSET + victim - 1] = 1.0
    vector[PROMOTION_OFFSET] = float(bool(move.promotion))
    vector[CHECK_OFFSET] = float(board.gives_check(move))
    vector[CASTLING_OFFSET] = float(board.is_castling(move))
    return vector


def current_priority(board: chess.Board, move: chess.Move) -> tuple[int, int, str]:
    victim = board.piece_type_at(move.to_square) or 0
    attacker = board.piece_type_at(move.from_square) or 0
    return (int(bool(move.promotion)), victim * 10 - attacker, move.uci())


def grouped_candidates(
    rows: list[dict[str, Any]], label_clip: float
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["game_id"]), int(row["parent_ply"])), []).append(row)
    groups: list[dict[str, Any]] = []
    for (game_id, parent_ply), items in sorted(grouped.items()):
        parents = [row for row in items if row["source"] == "parent"]
        children = [row for row in items if row["source"] != "parent"]
        if len(parents) != 1 or len(children) < 2:
            continue
        parent_fen = str(parents[0]["fen"])
        candidates = [
            {
                "move": move_between(parent_fen, str(child["fen"])),
                "label": float(np.clip(float(child["label"]), -label_clip, label_clip)),
            }
            for child in children
        ]
        groups.append(
            {
                "game_id": game_id,
                "parent_ply": parent_ply,
                "board": chess.Board(parent_fen),
                "candidates": candidates,
            }
        )
    if not groups:
        raise ValueError("active dataset produced no move-ordering groups")
    return groups


def pairwise_training(
    groups: list[dict[str, Any]], training_games: set[int]
) -> tuple[np.ndarray, np.ndarray]:
    design: list[np.ndarray] = []
    margins: list[float] = []
    for group in groups:
        if int(group["game_id"]) not in training_games:
            continue
        board = group["board"]
        candidates = group["candidates"]
        best = min(candidates, key=lambda item: (float(item["label"]), item["move"].uci()))
        best_features = move_features(board, best["move"])
        for alternative in candidates:
            margin = float(alternative["label"] - best["label"])
            if alternative is best or margin <= 0.0:
                continue
            design.append(best_features - move_features(board, alternative["move"]))
            margins.append(margin)
    if not design:
        raise ValueError("active dataset produced no move-ordering pairs")
    return np.vstack(design), np.asarray(margins)


def fit_ordering(design: np.ndarray, margins: np.ndarray, penalty: float) -> np.ndarray:
    regularizer = np.eye(design.shape[1], dtype=np.float64) * penalty
    return np.linalg.solve(design.T @ design + regularizer, design.T @ margins)


def ordering_metrics(
    groups: list[dict[str, Any]], validation_games: set[int], weights: np.ndarray
) -> dict[str, float | int]:
    total = learned_correct = baseline_correct = 0
    learned_reciprocal = baseline_reciprocal = 0.0
    for group in groups:
        if int(group["game_id"]) not in validation_games:
            continue
        board = group["board"]
        candidates = group["candidates"]
        truth = min(candidates, key=lambda item: (float(item["label"]), item["move"].uci()))
        learned = sorted(
            candidates,
            key=lambda item: (
                float(move_features(board, item["move"]) @ weights),
                item["move"].uci(),
            ),
            reverse=True,
        )
        baseline = sorted(
            candidates,
            key=lambda item: current_priority(board, item["move"]),
            reverse=True,
        )
        truth_uci = truth["move"].uci()
        learned_rank = next(
            index for index, item in enumerate(learned, 1) if item["move"].uci() == truth_uci
        )
        baseline_rank = next(
            index for index, item in enumerate(baseline, 1) if item["move"].uci() == truth_uci
        )
        total += 1
        learned_correct += int(learned_rank == 1)
        baseline_correct += int(baseline_rank == 1)
        learned_reciprocal += 1.0 / learned_rank
        baseline_reciprocal += 1.0 / baseline_rank
    if total == 0:
        raise ValueError("validation split produced no move-ordering groups")
    return {
        "validation_groups": total,
        "validation_top1_accuracy": learned_correct / total,
        "baseline_validation_top1_accuracy": baseline_correct / total,
        "validation_mean_reciprocal_rank": learned_reciprocal / total,
        "baseline_validation_mean_reciprocal_rank": baseline_reciprocal / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-dataset", type=Path, required=True)
    parser.add_argument("--ridge-penalty", type=float, default=1000.0)
    parser.add_argument("--label-clip", type=float, default=1500.0)
    parser.add_argument("--output", type=Path, default=Path("weights/ordering.json"))
    args = parser.parse_args()
    if args.ridge_penalty <= 0 or args.label_clip <= 0:
        parser.error("ridge penalty and label clip must be positive")
    rows, metadata = load_active_dataset(args.active_dataset)
    training_games, validation_games = split_game_ids(rows)
    groups = grouped_candidates(rows, args.label_clip)
    design, margins = pairwise_training(groups, training_games)
    weights = fit_ordering(design, margins, args.ridge_penalty)
    prediction = design @ weights
    metrics: dict[str, float | int] = {
        "training_pairs": len(design),
        "training_margin_rmse": root_mean_square_error(margins, prediction),
        **ordering_metrics(groups, validation_games, weights),
    }
    script = Path(__file__)
    payload = {
        "schema_version": 1,
        "model_kind": "game_split_active_move_ordering_policy",
        "layout": {
            "from_piece_square": [FROM_OFFSET, TO_OFFSET],
            "to_piece_square": [TO_OFFSET, CAPTURE_OFFSET],
            "capture": CAPTURE_OFFSET,
            "promotion": PROMOTION_OFFSET,
            "check": CHECK_OFFSET,
            "castling": CASTLING_OFFSET,
            "victim_piece": [VICTIM_OFFSET, FEATURES],
        },
        "training": {
            "method": "pairwise teacher-margin move ordering",
            "split": "complete game ids; no position-level leakage",
            "split_seed": ACTIVE_SPLIT_SEED,
            "training_games": sorted(training_games),
            "validation_games": sorted(validation_games),
            "ridge_penalty": args.ridge_penalty,
            "label_clip_centipawns": args.label_clip,
            "active_dataset_sha256": metadata["dataset_sha256"],
            "teacher_name": metadata["teacher_name"],
            "teacher_binary_sha256": metadata["teacher_binary_sha256"],
            "protected_opening_list_used": False,
            "script": str(script.relative_to(Path.cwd())),
            "script_sha256": file_sha256(script),
        },
        "weights": [float(value) for value in weights],
        **metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
