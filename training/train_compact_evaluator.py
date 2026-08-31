#!/usr/bin/env python3
"""Train a compact tapered chess evaluator from reproducible offline labels.

The runtime feature set is deliberately small enough for Python search while
covering material, pawn structure, mobility, king safety, and basic positional
terms. The external engine is a development-only teacher and is never shipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import chess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_stockfish_evaluator import (  # noqa: E402
    MAX_PHASE,
    PHASE_VALUES,
    SEED,
    file_sha256,
    generated_positions,
    root_mean_square_error,
    teacher_labels,
)

MATERIAL_NAMES = ("pawn", "knight", "bishop", "rook", "queen")
POSITIONAL_NAMES = (
    "pawn_advancement",
    "passed_pawns",
    "isolated_pawns",
    "doubled_pawns",
    "pawn_islands",
    "bishop_pair",
    "knight_mobility",
    "bishop_mobility",
    "rook_mobility",
    "queen_mobility",
    "rook_open_files",
    "rook_semi_open_files",
    "king_shield",
    "castling_rights",
    "center_pawns",
    "center_control",
    "king_centrality",
)
FEATURE_NAMES = (
    *MATERIAL_NAMES,
    *(f"midgame_{name}" for name in POSITIONAL_NAMES),
    *(f"endgame_{name}" for name in POSITIONAL_NAMES),
)
FEATURES = len(FEATURE_NAMES)
CENTER = chess.BB_D4 | chess.BB_E4 | chess.BB_D5 | chess.BB_E5
MATERIAL_PRIOR = (100.0, 320.0, 330.0, 500.0, 900.0)
MIDGAME_PRIOR = (
    2.0,
    20.0,
    -10.0,
    -12.0,
    -6.0,
    30.0,
    4.0,
    3.0,
    2.0,
    1.0,
    15.0,
    8.0,
    12.0,
    15.0,
    12.0,
    3.0,
    -12.0,
)
ENDGAME_PRIOR = (
    8.0,
    50.0,
    -8.0,
    -10.0,
    -4.0,
    40.0,
    4.0,
    4.0,
    3.0,
    2.0,
    10.0,
    5.0,
    0.0,
    0.0,
    4.0,
    2.0,
    12.0,
)
MIDGAME_BOUNDS = (
    (0.0, 8.0),
    (10.0, 60.0),
    (-30.0, 0.0),
    (-30.0, 0.0),
    (-15.0, 0.0),
    (10.0, 60.0),
    (0.0, 10.0),
    (0.0, 8.0),
    (0.0, 6.0),
    (0.0, 3.0),
    (0.0, 30.0),
    (0.0, 20.0),
    (0.0, 25.0),
    (0.0, 30.0),
    (0.0, 25.0),
    (0.0, 8.0),
    (-30.0, 0.0),
)
ENDGAME_BOUNDS = (
    (2.0, 15.0),
    (20.0, 100.0),
    (-25.0, 0.0),
    (-25.0, 0.0),
    (-15.0, 0.0),
    (15.0, 70.0),
    (0.0, 10.0),
    (0.0, 10.0),
    (0.0, 8.0),
    (0.0, 5.0),
    (0.0, 25.0),
    (0.0, 15.0),
    (0.0, 5.0),
    (0.0, 5.0),
    (0.0, 15.0),
    (0.0, 6.0),
    (5.0, 30.0),
)


def passed_mask(colour: chess.Color, square: chess.Square) -> int:
    direction = 1 if colour == chess.WHITE else -1
    rank = chess.square_rank(square) + direction
    file = chess.square_file(square)
    mask = 0
    while 0 <= rank < 8:
        for candidate_file in range(max(0, file - 1), min(8, file + 2)):
            mask |= chess.BB_SQUARES[chess.square(candidate_file, rank)]
        rank += direction
    return mask


PASSED_MASKS = {
    colour: tuple(passed_mask(colour, square) for square in chess.SQUARES)
    for colour in chess.COLORS
}


def phase(board: chess.Board) -> float:
    current = sum(
        PHASE_VALUES[piece_type] * len(board.pieces(piece_type, colour))
        for piece_type in range(chess.PAWN, chess.KING + 1)
        for colour in chess.COLORS
    )
    return min(1.0, current / MAX_PHASE)


def positional_values(board: chess.Board, colour: chess.Color) -> np.ndarray:
    pawns = board.pieces(chess.PAWN, colour)
    enemy_pawns = board.pieces(chess.PAWN, not colour)
    file_counts = [len(pawns & chess.BB_FILES[file]) for file in range(8)]
    occupied_files = [count > 0 for count in file_counts]
    pawn_advancement = 0
    passed_pawns = 0
    isolated_pawns = 0
    for square in pawns:
        rank = chess.square_rank(square)
        relative_rank = rank if colour == chess.WHITE else 7 - rank
        pawn_advancement += max(0, relative_rank - 1)
        if not enemy_pawns & PASSED_MASKS[colour][square]:
            passed_pawns += 1
        file = chess.square_file(square)
        adjacent = occupied_files[max(0, file - 1) : file] + occupied_files[file + 1 : file + 2]
        if not any(adjacent):
            isolated_pawns += 1
    doubled_pawns = sum(max(0, count - 1) for count in file_counts)
    pawn_islands = sum(
        occupied and (file == 0 or not occupied_files[file - 1])
        for file, occupied in enumerate(occupied_files)
    )

    own_occupied = board.occupied_co[colour]
    mobility: list[int] = []
    attacks = 0
    for piece_type in range(chess.PAWN, chess.KING + 1):
        piece_mobility = 0
        for square in board.pieces(piece_type, colour):
            destinations = int(board.attacks(square)) & ~own_occupied
            attacks |= destinations
            if piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
                piece_mobility += destinations.bit_count()
        if piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            mobility.append(piece_mobility)

    open_files = 0
    semi_open_files = 0
    all_pawns = board.pawns
    for square in board.pieces(chess.ROOK, colour):
        file_mask = chess.BB_FILES[chess.square_file(square)]
        if not all_pawns & file_mask:
            open_files += 1
        elif not pawns & file_mask:
            semi_open_files += 1

    king_square = board.king(colour)
    king_shield = 0
    king_centrality = 0.0
    if king_square is not None:
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        shield_rank = king_rank + (1 if colour == chess.WHITE else -1)
        if 0 <= shield_rank < 8:
            for file in range(max(0, king_file - 1), min(8, king_file + 2)):
                king_shield += int(bool(pawns & chess.BB_SQUARES[chess.square(file, shield_rank)]))
        king_centrality = 3.5 - (
            abs(king_file - 3.5) + abs(king_rank - 3.5)
        ) / 2.0

    return np.asarray(
        [
            pawn_advancement,
            passed_pawns,
            isolated_pawns,
            doubled_pawns,
            pawn_islands,
            int(len(board.pieces(chess.BISHOP, colour)) >= 2),
            *mobility,
            open_files,
            semi_open_files,
            king_shield,
            int(board.has_kingside_castling_rights(colour))
            + int(board.has_queenside_castling_rights(colour)),
            len(pawns & CENTER),
            (attacks & CENTER).bit_count(),
            king_centrality,
        ],
        dtype=np.float64,
    )


def features(board: chess.Board) -> np.ndarray:
    side = board.turn
    material = np.asarray(
        [
            len(board.pieces(piece_type, side))
            - len(board.pieces(piece_type, not side))
            for piece_type in range(chess.PAWN, chess.QUEEN + 1)
        ],
        dtype=np.float64,
    )
    positional = positional_values(board, side) - positional_values(board, not side)
    current_phase = phase(board)
    return np.concatenate(
        (material, positional * current_phase, positional * (1.0 - current_phase))
    )


def coefficient_prior() -> np.ndarray:
    return np.asarray((10.0, *MATERIAL_PRIOR, *MIDGAME_PRIOR, *ENDGAME_PRIOR))


def dataset_digest(positions: list[chess.Board], labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    for board, label in zip(positions, labels, strict=True):
        digest.update(f"{board.fen()}\t{label:.1f}\n".encode())
    return digest.hexdigest()


def save_dataset(
    path: Path,
    positions: list[chess.Board],
    labels: np.ndarray,
    engine_path: Path,
    nodes: int,
    digest: str,
) -> None:
    payload = {
        "schema_version": 1,
        "seed": SEED,
        "nodes_per_position": nodes,
        "teacher_binary_sha256": file_sha256(engine_path),
        "dataset_sha256": digest,
        "rows": [
            {"fen": board.fen(), "label": float(label)}
            for board, label in zip(positions, labels, strict=True)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_dataset(
    path: Path, engine_path: Path, examples: int, nodes: int
) -> tuple[list[chess.Board], np.ndarray, str]:
    payload = json.loads(path.read_text())
    expected = {
        "schema_version": 1,
        "seed": SEED,
        "nodes_per_position": nodes,
        "teacher_binary_sha256": file_sha256(engine_path),
    }
    actual = {key: payload.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"dataset cache metadata mismatch: {actual!r} != {expected!r}")
    rows = payload.get("rows", [])
    if len(rows) != examples:
        raise ValueError(f"dataset cache has {len(rows)} rows; expected {examples}")
    positions = [chess.Board(row["fen"]) for row in rows]
    labels = np.asarray([float(row["label"]) for row in rows], dtype=np.float64)
    digest = dataset_digest(positions, labels)
    if digest != payload.get("dataset_sha256"):
        raise ValueError("dataset cache digest mismatch")
    return positions, labels, digest


def ridge_fit(design: np.ndarray, labels: np.ndarray, penalty: float) -> np.ndarray:
    material_count = len(MATERIAL_PRIOR)
    prior = coefficient_prior()
    material = np.asarray(MATERIAL_PRIOR)
    residual_labels = labels - design[:, :material_count] @ material
    residual_design = np.column_stack((np.ones(len(design)), design[:, material_count:]))
    regularizer = np.eye(residual_design.shape[1], dtype=np.float64) * penalty
    residual_prior = np.concatenate(([prior[0]], prior[1 + material_count :]))
    residual_coefficients = np.linalg.solve(
        residual_design.T @ residual_design + regularizer,
        residual_design.T @ residual_labels + penalty * residual_prior,
    )
    coefficients = np.concatenate(
        ([residual_coefficients[0]], material, residual_coefficients[1:])
    )
    bounds = (
        (-100.0, 100.0),
        *((value, value) for value in MATERIAL_PRIOR),
        *MIDGAME_BOUNDS,
        *ENDGAME_BOUNDS,
    )
    lower = np.asarray([bound[0] for bound in bounds])
    upper = np.asarray([bound[1] for bound in bounds])
    return np.clip(coefficients, lower, upper)


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
        "schema_version": 3,
        "model_kind": "compact_tapered_offline_teacher_evaluator",
        "materially_drives": "all non-terminal search leaf evaluations",
        "features": list(FEATURE_NAMES),
        "training": {
            "method": "deterministic quiet positions labelled by an offline engine",
            "seed": SEED,
            "examples": args.examples,
            "nodes_per_position": args.nodes,
            "ridge_penalty": args.ridge_penalty,
            "coefficient_prior": coefficient_prior().tolist(),
            "midgame_coefficient_bounds": [list(bound) for bound in MIDGAME_BOUNDS],
            "endgame_coefficient_bounds": [list(bound) for bound in ENDGAME_BOUNDS],
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
    parser.add_argument("--ridge-penalty", type=float, default=1000.0)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--dataset-cache", type=Path)
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
    if args.dataset_cache is not None and args.dataset_cache.exists():
        positions, labels, dataset_sha256 = load_dataset(
            args.dataset_cache, engine_path, args.examples, args.nodes
        )
    else:
        positions = generated_positions(args.examples)
        labels, dataset_sha256 = teacher_labels(
            positions, engine_path, args.nodes, args.progress_every
        )
        if args.dataset_cache is not None:
            save_dataset(
                args.dataset_cache,
                positions,
                labels,
                engine_path,
                args.nodes,
                dataset_sha256,
            )
    coefficients, metrics = train(positions, labels, args.ridge_penalty)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            model_payload(coefficients, metrics, args, engine_path, dataset_sha256),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
