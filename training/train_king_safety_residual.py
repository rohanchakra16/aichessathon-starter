#!/usr/bin/env python3
"""Fit a tiny king-safety residual while freezing the accepted 770 weights.

Training uses game-grouped teacher move pairs. A second, independent active
dataset is reported separately and is never used to select the ridge penalty.
"""

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

from training.train_active_psqt_finetune import game_folds, ranking_totals  # noqa: E402
from training.train_active_residual_evaluator import (  # noqa: E402
    baseline_prediction,
    load_active_dataset,
)
from training.train_pairwise_psqt_finetune import (  # noqa: E402
    pairwise_design,
    rmse,
)
from training.train_stockfish_evaluator import (  # noqa: E402
    MAX_PHASE,
    PHASE_VALUES,
    file_sha256,
)

FEATURE_NAMES = (
    "midgame_pawn_shield",
    "midgame_zone_attackers",
    "midgame_open_king_files",
    "endgame_pawn_shield",
    "endgame_zone_attackers",
    "endgame_open_king_files",
)
LOWER_BOUNDS = np.asarray((0.0, -60.0, -30.0, 0.0, -20.0, -15.0))
UPPER_BOUNDS = np.asarray((40.0, 0.0, 0.0, 10.0, 0.0, 0.0))


def phase(board: chess.Board) -> float:
    current = sum(
        PHASE_VALUES[piece_type] * len(board.pieces(piece_type, colour))
        for piece_type in range(chess.PAWN, chess.KING + 1)
        for colour in chess.COLORS
    )
    return float(min(1.0, current / MAX_PHASE))


def king_safety_values(
    board: chess.Board, colour: chess.Color
) -> tuple[float, float, float]:
    """Return shield pawns, distinct zone attackers, and nearby open files."""
    king_square = board.king(colour)
    if king_square is None:
        return 0.0, 0.0, 0.0
    pawns = board.pieces(chess.PAWN, colour)
    king_file = chess.square_file(king_square)
    king_rank = chess.square_rank(king_square)
    shield_rank = king_rank + (1 if colour == chess.WHITE else -1)
    shield = 0
    if 0 <= shield_rank < 8:
        for file_index in range(max(0, king_file - 1), min(8, king_file + 2)):
            shield += int(
                bool(pawns & chess.BB_SQUARES[chess.square(file_index, shield_rank)])
            )

    zone = chess.BB_KING_ATTACKS[king_square] | chess.BB_SQUARES[king_square]
    attackers = 0
    for square in chess.scan_forward(zone):
        attackers |= board.attackers_mask(not colour, square)

    open_files = sum(
        not bool(board.pawns & chess.BB_FILES[file_index])
        for file_index in range(max(0, king_file - 1), min(8, king_file + 2))
    )
    return float(shield), float(attackers.bit_count()), float(open_files)


def king_safety_features(board: chess.Board) -> np.ndarray:
    white = np.asarray(king_safety_values(board, chess.WHITE))
    black = np.asarray(king_safety_values(board, chess.BLACK))
    difference = white - black
    if board.turn == chess.BLACK:
        difference = -difference
    blend = phase(board)
    features: np.ndarray = np.concatenate(
        (difference * blend, difference * (1.0 - blend))
    )
    return features


def fit_coefficients(
    design: np.ndarray, targets: np.ndarray, penalty: float
) -> np.ndarray:
    regularizer = np.eye(design.shape[1], dtype=np.float64) * penalty
    coefficients: np.ndarray = np.linalg.solve(
        design.T @ design + regularizer,
        design.T @ targets,
    )
    clipped: np.ndarray = np.clip(coefficients, LOWER_BOUNDS, UPPER_BOUNDS)
    return clipped


def cross_validate(
    rows: list[dict[str, Any]],
    position_design: np.ndarray,
    labels: np.ndarray,
    baseline: np.ndarray,
    pair_design_matrix: np.ndarray,
    pair_targets: np.ndarray,
    teacher_margins: np.ndarray,
    baseline_margins: np.ndarray,
    pair_game_ids: np.ndarray,
    penalty: float,
    fold_count: int,
) -> dict[str, Any]:
    row_game_ids = np.asarray([int(row["game_id"]) for row in rows])
    totals = {
        "groups": 0,
        "baseline_top1": 0,
        "candidate_top1": 0,
        "baseline_reciprocal": 0.0,
        "candidate_reciprocal": 0.0,
        "pairs": 0,
        "baseline_squared_error": 0.0,
        "candidate_squared_error": 0.0,
    }
    for validation_games in game_folds(rows, fold_count):
        training_pairs = ~np.isin(pair_game_ids, list(validation_games))
        validation_pairs = ~training_pairs
        validation_rows = np.isin(row_game_ids, list(validation_games))
        coefficients = fit_coefficients(
            pair_design_matrix[training_pairs], pair_targets[training_pairs], penalty
        )
        candidate = baseline + position_design @ coefficients
        groups, candidate_top1, candidate_reciprocal = ranking_totals(
            rows, labels, candidate, validation_rows
        )
        _, baseline_top1, baseline_reciprocal = ranking_totals(
            rows, labels, baseline, validation_rows
        )
        candidate_margins = baseline_margins[validation_pairs] + (
            pair_design_matrix[validation_pairs] @ coefficients
        )
        baseline_error = teacher_margins[validation_pairs] - baseline_margins[validation_pairs]
        candidate_error = teacher_margins[validation_pairs] - candidate_margins
        pair_count = int(validation_pairs.sum())
        totals["groups"] += groups
        totals["baseline_top1"] += baseline_top1
        totals["candidate_top1"] += candidate_top1
        totals["baseline_reciprocal"] += baseline_reciprocal
        totals["candidate_reciprocal"] += candidate_reciprocal
        totals["pairs"] += pair_count
        totals["baseline_squared_error"] += float(baseline_error @ baseline_error)
        totals["candidate_squared_error"] += float(candidate_error @ candidate_error)
    groups = int(totals["groups"])
    pairs = int(totals["pairs"])
    return {
        "fold_count": fold_count,
        "validation_groups": groups,
        "validation_pairs": pairs,
        "baseline_margin_rmse": float(
            np.sqrt(float(totals["baseline_squared_error"]) / pairs)
        ),
        "candidate_margin_rmse": float(
            np.sqrt(float(totals["candidate_squared_error"]) / pairs)
        ),
        "baseline_top1": int(totals["baseline_top1"]) / groups,
        "candidate_top1": int(totals["candidate_top1"]) / groups,
        "baseline_mrr": float(totals["baseline_reciprocal"]) / groups,
        "candidate_mrr": float(totals["candidate_reciprocal"]) / groups,
    }


def danger_score(board: chess.Board) -> float:
    values = (
        king_safety_values(board, chess.WHITE),
        king_safety_values(board, chess.BLACK),
    )
    return max(
        (3.0 - shield) + 2.0 * attackers + open_files
        for shield, attackers, open_files in values
    )


def ranking_metrics(
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    selected: np.ndarray,
) -> dict[str, float | int]:
    groups, baseline_top1, baseline_reciprocal = ranking_totals(
        rows, labels, baseline, selected
    )
    _, candidate_top1, candidate_reciprocal = ranking_totals(
        rows, labels, candidate, selected
    )
    return {
        "groups": groups,
        "baseline_top1": baseline_top1 / groups,
        "candidate_top1": candidate_top1 / groups,
        "baseline_mrr": baseline_reciprocal / groups,
        "candidate_mrr": candidate_reciprocal / groups,
    }


def independent_validation(
    path: Path,
    model: dict[str, Any],
    coefficients: np.ndarray,
) -> dict[str, Any]:
    rows, metadata = load_active_dataset(path)
    positions = [chess.Board(row["fen"]) for row in rows]
    design = np.vstack([king_safety_features(board) for board in positions])
    label_clip = float(model["training"]["label_clip_centipawns"])
    labels = np.clip(
        np.asarray([float(row["label"]) for row in rows]), -label_clip, label_clip
    )
    baseline = baseline_prediction(positions, model)
    candidate = baseline + design @ coefficients
    all_rows: np.ndarray = np.ones(len(rows), dtype=bool)

    group_danger: dict[tuple[int, int], float] = {}
    for row, board in zip(rows, positions, strict=True):
        if row["source"] == "parent":
            group_danger[(int(row["game_id"]), int(row["parent_ply"]))] = danger_score(
                board
            )
    threshold = float(np.median(list(group_danger.values())))
    dangerous_groups = {
        key for key, value in group_danger.items() if value >= threshold
    }
    dangerous_rows: np.ndarray = np.asarray(
        [
            (int(row["game_id"]), int(row["parent_ply"])) in dangerous_groups
            for row in rows
        ],
        dtype=bool,
    )
    return {
        "dataset_sha256": metadata["dataset_sha256"],
        "danger_threshold": threshold,
        "all": ranking_metrics(rows, labels, baseline, candidate, all_rows),
        "high_danger": ranking_metrics(
            rows, labels, baseline, candidate, dangerous_rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, default=Path("weights/model.json"))
    parser.add_argument("--training-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument(
        "--ridge-penalties",
        type=float,
        nargs="+",
        default=(1.0, 10.0, 100.0, 1000.0, 10000.0),
    )
    parser.add_argument("--margin-clip", type=float, default=300.0)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(penalty <= 0.0 for penalty in args.ridge_penalties):
        parser.error("ridge penalties must be positive")

    model = json.loads(args.base_model.read_text())
    if len(model["weights"]) != 770:
        raise ValueError("base model must be the accepted 770-weight evaluator")
    rows, metadata = load_active_dataset(args.training_dataset)
    positions = [chess.Board(row["fen"]) for row in rows]
    design = np.vstack([king_safety_features(board) for board in positions])
    label_clip = float(model["training"]["label_clip_centipawns"])
    labels = np.clip(
        np.asarray([float(row["label"]) for row in rows]), -label_clip, label_clip
    )
    baseline = baseline_prediction(positions, model)
    pairs = pairwise_design(rows, design, labels, baseline, args.margin_clip)
    pair_matrix, targets, teacher_margins, baseline_margins, pair_game_ids = pairs

    candidates = []
    for penalty in sorted(set(args.ridge_penalties)):
        validation = cross_validate(
            rows,
            design,
            labels,
            baseline,
            pair_matrix,
            targets,
            teacher_margins,
            baseline_margins,
            pair_game_ids,
            penalty,
            args.folds,
        )
        candidates.append({"ridge_penalty": penalty, "cross_validation": validation})
    selected = max(
        candidates,
        key=lambda item: (
            item["cross_validation"]["candidate_mrr"],
            item["cross_validation"]["candidate_top1"],
            -item["cross_validation"]["candidate_margin_rmse"],
            item["ridge_penalty"],
        ),
    )
    penalty = float(selected["ridge_penalty"])
    coefficients = fit_coefficients(pair_matrix, targets, penalty)
    independent = independent_validation(args.validation_dataset, model, coefficients)

    layout = dict(model["layout"])
    layout.update(
        {
            "king_safety_offset": 770,
            "king_safety_feature_names": list(FEATURE_NAMES),
        }
    )
    script = Path(__file__)
    payload = {
        "schema_version": 6,
        "model_kind": "frozen_psqt_with_king_safety_residual",
        "materially_drives": "all non-terminal search leaf evaluations",
        "layout": layout,
        "training": {
            **model["training"],
            "method": "frozen accepted evaluator plus constrained king-safety residual",
            "selection": (
                "game-grouped cross-validation on training set; "
                "independent game-set confirmation"
            ),
            "ridge_penalty": penalty,
            "ridge_penalties_considered": sorted(set(args.ridge_penalties)),
            "margin_clip_centipawns": args.margin_clip,
            "training_dataset_sha256": metadata["dataset_sha256"],
            "validation_dataset_sha256": independent["dataset_sha256"],
            "base_model_sha256": file_sha256(args.base_model),
            "script": str(script.relative_to(Path.cwd())),
            "script_sha256": file_sha256(script),
            "external_engine_used": True,
            "protected_opening_list_used": False,
        },
        "bias": float(model["bias"]),
        "weights": [*map(float, model["weights"]), *map(float, coefficients)],
        "king_safety_coefficients": dict(
            zip(FEATURE_NAMES, map(float, coefficients), strict=True)
        ),
        "penalty_cross_validation": candidates,
        "selected_cross_validation": selected["cross_validation"],
        "independent_validation": independent,
        "baseline_margin_rmse": rmse(teacher_margins, baseline_margins),
        "final_margin_rmse": rmse(
            teacher_margins, baseline_margins + pair_matrix @ coefficients
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "king_safety_coefficients": payload["king_safety_coefficients"],
                "selected_ridge_penalty": penalty,
                "selected_cross_validation": selected["cross_validation"],
                "independent_validation": independent,
                "baseline_margin_rmse": payload["baseline_margin_rmse"],
                "final_margin_rmse": payload["final_margin_rmse"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
