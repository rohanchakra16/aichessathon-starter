#!/usr/bin/env python3
"""Fit a small strategic residual on game-grouped champion-disagreement data.

The proven tapered piece-square model remains fixed. Only features it cannot
represent relationally are learned, and validation holds out complete games.
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

from training.generate_active_learning_dataset import dataset_digest  # noqa: E402
from training.train_compact_evaluator import positional_values  # noqa: E402
from training.train_stockfish_evaluator import (  # noqa: E402
    MAX_PHASE,
    PHASE_VALUES,
    file_sha256,
    root_mean_square_error,
)
from training.train_stockfish_evaluator import (  # noqa: E402
    features as piece_square_features,
)

ACTIVE_SPLIT_SEED = 2026090102
STRATEGIC_INDICES = (1, 2, 3, 5, 10, 11, 12)
STRATEGIC_NAMES = (
    "passed_pawns",
    "isolated_pawns",
    "doubled_pawns",
    "bishop_pair",
    "rook_open_files",
    "rook_semi_open_files",
    "king_shield",
)
STRATEGIC_FEATURE_NAMES = (
    *(f"midgame_{name}" for name in STRATEGIC_NAMES),
    *(f"endgame_{name}" for name in STRATEGIC_NAMES),
)
STRATEGIC_PRIOR = np.zeros(len(STRATEGIC_FEATURE_NAMES), dtype=np.float64)
STRATEGIC_BOUNDS = (
    (0.0, 60.0),
    (-30.0, 0.0),
    (-30.0, 0.0),
    (0.0, 60.0),
    (0.0, 30.0),
    (0.0, 20.0),
    (0.0, 25.0),
    (0.0, 100.0),
    (-25.0, 0.0),
    (-25.0, 0.0),
    (0.0, 70.0),
    (0.0, 25.0),
    (0.0, 15.0),
    (0.0, 5.0),
)


def phase(board: chess.Board) -> float:
    current = sum(
        PHASE_VALUES[piece_type] * len(board.pieces(piece_type, colour))
        for piece_type in range(chess.PAWN, chess.KING + 1)
        for colour in chess.COLORS
    )
    return min(1.0, current / MAX_PHASE)


def strategic_features(board: chess.Board) -> np.ndarray:
    positional = positional_values(board, board.turn) - positional_values(
        board, not board.turn
    )
    selected = positional[np.asarray(STRATEGIC_INDICES)]
    blend = phase(board)
    return np.concatenate((selected * blend, selected * (1.0 - blend)))


def load_active_dataset(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text())
    required = {
        "schema_version": 1,
        "kind": "champion_disagreement_active_learning_dataset",
        "protected_opening_list_used": False,
        "game_grouped": True,
    }
    actual = {key: payload.get(key) for key in required}
    if actual != required:
        raise ValueError(f"active dataset metadata mismatch: {actual!r} != {required!r}")
    rows = payload.get("rows", [])
    if not rows or any("game_id" not in row for row in rows):
        raise ValueError("active dataset has no game-grouped rows")
    if len(rows) != int(payload.get("rows_count", -1)):
        raise ValueError("active dataset row count mismatch")
    if dataset_digest(rows) != payload.get("dataset_sha256"):
        raise ValueError("active dataset digest mismatch")
    return rows, payload


def load_base_dataset(path: Path) -> tuple[list[chess.Board], np.ndarray, dict[str, Any]]:
    payload = json.loads(path.read_text())
    if payload.get("kind") != "engine_guided_selfplay_evaluation_dataset":
        raise ValueError("base dataset has the wrong kind")
    if payload.get("protected_opening_list_used") is not False:
        raise ValueError("base dataset provenance is not independent")
    rows = payload.get("rows", [])
    if not rows:
        raise ValueError("base dataset has no rows")
    positions = [chess.Board(row["fen"]) for row in rows]
    labels = np.asarray([float(row["label"]) for row in rows], dtype=np.float64)
    return positions, labels, payload


def baseline_prediction(
    positions: list[chess.Board], model: dict[str, Any]
) -> np.ndarray:
    weights = np.asarray(model["weights"], dtype=np.float64)
    if len(weights) != 770:
        raise ValueError("base model must be the 770-weight tapered piece-square champion")
    design = np.vstack([piece_square_features(board) for board in positions])
    return float(model["bias"]) + design @ weights


def split_game_ids(rows: list[dict[str, Any]]) -> tuple[set[int], set[int]]:
    game_ids = sorted({int(row["game_id"]) for row in rows})
    if len(game_ids) < 5:
        raise ValueError("at least five games are required for grouped validation")
    rng = np.random.default_rng(ACTIVE_SPLIT_SEED)
    shuffled = list(rng.permutation(game_ids))
    validation_count = max(1, len(shuffled) // 5)
    validation = {int(game_id) for game_id in shuffled[:validation_count]}
    training = set(game_ids) - validation
    return training, validation


def fit_residual(
    design: np.ndarray,
    residual_labels: np.ndarray,
    penalty: float,
) -> np.ndarray:
    regularizer = np.eye(design.shape[1], dtype=np.float64) * penalty
    coefficients = np.linalg.solve(
        design.T @ design + regularizer,
        design.T @ residual_labels + penalty * STRATEGIC_PRIOR,
    )
    lower = np.asarray([bound[0] for bound in STRATEGIC_BOUNDS])
    upper = np.asarray([bound[1] for bound in STRATEGIC_BOUNDS])
    return np.clip(coefficients, lower, upper)


def pairwise_samples(
    rows: list[dict[str, Any]],
    design: np.ndarray,
    labels: np.ndarray,
    baseline: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build alternative-minus-best margins, grouped by their source game."""
    groups: dict[tuple[int, int], list[int]] = {}
    for index, row in enumerate(rows):
        if row["source"] != "parent":
            groups.setdefault((int(row["game_id"]), int(row["parent_ply"])), []).append(
                index
            )
    pair_design: list[np.ndarray] = []
    residual_margins: list[float] = []
    game_ids: list[int] = []
    target_margins: list[float] = []
    baseline_margins: list[float] = []
    for (game_id, _), indices in sorted(groups.items()):
        if len(indices) < 2:
            continue
        best = min(indices, key=lambda index: (labels[index], index))
        for alternative in indices:
            target_margin = float(labels[alternative] - labels[best])
            if alternative == best or target_margin <= 0.0:
                continue
            baseline_margin = float(baseline[alternative] - baseline[best])
            pair_design.append(design[alternative] - design[best])
            residual_margins.append(target_margin - baseline_margin)
            game_ids.append(game_id)
            target_margins.append(target_margin)
            baseline_margins.append(baseline_margin)
    if not pair_design:
        raise ValueError("active dataset produced no non-tied child pairs")
    return (
        np.vstack(pair_design),
        np.asarray(residual_margins),
        np.asarray(game_ids),
        np.asarray(target_margins),
        np.asarray(baseline_margins),
    )


def train(
    base_positions: list[chess.Board],
    base_labels: np.ndarray,
    active_rows: list[dict[str, Any]],
    base_model: dict[str, Any],
    penalty: float,
    active_weight: float,
    objective: str,
) -> tuple[np.ndarray, dict[str, float | int | list[int]]]:
    active_positions = [chess.Board(row["fen"]) for row in active_rows]
    active_labels = np.asarray([float(row["label"]) for row in active_rows])
    label_clip = float(base_model["training"]["label_clip_centipawns"])
    base_labels = np.clip(base_labels, -label_clip, label_clip)
    active_labels = np.clip(active_labels, -label_clip, label_clip)
    training_games, validation_games = split_game_ids(active_rows)
    base_design = np.vstack([strategic_features(board) for board in base_positions])
    active_design = np.vstack([strategic_features(board) for board in active_positions])
    base_baseline = baseline_prediction(base_positions, base_model)
    base_residual = base_labels - base_baseline
    active_baseline = baseline_prediction(active_positions, base_model)
    active_residual = active_labels - active_baseline
    active_scale = np.sqrt(active_weight)
    if objective == "pointwise":
        active_training = np.asarray(
            [int(row["game_id"]) in training_games for row in active_rows], dtype=bool
        )
        active_validation = ~active_training
        fit_design = np.vstack(
            (base_design, active_design[active_training] * active_scale)
        )
        fit_labels = np.concatenate(
            (base_residual, active_residual[active_training] * active_scale)
        )
    elif objective == "pairwise":
        (
            margin_design,
            margin_residual,
            margin_games,
            target_margins,
            baseline_margins,
        ) = pairwise_samples(active_rows, active_design, active_labels, active_baseline)
        active_training = np.asarray(
            [int(game_id) in training_games for game_id in margin_games], dtype=bool
        )
        active_validation = ~active_training
        fit_design = np.vstack(
            (base_design, margin_design[active_training] * active_scale)
        )
        fit_labels = np.concatenate(
            (base_residual, margin_residual[active_training] * active_scale)
        )
    else:
        raise ValueError(f"unknown active objective: {objective}")
    coefficients = fit_residual(fit_design, fit_labels, penalty)

    base_prediction = base_baseline + base_design @ coefficients
    metrics: dict[str, float | int | list[int]] = {
        "base_training_examples": len(base_positions),
        "active_training_examples": int(active_training.sum()),
        "active_validation_examples": int(active_validation.sum()),
        "active_training_games": sorted(training_games),
        "active_validation_games": sorted(validation_games),
        "base_baseline_training_rmse": root_mean_square_error(
            base_labels, base_baseline
        ),
        "base_training_rmse": root_mean_square_error(base_labels, base_prediction),
    }
    if objective == "pointwise":
        active_prediction = active_baseline + active_design @ coefficients
        metrics.update(
            {
                "active_training_rmse": root_mean_square_error(
                    active_labels[active_training], active_prediction[active_training]
                ),
                "active_validation_rmse": root_mean_square_error(
                    active_labels[active_validation], active_prediction[active_validation]
                ),
                "active_baseline_validation_rmse": root_mean_square_error(
                    active_labels[active_validation], active_baseline[active_validation]
                ),
            }
        )
    else:
        predicted_margins = baseline_margins + margin_design @ coefficients
        metrics.update(
            {
                "active_training_margin_rmse": root_mean_square_error(
                    target_margins[active_training], predicted_margins[active_training]
                ),
                "active_validation_margin_rmse": root_mean_square_error(
                    target_margins[active_validation], predicted_margins[active_validation]
                ),
                "active_baseline_validation_margin_rmse": root_mean_square_error(
                    target_margins[active_validation], baseline_margins[active_validation]
                ),
                "active_validation_ranking_accuracy": float(
                    np.mean(predicted_margins[active_validation] > 0.0)
                ),
                "active_baseline_validation_ranking_accuracy": float(
                    np.mean(baseline_margins[active_validation] > 0.0)
                ),
            }
        )
    return coefficients, metrics


def model_payload(
    base_model: dict[str, Any],
    coefficients: np.ndarray,
    metrics: dict[str, float | int | list[int]],
    args: argparse.Namespace,
    active_metadata: dict[str, Any],
    base_metadata: dict[str, Any],
) -> dict[str, Any]:
    script = Path(__file__)
    base_weights = [float(value) for value in base_model["weights"]]
    return {
        "schema_version": 6,
        "model_kind": "active_game_split_strategic_residual_evaluator",
        "materially_drives": "all non-terminal search leaf evaluations",
        "layout": {
            **base_model["layout"],
            "strategic": [len(base_weights), len(base_weights) + len(coefficients)],
            "strategic_features": list(STRATEGIC_FEATURE_NAMES),
        },
        "training": {
            "method": "fixed champion PSQT plus active strategic residual",
            "split": "complete game ids; no position-level leakage",
            "split_seed": ACTIVE_SPLIT_SEED,
            "ridge_penalty": args.ridge_penalty,
            "active_example_weight": args.active_weight,
            "active_objective": args.objective,
            "label_clip_centipawns": base_model["training"]["label_clip_centipawns"],
            "coefficient_prior": STRATEGIC_PRIOR.tolist(),
            "coefficient_bounds": [list(bound) for bound in STRATEGIC_BOUNDS],
            "base_model_sha256": file_sha256(args.base_model),
            "base_dataset_sha256": base_metadata["dataset_sha256"],
            "active_dataset_sha256": active_metadata["dataset_sha256"],
            "active_champion_agent_sha256": active_metadata["champion_agent_sha256"],
            "active_champion_model_sha256": active_metadata["champion_model_sha256"],
            "external_engine_used": True,
            "teacher_name": active_metadata["teacher_name"],
            "teacher_binary_sha256": active_metadata["teacher_binary_sha256"],
            "protected_opening_list_used": False,
            "script": str(script.relative_to(Path.cwd())),
            "script_sha256": file_sha256(script),
        },
        "bias": float(base_model["bias"]),
        "weights": [*base_weights, *(float(value) for value in coefficients)],
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, default=Path("weights/model.json"))
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--active-dataset", type=Path, required=True)
    parser.add_argument("--ridge-penalty", type=float, default=100.0)
    parser.add_argument("--active-weight", type=float, default=1.0)
    parser.add_argument("--objective", choices=("pointwise", "pairwise"), default="pointwise")
    parser.add_argument("--output", type=Path, default=Path("weights/model.json"))
    args = parser.parse_args()
    if args.ridge_penalty < 0 or args.active_weight <= 0:
        parser.error("ridge penalty must be non-negative and active weight positive")
    base_model = json.loads(args.base_model.read_text())
    base_positions, base_labels, base_metadata = load_base_dataset(args.base_dataset)
    active_rows, active_metadata = load_active_dataset(args.active_dataset)
    coefficients, metrics = train(
        base_positions,
        base_labels,
        active_rows,
        base_model,
        args.ridge_penalty,
        args.active_weight,
        args.objective,
    )
    payload = model_payload(
        base_model, coefficients, metrics, args, active_metadata, base_metadata
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"strategic_coefficients": coefficients.tolist(), **metrics}, indent=2))


if __name__ == "__main__":
    main()
