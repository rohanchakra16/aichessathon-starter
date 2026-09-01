#!/usr/bin/env python3
"""Conservatively fine-tune the champion PSQT on fresh champion trajectories.

The current 770-weight evaluator is the prior.  Complete games, rather than
individual positions, are held out together during cross-validation.  The
final fit uses every active game only after the regularisation and weighting
have been selected from the cross-validated experiment.
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

from training.train_active_residual_evaluator import (  # noqa: E402
    baseline_prediction,
    load_active_dataset,
    load_base_dataset,
)
from training.train_stockfish_evaluator import (  # noqa: E402
    features,
    file_sha256,
    root_mean_square_error,
)

FOLD_SEED = 2026090103


def design_matrix(positions: list[chess.Board]) -> np.ndarray:
    matrix: np.ndarray = np.column_stack(
        (np.ones(len(positions)), np.vstack([features(board) for board in positions]))
    )
    return matrix


def game_folds(
    rows: list[dict[str, Any]], fold_count: int, seed: int = FOLD_SEED
) -> list[set[int]]:
    game_ids = np.asarray(sorted({int(row["game_id"]) for row in rows}))
    if fold_count < 2 or len(game_ids) < fold_count:
        raise ValueError("cross-validation requires at least one game per fold")
    rng = np.random.default_rng(seed)
    rng.shuffle(game_ids)
    return [{int(game_id) for game_id in fold} for fold in np.array_split(game_ids, fold_count)]


def fit_delta(
    base_design: np.ndarray,
    base_residual: np.ndarray,
    active_design: np.ndarray,
    active_residual: np.ndarray,
    penalty: float,
    active_weight: float,
) -> np.ndarray:
    scale = np.sqrt(active_weight)
    combined_design = np.vstack((base_design, active_design * scale))
    combined_residual = np.concatenate((base_residual, active_residual * scale))
    regularizer = np.eye(combined_design.shape[1], dtype=np.float64) * penalty
    delta: np.ndarray = np.linalg.solve(
        combined_design.T @ combined_design + regularizer,
        combined_design.T @ combined_residual,
    )
    return delta


def ranking_totals(
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    prediction: np.ndarray,
    selected: np.ndarray,
) -> tuple[int, int, float]:
    groups: dict[tuple[int, int], list[int]] = {}
    for index, row in enumerate(rows):
        if selected[index] and row["source"] != "parent":
            groups.setdefault((int(row["game_id"]), int(row["parent_ply"])), []).append(index)
    top_one = 0
    reciprocal_rank = 0.0
    for indices in groups.values():
        truth = min(indices, key=lambda index: (labels[index], index))
        ordered = sorted(indices, key=lambda index: (prediction[index], index))
        rank = ordered.index(truth) + 1
        top_one += int(rank == 1)
        reciprocal_rank += 1.0 / rank
    return len(groups), top_one, reciprocal_rank


def cross_validate(
    base_design: np.ndarray,
    base_residual: np.ndarray,
    active_rows: list[dict[str, Any]],
    active_design: np.ndarray,
    active_labels: np.ndarray,
    active_baseline: np.ndarray,
    penalty: float,
    active_weight: float,
    fold_count: int,
) -> dict[str, Any]:
    game_ids = np.asarray([int(row["game_id"]) for row in active_rows])
    folds = game_folds(active_rows, fold_count)
    candidate_squared_error = 0.0
    baseline_squared_error = 0.0
    validation_examples = 0
    groups = candidate_top_one = baseline_top_one = 0
    candidate_reciprocal = baseline_reciprocal = 0.0
    fold_metrics: list[dict[str, Any]] = []
    active_residual = active_labels - active_baseline
    for fold_index, validation_games in enumerate(folds, 1):
        validation = np.isin(game_ids, list(validation_games))
        training = ~validation
        delta = fit_delta(
            base_design,
            base_residual,
            active_design[training],
            active_residual[training],
            penalty,
            active_weight,
        )
        candidate = active_baseline + active_design @ delta
        count, candidate_top, candidate_mrr = ranking_totals(
            active_rows, active_labels, candidate, validation
        )
        _, baseline_top, baseline_mrr = ranking_totals(
            active_rows, active_labels, active_baseline, validation
        )
        candidate_error = active_labels[validation] - candidate[validation]
        baseline_error = active_labels[validation] - active_baseline[validation]
        candidate_squared_error += float(candidate_error @ candidate_error)
        baseline_squared_error += float(baseline_error @ baseline_error)
        validation_examples += int(validation.sum())
        groups += count
        candidate_top_one += candidate_top
        baseline_top_one += baseline_top
        candidate_reciprocal += candidate_mrr
        baseline_reciprocal += baseline_mrr
        fold_metrics.append(
            {
                "fold": fold_index,
                "validation_games": sorted(validation_games),
                "validation_examples": int(validation.sum()),
                "validation_groups": count,
                "candidate_rmse": root_mean_square_error(
                    active_labels[validation], candidate[validation]
                ),
                "baseline_rmse": root_mean_square_error(
                    active_labels[validation], active_baseline[validation]
                ),
                "candidate_top1": candidate_top / count,
                "baseline_top1": baseline_top / count,
                "candidate_mrr": candidate_mrr / count,
                "baseline_mrr": baseline_mrr / count,
            }
        )
    return {
        "fold_count": fold_count,
        "split_seed": FOLD_SEED,
        "validation_examples": validation_examples,
        "validation_groups": groups,
        "candidate_rmse": float(np.sqrt(candidate_squared_error / validation_examples)),
        "baseline_rmse": float(np.sqrt(baseline_squared_error / validation_examples)),
        "candidate_top1": candidate_top_one / groups,
        "baseline_top1": baseline_top_one / groups,
        "candidate_mrr": candidate_reciprocal / groups,
        "baseline_mrr": baseline_reciprocal / groups,
        "folds": fold_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, default=Path("weights/model.json"))
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--active-dataset", type=Path, required=True)
    parser.add_argument("--ridge-penalty", type=float, default=10_000.0)
    parser.add_argument("--active-weight", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("weights/model.json"))
    args = parser.parse_args()
    if args.ridge_penalty <= 0 or args.active_weight <= 0:
        parser.error("ridge penalty and active weight must be positive")

    base_model = json.loads(args.base_model.read_text())
    base_positions, base_labels, base_metadata = load_base_dataset(args.base_dataset)
    active_rows, active_metadata = load_active_dataset(args.active_dataset)
    active_positions = [chess.Board(row["fen"]) for row in active_rows]
    clip = float(base_model["training"]["label_clip_centipawns"])
    base_labels = np.clip(base_labels, -clip, clip)
    active_labels = np.clip(np.asarray([float(row["label"]) for row in active_rows]), -clip, clip)
    base_design = design_matrix(base_positions)
    active_design = design_matrix(active_positions)
    prior = np.concatenate(
        ([float(base_model["bias"])], np.asarray(base_model["weights"], dtype=np.float64))
    )
    if len(prior) != base_design.shape[1]:
        raise ValueError("base model is not the expected 770-weight PSQT evaluator")
    base_baseline = baseline_prediction(base_positions, base_model)
    active_baseline = baseline_prediction(active_positions, base_model)
    base_residual = base_labels - base_baseline
    active_residual = active_labels - active_baseline
    cross_validation = cross_validate(
        base_design,
        base_residual,
        active_rows,
        active_design,
        active_labels,
        active_baseline,
        args.ridge_penalty,
        args.active_weight,
        args.folds,
    )
    delta = fit_delta(
        base_design,
        base_residual,
        active_design,
        active_residual,
        args.ridge_penalty,
        args.active_weight,
    )
    coefficients = prior + delta
    final_base_prediction = base_design @ coefficients
    final_active_prediction = active_design @ coefficients
    script = Path(__file__)
    payload = {
        "schema_version": 4,
        "model_kind": "active_finetuned_selfplay_tapered_piece_square_evaluator",
        "materially_drives": "all non-terminal search leaf evaluations",
        "layout": base_model["layout"],
        "training": {
            "method": "ridge-anchored full PSQT fine-tune on champion trajectories",
            "selection": "hyperparameters accepted only after complete-game cross-validation",
            "ridge_penalty": args.ridge_penalty,
            "active_example_weight": args.active_weight,
            "label_clip_centipawns": clip,
            "base_examples": len(base_positions),
            "active_examples": len(active_rows),
            "base_model_sha256": file_sha256(args.base_model),
            "base_dataset_sha256": base_metadata["dataset_sha256"],
            "active_dataset_sha256": active_metadata["dataset_sha256"],
            "active_champion_agent_sha256": active_metadata["champion_agent_sha256"],
            "active_champion_model_sha256": active_metadata["champion_model_sha256"],
            "teacher_name": active_metadata["teacher_name"],
            "teacher_binary_sha256": active_metadata["teacher_binary_sha256"],
            "external_engine_used": True,
            "protected_opening_list_used": False,
            "script": str(script.relative_to(Path.cwd())),
            "script_sha256": file_sha256(script),
        },
        "bias": float(coefficients[0]),
        "weights": [float(value) for value in coefficients[1:]],
        "cross_validation": cross_validation,
        "base_baseline_rmse": root_mean_square_error(base_labels, base_baseline),
        "base_final_rmse": root_mean_square_error(base_labels, final_base_prediction),
        "active_baseline_rmse": root_mean_square_error(active_labels, active_baseline),
        "active_final_rmse": root_mean_square_error(active_labels, final_active_prediction),
        "coefficient_delta_l2": float(np.linalg.norm(delta)),
        "coefficient_delta_max": float(np.max(np.abs(delta))),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "cross_validation",
                    "base_baseline_rmse",
                    "base_final_rmse",
                    "active_baseline_rmse",
                    "active_final_rmse",
                    "coefficient_delta_l2",
                    "coefficient_delta_max",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
