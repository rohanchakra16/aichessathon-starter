#!/usr/bin/env python3
"""Fine-tune the champion PSQT on teacher-ranked move pairs.

Unlike pointwise score fitting, this objective learns only the evaluation
differences required to rank the teacher's best child ahead of its alternatives.
Complete games are held out together during cross-validation.
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
from training.train_stockfish_evaluator import features, file_sha256  # noqa: E402


def pairwise_design(
    rows: list[dict[str, Any]],
    design: np.ndarray,
    labels: np.ndarray,
    baseline: np.ndarray,
    margin_clip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    groups: dict[tuple[int, int], list[int]] = {}
    for index, row in enumerate(rows):
        if row["source"] != "parent":
            groups.setdefault((int(row["game_id"]), int(row["parent_ply"])), []).append(index)
    differences: list[np.ndarray] = []
    targets: list[float] = []
    teacher_margins: list[float] = []
    baseline_margins: list[float] = []
    game_ids: list[int] = []
    for (game_id, _), indices in groups.items():
        best = min(indices, key=lambda index: (labels[index], index))
        for alternative in indices:
            if alternative == best:
                continue
            teacher_margin = max(-margin_clip, float(labels[best] - labels[alternative]))
            baseline_margin = float(baseline[best] - baseline[alternative])
            differences.append(design[best] - design[alternative])
            targets.append(teacher_margin - baseline_margin)
            teacher_margins.append(teacher_margin)
            baseline_margins.append(baseline_margin)
            game_ids.append(game_id)
    if not differences:
        raise ValueError("active dataset produced no teacher-ranked move pairs")
    return (
        np.vstack(differences),
        np.asarray(targets),
        np.asarray(teacher_margins),
        np.asarray(baseline_margins),
        np.asarray(game_ids),
    )


def fit_delta(design: np.ndarray, targets: np.ndarray, penalty: float) -> np.ndarray:
    regularizer = np.eye(design.shape[1], dtype=np.float64) * penalty
    delta: np.ndarray = np.linalg.solve(
        design.T @ design + regularizer,
        design.T @ targets,
    )
    return delta


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def cross_validate(
    rows: list[dict[str, Any]],
    position_design: np.ndarray,
    labels: np.ndarray,
    baseline: np.ndarray,
    pair_design: np.ndarray,
    pair_targets: np.ndarray,
    teacher_margins: np.ndarray,
    baseline_margins: np.ndarray,
    pair_game_ids: np.ndarray,
    penalty: float,
    fold_count: int,
) -> dict[str, Any]:
    row_game_ids = np.asarray([int(row["game_id"]) for row in rows])
    folds = game_folds(rows, fold_count)
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
    fold_metrics: list[dict[str, Any]] = []
    for fold_index, validation_games in enumerate(folds, 1):
        train_pairs = ~np.isin(pair_game_ids, list(validation_games))
        validation_pairs = ~train_pairs
        validation_rows = np.isin(row_game_ids, list(validation_games))
        delta = fit_delta(pair_design[train_pairs], pair_targets[train_pairs], penalty)
        candidate = baseline + position_design @ delta
        groups, candidate_top, candidate_reciprocal = ranking_totals(
            rows, labels, candidate, validation_rows
        )
        _, baseline_top, baseline_reciprocal = ranking_totals(
            rows, labels, baseline, validation_rows
        )
        candidate_margins = baseline_margins[validation_pairs] + (
            pair_design[validation_pairs] @ delta
        )
        pair_count = int(validation_pairs.sum())
        baseline_error = teacher_margins[validation_pairs] - baseline_margins[validation_pairs]
        candidate_error = teacher_margins[validation_pairs] - candidate_margins
        totals["groups"] += groups
        totals["baseline_top1"] += baseline_top
        totals["candidate_top1"] += candidate_top
        totals["baseline_reciprocal"] += baseline_reciprocal
        totals["candidate_reciprocal"] += candidate_reciprocal
        totals["pairs"] += pair_count
        totals["baseline_squared_error"] += float(baseline_error @ baseline_error)
        totals["candidate_squared_error"] += float(candidate_error @ candidate_error)
        fold_metrics.append(
            {
                "fold": fold_index,
                "validation_games": sorted(validation_games),
                "validation_groups": groups,
                "validation_pairs": pair_count,
                "baseline_margin_rmse": rmse(
                    teacher_margins[validation_pairs], baseline_margins[validation_pairs]
                ),
                "candidate_margin_rmse": rmse(
                    teacher_margins[validation_pairs], candidate_margins
                ),
                "baseline_top1": baseline_top / groups,
                "candidate_top1": candidate_top / groups,
                "baseline_mrr": baseline_reciprocal / groups,
                "candidate_mrr": candidate_reciprocal / groups,
            }
        )
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
        "folds": fold_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, default=Path("weights/model.json"))
    parser.add_argument("--active-dataset", type=Path, required=True)
    parser.add_argument("--ridge-penalty", type=float, default=100_000.0)
    parser.add_argument("--margin-clip", type=float, default=1000.0)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("weights/model.json"))
    args = parser.parse_args()
    if args.ridge_penalty <= 0.0 or args.margin_clip <= 0.0:
        parser.error("ridge penalty and margin clip must be positive")
    model = json.loads(args.base_model.read_text())
    rows, metadata = load_active_dataset(args.active_dataset)
    positions = [chess.Board(row["fen"]) for row in rows]
    design = np.vstack([features(board) for board in positions])
    label_clip = float(model["training"]["label_clip_centipawns"])
    labels = np.clip(np.asarray([float(row["label"]) for row in rows]), -label_clip, label_clip)
    baseline = baseline_prediction(positions, model)
    pair_design, pair_targets, teacher_margins, baseline_margins, pair_game_ids = (
        pairwise_design(rows, design, labels, baseline, args.margin_clip)
    )
    validation = cross_validate(
        rows,
        design,
        labels,
        baseline,
        pair_design,
        pair_targets,
        teacher_margins,
        baseline_margins,
        pair_game_ids,
        args.ridge_penalty,
        args.folds,
    )
    delta = fit_delta(pair_design, pair_targets, args.ridge_penalty)
    prior = np.asarray(model["weights"], dtype=np.float64)
    coefficients = prior + delta
    final_margins = baseline_margins + pair_design @ delta
    script = Path(__file__)
    payload = {
        "schema_version": 5,
        "model_kind": "pairwise_finetuned_tapered_piece_square_evaluator",
        "materially_drives": "all non-terminal search leaf evaluations",
        "layout": model["layout"],
        "training": {
            "method": "ridge-anchored pairwise teacher move-ranking fine-tune",
            "selection": "complete-game cross-validation before full-data fit",
            "ridge_penalty": args.ridge_penalty,
            "margin_clip_centipawns": args.margin_clip,
            "label_clip_centipawns": label_clip,
            "active_examples": len(rows),
            "active_pairs": len(pair_design),
            "active_dataset_sha256": metadata["dataset_sha256"],
            "active_champion_agent_sha256": metadata["champion_agent_sha256"],
            "active_champion_model_sha256": metadata["champion_model_sha256"],
            "teacher_name": metadata["teacher_name"],
            "teacher_binary_sha256": metadata["teacher_binary_sha256"],
            "external_engine_used": True,
            "protected_opening_list_used": False,
            "base_model_sha256": file_sha256(args.base_model),
            "script": str(script.relative_to(Path.cwd())),
            "script_sha256": file_sha256(script),
        },
        "bias": float(model["bias"]),
        "weights": [float(value) for value in coefficients],
        "cross_validation": validation,
        "baseline_margin_rmse": rmse(teacher_margins, baseline_margins),
        "final_margin_rmse": rmse(teacher_margins, final_margins),
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
                    "baseline_margin_rmse",
                    "final_margin_rmse",
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
