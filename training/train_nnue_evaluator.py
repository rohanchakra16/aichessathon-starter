#!/usr/bin/env python3
"""Train a compact king-aware neural residual evaluator with NumPy.

The submitted evaluator never invokes the external teacher.  This offline
trainer consumes independently generated, engine-labelled caches and retains
the proven tapered piece-square champion as a fixed skip connection.  Complete
active-learning games are held out for validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.generate_active_learning_dataset import (  # noqa: E402
    dataset_digest as active_dataset_digest,
)
from training.train_active_residual_evaluator import split_game_ids  # noqa: E402
from training.train_stockfish_evaluator import (  # noqa: E402
    features as piece_square_features,
)

SEED = 2026090103
BASE_WEIGHTS = 770
KING_BUCKETS = 32
PIECE_PLANES = 12
SQUARES = 64
SPARSE_FEATURES = KING_BUCKETS * PIECE_PLANES * SQUARES
PADDING_INDEX = SPARSE_FEATURES
MAX_PIECES = 32


@dataclass(frozen=True)
class Dataset:
    positions: list[chess.Board]
    labels: np.ndarray
    game_ids: np.ndarray
    sources: list[str]
    parent_plies: np.ndarray


def _relative_square(square: int, perspective: chess.Color) -> int:
    return square if perspective == chess.WHITE else chess.square_mirror(square)


def perspective_indices(board: chess.Board, perspective: chess.Color) -> list[int]:
    """Encode pieces relative to one king, with horizontal mirror canonicalisation."""
    king = board.king(perspective)
    if king is None:
        raise ValueError("NNUE positions must contain both kings")
    relative_king = _relative_square(king, perspective)
    mirror_files = chess.square_file(relative_king) >= 4
    if mirror_files:
        relative_king ^= 7
    king_bucket = chess.square_rank(relative_king) * 4 + chess.square_file(relative_king)

    indices: list[int] = []
    for square, piece in board.piece_map().items():
        relative = _relative_square(square, perspective)
        if mirror_files:
            relative ^= 7
        relation = 0 if piece.color == perspective else 1
        plane = relation * 6 + piece.piece_type - 1
        indices.append((king_bucket * PIECE_PLANES + plane) * SQUARES + relative)
    return sorted(indices)


def sparse_batch(positions: list[chess.Board]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    own = np.full((len(positions), MAX_PIECES), PADDING_INDEX, dtype=np.int32)
    opponent = np.full((len(positions), MAX_PIECES), PADDING_INDEX, dtype=np.int32)
    mask = np.zeros((len(positions), MAX_PIECES), dtype=np.float32)
    for row, board in enumerate(positions):
        own_indices = perspective_indices(board, board.turn)
        opponent_indices = perspective_indices(board, not board.turn)
        if len(own_indices) != len(opponent_indices) or len(own_indices) > MAX_PIECES:
            raise ValueError("invalid piece count in NNUE position")
        count = len(own_indices)
        own[row, :count] = own_indices
        opponent[row, :count] = opponent_indices
        mask[row, :count] = 1.0
    return own, opponent, mask


def baseline_prediction(positions: list[chess.Board], model: dict[str, Any]) -> np.ndarray:
    weights = np.asarray(model["weights"], dtype=np.float64)
    if len(weights) != BASE_WEIGHTS:
        raise ValueError("base model must be the protected 770-weight champion")
    design = np.vstack([piece_square_features(board) for board in positions])
    return float(model["bias"]) + design @ weights


def load_base_dataset(path: Path) -> Dataset:
    payload = json.loads(path.read_text())
    if payload.get("kind") != "engine_guided_selfplay_evaluation_dataset":
        raise ValueError("base dataset has the wrong kind")
    if payload.get("protected_opening_list_used") is not False:
        raise ValueError("base dataset provenance is not independent")
    rows = payload.get("rows", [])
    if not rows:
        raise ValueError("base dataset is empty")
    return Dataset(
        positions=[chess.Board(row["fen"]) for row in rows],
        labels=np.asarray([float(row["label"]) for row in rows], dtype=np.float64),
        game_ids=np.full(len(rows), -1, dtype=np.int32),
        sources=["base"] * len(rows),
        parent_plies=np.full(len(rows), -1, dtype=np.int32),
    )


def load_active_dataset(path: Path) -> tuple[Dataset, dict[str, Any]]:
    payload = json.loads(path.read_text())
    expected = {
        "schema_version": 1,
        "kind": "champion_disagreement_active_learning_dataset",
        "protected_opening_list_used": False,
        "game_grouped": True,
    }
    actual = {key: payload.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"active dataset metadata mismatch: {actual!r} != {expected!r}")
    rows = payload.get("rows", [])
    if len(rows) != int(payload.get("rows_count", -1)) or not rows:
        raise ValueError("active dataset row count mismatch")
    if active_dataset_digest(rows) != payload.get("dataset_sha256"):
        raise ValueError("active dataset digest mismatch")
    return (
        Dataset(
            positions=[chess.Board(row["fen"]) for row in rows],
            labels=np.asarray([float(row["label"]) for row in rows], dtype=np.float64),
            game_ids=np.asarray([int(row["game_id"]) for row in rows], dtype=np.int32),
            sources=[str(row["source"]) for row in rows],
            parent_plies=np.asarray([int(row["parent_ply"]) for row in rows], dtype=np.int32),
        ),
        payload,
    )


def _activation(accumulator: np.ndarray) -> np.ndarray:
    return np.clip(accumulator, 0.0, 1.0)


def neural_residual(
    own: np.ndarray,
    opponent: np.ndarray,
    mask: np.ndarray,
    embedding: np.ndarray,
    output: np.ndarray,
    output_scale: float,
) -> np.ndarray:
    own_sum = (embedding[own] * mask[..., None]).sum(axis=1)
    opponent_sum = (embedding[opponent] * mask[..., None]).sum(axis=1)
    difference = _activation(own_sum) - _activation(opponent_sum)
    return difference @ output * output_scale


def root_mean_square_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def ranking_metrics(
    dataset: Dataset,
    prediction: np.ndarray,
    validation_games: set[int],
) -> dict[str, float | int]:
    groups: dict[tuple[int, int], list[int]] = {}
    for index, (game_id, parent_ply, source) in enumerate(
        zip(dataset.game_ids, dataset.parent_plies, dataset.sources, strict=True)
    ):
        if int(game_id) in validation_games and source != "parent":
            groups.setdefault((int(game_id), int(parent_ply)), []).append(index)
    top_one = 0
    reciprocal_rank = 0.0
    for indices in groups.values():
        truth = min(indices, key=lambda index: (dataset.labels[index], index))
        ordered = sorted(indices, key=lambda index: (prediction[index], index))
        top_one += ordered[0] == truth
        reciprocal_rank += 1.0 / (ordered.index(truth) + 1)
    count = len(groups)
    if count == 0:
        raise ValueError("active validation split has no move-ranking groups")
    return {
        "groups": count,
        "top_one_accuracy": top_one / count,
        "mean_reciprocal_rank": reciprocal_rank / count,
    }


def _adam_step(
    value: np.ndarray,
    gradient: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    step: int,
    learning_rate: float,
) -> None:
    first *= 0.9
    first += 0.1 * gradient
    second *= 0.999
    second += 0.001 * np.square(gradient)
    corrected_first = first / (1.0 - 0.9**step)
    corrected_second = second / (1.0 - 0.999**step)
    value -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)


def fit(
    own: np.ndarray,
    opponent: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    sample_weights: np.ndarray,
    hidden: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    output_scale: float,
    l2_penalty: float,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    rng = np.random.default_rng(SEED)
    embedding = rng.normal(0.0, 0.025, size=(SPARSE_FEATURES + 1, hidden)).astype(
        np.float32
    )
    output = rng.normal(0.0, 0.025, size=hidden).astype(np.float32)
    embedding[PADDING_INDEX] = 0.0
    first_embedding = np.zeros_like(embedding)
    second_embedding = np.zeros_like(embedding)
    first_output = np.zeros_like(output)
    second_output = np.zeros_like(output)
    normalized_targets = (targets / output_scale).astype(np.float32)
    losses: list[float] = []
    step = 0
    for _epoch in range(epochs):
        permutation = rng.permutation(len(targets))
        for start in range(0, len(targets), batch_size):
            selection = permutation[start : start + batch_size]
            batch_own = own[selection]
            batch_opponent = opponent[selection]
            batch_mask = mask[selection]
            weights = sample_weights[selection].astype(np.float32)
            weights /= weights.sum()

            own_sum = (embedding[batch_own] * batch_mask[..., None]).sum(axis=1)
            opponent_sum = (embedding[batch_opponent] * batch_mask[..., None]).sum(axis=1)
            own_activation = _activation(own_sum)
            opponent_activation = _activation(opponent_sum)
            difference = own_activation - opponent_activation
            prediction = difference @ output
            error = prediction - normalized_targets[selection]
            gradient_prediction = np.clip(error, -1.0, 1.0) * weights

            gradient_output = difference.T @ gradient_prediction + l2_penalty * output
            gradient_difference = gradient_prediction[:, None] * output[None, :]
            gradient_own = gradient_difference * ((own_sum > 0.0) & (own_sum < 1.0))
            gradient_opponent = -gradient_difference * (
                (opponent_sum > 0.0) & (opponent_sum < 1.0)
            )
            gradient_embedding = np.zeros_like(embedding)
            for column in range(MAX_PIECES):
                valid = batch_mask[:, column] != 0.0
                np.add.at(
                    gradient_embedding,
                    batch_own[valid, column],
                    gradient_own[valid],
                )
                np.add.at(
                    gradient_embedding,
                    batch_opponent[valid, column],
                    gradient_opponent[valid],
                )
            gradient_embedding += l2_penalty * embedding
            gradient_embedding[PADDING_INDEX] = 0.0
            step += 1
            _adam_step(
                embedding,
                gradient_embedding,
                first_embedding,
                second_embedding,
                step,
                learning_rate,
            )
            _adam_step(
                output,
                gradient_output,
                first_output,
                second_output,
                step,
                learning_rate,
            )
            embedding[PADDING_INDEX] = 0.0
        epoch_prediction = neural_residual(
            own, opponent, mask, embedding, output, output_scale
        )
        losses.append(root_mean_square_error(targets, epoch_prediction))
    return embedding, output, losses


def train(
    base: Dataset,
    active: Dataset,
    base_model: dict[str, Any],
    hidden: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    output_scale: float,
    l2_penalty: float,
    active_weight: float,
    label_clip: float,
    residual_blend: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    training_games, validation_games = split_game_ids(
        [
            {"game_id": int(game_id)}
            for game_id in active.game_ids
        ]
    )
    active_training = np.asarray(
        [int(game_id) in training_games for game_id in active.game_ids], dtype=bool
    )
    training_positions = base.positions + [
        board for board, selected in zip(active.positions, active_training, strict=True) if selected
    ]
    training_labels = np.concatenate((base.labels, active.labels[active_training]))
    training_labels = np.clip(training_labels, -label_clip, label_clip)
    training_baseline = baseline_prediction(training_positions, base_model)
    training_targets = training_labels - training_baseline
    sample_weights = np.concatenate(
        (
            np.ones(len(base.positions), dtype=np.float32),
            np.full(int(active_training.sum()), active_weight, dtype=np.float32),
        )
    )
    own, opponent, mask = sparse_batch(training_positions)
    embedding, output, losses = fit(
        own,
        opponent,
        mask,
        training_targets,
        sample_weights,
        hidden,
        epochs,
        batch_size,
        learning_rate,
        output_scale,
        l2_penalty,
    )

    active_own, active_opponent, active_mask = sparse_batch(active.positions)
    active_baseline = baseline_prediction(active.positions, base_model)
    active_residual = neural_residual(
        active_own, active_opponent, active_mask, embedding, output, output_scale
    )
    active_prediction = active_baseline + residual_blend * active_residual
    validation = ~active_training
    baseline_ranking = ranking_metrics(active, active_baseline, validation_games)
    neural_ranking = ranking_metrics(active, active_prediction, validation_games)
    metrics: dict[str, Any] = {
        "base_examples": len(base.positions),
        "active_training_examples": int(active_training.sum()),
        "active_validation_examples": int(validation.sum()),
        "active_training_game_ids": sorted(training_games),
        "active_validation_game_ids": sorted(validation_games),
        "final_training_residual_rmse": losses[-1],
        "initial_training_residual_rmse": losses[0],
        "active_validation_baseline_rmse": root_mean_square_error(
            np.clip(active.labels[validation], -label_clip, label_clip),
            active_baseline[validation],
        ),
        "active_validation_nnue_rmse": root_mean_square_error(
            np.clip(active.labels[validation], -label_clip, label_clip),
            active_prediction[validation],
        ),
        "active_validation_baseline_ranking": baseline_ranking,
        "active_validation_nnue_ranking": neural_ranking,
        "training_loss_by_epoch": losses,
    }
    return embedding[:-1], output, metrics


def model_payload(
    base_model: dict[str, Any],
    embedding: np.ndarray,
    output: np.ndarray,
    metrics: dict[str, Any],
    args: argparse.Namespace,
    base_metadata: dict[str, Any],
    active_metadata: dict[str, Any],
) -> dict[str, Any]:
    script = Path(__file__)
    base_weights = [float(value) for value in base_model["weights"]]
    neural_weights = embedding.reshape(-1).tolist() + output.tolist()
    return {
        "schema_version": 7,
        "model_kind": "king_aware_nnue_residual_evaluator",
        "materially_drives": "all non-terminal search leaf evaluations",
        "layout": {
            "base_weights": [0, BASE_WEIGHTS],
            "embedding": [BASE_WEIGHTS, BASE_WEIGHTS + SPARSE_FEATURES * args.hidden],
            "output": [
                BASE_WEIGHTS + SPARSE_FEATURES * args.hidden,
                BASE_WEIGHTS + SPARSE_FEATURES * args.hidden + args.hidden,
            ],
            "king_buckets": KING_BUCKETS,
            "piece_planes": PIECE_PLANES,
            "squares": SQUARES,
            "hidden": args.hidden,
            "activation": "clipped_relu_0_1",
            "two_perspective_difference": True,
            "output_scale_centipawns": args.output_scale * args.residual_blend,
        },
        "training": {
            "method": "fixed champion skip connection plus compact king-aware neural residual",
            "seed": SEED,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "l2_penalty": args.l2_penalty,
            "active_weight": args.active_weight,
            "residual_blend": args.residual_blend,
            "label_clip_centipawns": args.label_clip,
            "script": str(script.relative_to(Path.cwd())),
            "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            "external_engine_used": True,
            "teacher_name": "Stockfish 18",
            "base_dataset_sha256": base_metadata["dataset_sha256"],
            "active_dataset_sha256": active_metadata["dataset_sha256"],
            "protected_opening_list_used": False,
        },
        "bias": float(base_model["bias"]),
        "weights": base_weights + neural_weights,
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--active-dataset", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=Path("weights/model.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.004)
    parser.add_argument("--output-scale", type=float, default=400.0)
    parser.add_argument("--l2-penalty", type=float, default=1e-6)
    parser.add_argument("--active-weight", type=float, default=4.0)
    parser.add_argument("--residual-blend", type=float, default=1.0)
    parser.add_argument("--label-clip", type=float, default=1500.0)
    args = parser.parse_args()
    if args.hidden < 1 or args.epochs < 1 or args.batch_size < 1:
        parser.error("hidden, epochs, and batch size must be positive")
    if args.learning_rate <= 0 or args.output_scale <= 0 or args.active_weight <= 0:
        parser.error("learning rate, output scale, and active weight must be positive")
    if not 0.0 < args.residual_blend <= 1.0:
        parser.error("residual blend must be in (0, 1]")

    base_payload = json.loads(args.base_dataset.read_text())
    base = load_base_dataset(args.base_dataset)
    active, verified_active_payload = load_active_dataset(args.active_dataset)
    base_model = json.loads(args.base_model.read_text())
    embedding, output, metrics = train(
        base,
        active,
        base_model,
        args.hidden,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.output_scale,
        args.l2_penalty,
        args.active_weight,
        args.label_clip,
        args.residual_blend,
    )
    payload = model_payload(
        base_model,
        embedding,
        output,
        metrics,
        args,
        base_payload,
        verified_active_payload,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
