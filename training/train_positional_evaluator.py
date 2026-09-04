#!/usr/bin/env python3
"""Fit a bounded positional/endgame residual on the frozen champion PSQT.

The exp-0096 audit stopped the self-contained hand-weighted leaf-evaluation
family with ``scientific_saturation`` and recommended an offline retrain of
``weights/model.json`` with *trained* positional and endgame-conversion
features. This trainer is that family's entrypoint.

The accepted 770-weight tapered piece-square evaluator is the frozen prior. Only
a small residual coefficient vector is fit, by game-grouped cross-validation on
the training dataset alone, against Stockfish-18 teacher move rankings (the same
pairwise margin objective that produced the current model). A second, disjoint
validation dataset is reported and never used for selection.

The runtime feature source lives in ``AGENT_FEATURE_SOURCE`` below and is the
single source of truth: this module executes that exact string to build its
design matrix, and ``controller.py``'s deterministic ``claude-retrain`` path
splices the same string into ``agent.py``. ``--check-agent-consistency``
verifies the two agree before any weights are written.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
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
from training.train_pairwise_psqt_finetune import pairwise_design, rmse  # noqa: E402
from training.train_stockfish_evaluator import file_sha256  # noqa: E402

FOLD_SEED = 20260904
DEFAULT_RIDGE_PENALTIES = (10.0, 100.0, 1_000.0, 10_000.0, 100_000.0)
CONSISTENCY_SAMPLE = 96

FEATURE_NAMES: tuple[str, ...] = (
    "passed_pawn_count",
    "passed_pawn_advance",
    "doubled_pawns",
    "isolated_pawns",
    "connected_pawns",
    "king_centralisation",
    "rook_behind_passer",
    "king_pawn_proximity",
    "king_tropism_decided",
    "progress_gradient",
)
# Conservative, sign-constrained bounds. A ~10-unit feature differential times
# its coefficient stays well under a pawn, so the residual can only break ties
# and steer conversion; it can never invert a material verdict. `progress_*` and
# `king_tropism_*` are gated to decided, reduced-material endgames.
LOWER_BOUNDS = np.asarray((0.0, 0.0, -28.0, -28.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
UPPER_BOUNDS = np.asarray((36.0, 22.0, 0.0, 0.0, 22.0, 30.0, 26.0, 26.0, 40.0, 44.0))

# --------------------------------------------------------------------------- #
# Runtime feature source — the single source of truth (executed here, spliced
# into agent.py by controller.py). Pure `chess` + `math`, mypy --strict clean.
# --------------------------------------------------------------------------- #

AGENT_FEATURE_SOURCE = '''\
_RESIDUAL_PHASE_VALUES = (0, 0, 1, 1, 2, 4, 0)
_RESIDUAL_PIECE_VALUES = (0, 100, 320, 330, 500, 900, 0)
_RESIDUAL_MAX_PHASE = 24
_RESIDUAL_DECIDED_MATERIAL = 300
_RESIDUAL_ENDGAME_PHASE = 10


def _residual_passed_mask(colour: chess.Color, square: chess.Square) -> int:
    step = 8 if colour == chess.WHITE else -8
    file_index = chess.square_file(square)
    mask = 0
    frontier = square + step
    while 0 <= frontier < 64:
        for adjacent in range(max(0, file_index - 1), min(8, file_index + 2)):
            mask |= chess.BB_SQUARES[chess.square(adjacent, chess.square_rank(frontier))]
        frontier += step
    return mask


def _residual_side_terms(board: chess.Board, colour: chess.Color) -> tuple[float, ...]:
    """Passed/structure/king terms for one colour, larger = better for `colour`."""
    pawns = board.pieces(chess.PAWN, colour)
    enemy_pawns = board.pieces(chess.PAWN, not colour)
    files = [0] * 8
    for square in pawns:
        files[chess.square_file(square)] += 1

    passed_count = 0
    passed_advance = 0.0
    isolated = 0
    connected = 0
    rook_behind = 0
    rooks = board.pieces(chess.ROOK, colour)
    for square in pawns:
        file_index = chess.square_file(square)
        rank = chess.square_rank(square)
        relative_rank = rank if colour == chess.WHITE else 7 - rank
        if not (enemy_pawns & _residual_passed_mask(colour, square)):
            passed_count += 1
            passed_advance += float(max(0, relative_rank - 1))
            for rook_square in rooks & chess.BB_FILES[file_index]:
                rook_rank = chess.square_rank(rook_square)
                if (colour == chess.WHITE and rook_rank < rank) or (
                    colour == chess.BLACK and rook_rank > rank
                ):
                    rook_behind += 1
                    break
        left = files[file_index - 1] if file_index > 0 else 0
        right = files[file_index + 1] if file_index < 7 else 0
        if left == 0 and right == 0:
            isolated += 1
        support_rank = rank - 1 if colour == chess.WHITE else rank + 1
        for adjacent in (file_index - 1, file_index + 1):
            if 0 <= adjacent < 8 and 0 <= support_rank < 8 and (
                pawns & chess.BB_SQUARES[chess.square(adjacent, support_rank)]
            ):
                connected += 1
                break
    doubled = sum(max(0, count - 1) for count in files)

    king_square = board.king(colour)
    centralisation = 0.0
    king_pawn_proximity = 0.0
    if king_square is not None:
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        centralisation = 6.0 - (abs(2 * king_file - 7) + abs(2 * king_rank - 7)) / 2.0
        if pawns:
            distances = [
                max(
                    abs(king_file - chess.square_file(pawn)),
                    abs(king_rank - chess.square_rank(pawn)),
                )
                for pawn in pawns
            ]
            king_pawn_proximity = 7.0 - (sum(distances) / len(distances))
    return (
        float(passed_count),
        passed_advance,
        float(doubled),
        float(isolated),
        float(connected),
        centralisation,
        float(rook_behind),
        king_pawn_proximity,
    )


def _positional_features(board: chess.Board) -> tuple[float, ...]:
    """Side-relative positional/endgame feature vector (side to move minus opponent)."""
    side = board.turn
    phase_units = 0
    for piece_type in range(chess.PAWN, chess.KING + 1):
        phase_units += _RESIDUAL_PHASE_VALUES[piece_type] * (
            len(board.pieces(piece_type, chess.WHITE))
            + len(board.pieces(piece_type, chess.BLACK))
        )
    endgame_blend = 1.0 - min(1.0, phase_units / _RESIDUAL_MAX_PHASE)

    own = _residual_side_terms(board, side)
    opponent = _residual_side_terms(board, not side)
    diff = [a - b for a, b in zip(own, opponent, strict=True)]

    (
        passed_count,
        passed_advance,
        doubled,
        isolated,
        connected,
        centralisation,
        rook_behind,
        king_pawn_proximity,
    ) = diff

    material_diff = 0
    for piece_type in range(chess.PAWN, chess.KING):
        material_diff += _RESIDUAL_PIECE_VALUES[piece_type] * (
            len(board.pieces(piece_type, side)) - len(board.pieces(piece_type, not side))
        )
    decided = (
        abs(material_diff) >= _RESIDUAL_DECIDED_MATERIAL
        and phase_units <= _RESIDUAL_ENDGAME_PHASE
    )
    tropism = 0.0
    progress = 0.0
    if decided:
        own_king = board.king(side)
        enemy_king = board.king(not side)
        sign = 1.0 if material_diff > 0 else -1.0
        if own_king is not None and enemy_king is not None:
            chebyshev = max(
                abs(chess.square_file(own_king) - chess.square_file(enemy_king)),
                abs(chess.square_rank(own_king) - chess.square_rank(enemy_king)),
            )
            tropism = sign * -float(chebyshev)
        progress = sign * -(min(board.halfmove_clock, 80) / 80.0)

    return (
        passed_count * endgame_blend,
        passed_advance * endgame_blend,
        doubled,
        isolated,
        connected,
        centralisation * endgame_blend,
        rook_behind * endgame_blend,
        king_pawn_proximity * endgame_blend,
        tropism,
        progress,
    )
'''

# The module-level glue that only agent.py needs (not executed here).
AGENT_RESIDUAL_GLUE = '''\
_RESIDUAL_OFFSET = int(MODEL.get("layout", {}).get("positional_offset", len(WEIGHTS)))
_RESIDUAL_COEFFS: tuple[float, ...] = WEIGHTS[_RESIDUAL_OFFSET:]


def _positional_residual(board: chess.Board) -> float:
    """Learned positional/endgame residual (0.0 unless a retrained model ships)."""
    if not _RESIDUAL_COEFFS:
        return 0.0
    total = 0.0
    for coefficient, value in zip(
        _RESIDUAL_COEFFS, _positional_features(board), strict=True
    ):
        total += coefficient * value
    return total
'''

_FEATURE_NAMESPACE: dict[str, Any] = {"chess": chess}
exec(AGENT_FEATURE_SOURCE, _FEATURE_NAMESPACE)
_positional_features_runtime: Callable[[chess.Board], tuple[float, ...]] = _FEATURE_NAMESPACE[
    "_positional_features"
]


def positional_features(board: chess.Board) -> np.ndarray:
    return np.asarray(_positional_features_runtime(board), dtype=np.float64)


def fit_coefficients(
    design: np.ndarray, targets: np.ndarray, penalty: float
) -> np.ndarray:
    regularizer = np.eye(design.shape[1], dtype=np.float64) * penalty
    raw: np.ndarray = np.linalg.solve(design.T @ design + regularizer, design.T @ targets)
    clipped: np.ndarray = np.clip(raw, LOWER_BOUNDS, UPPER_BOUNDS)
    return clipped


def cross_validate(
    rows: list[dict[str, Any]],
    position_design: np.ndarray,
    labels: np.ndarray,
    baseline: np.ndarray,
    pair_matrix: np.ndarray,
    pair_targets: np.ndarray,
    teacher_margins: np.ndarray,
    baseline_margins: np.ndarray,
    pair_game_ids: np.ndarray,
    penalty: float,
    fold_count: int,
    seed: int,
) -> dict[str, Any]:
    row_game_ids = np.asarray([int(row["game_id"]) for row in rows])
    groups = pairs = baseline_top1 = candidate_top1 = 0
    baseline_reciprocal = candidate_reciprocal = 0.0
    baseline_squared = candidate_squared = 0.0
    for validation_games in game_folds(rows, fold_count, seed):
        training_pairs = ~np.isin(pair_game_ids, list(validation_games))
        validation_pairs = ~training_pairs
        validation_rows = np.isin(row_game_ids, list(validation_games))
        coefficients = fit_coefficients(
            pair_matrix[training_pairs], pair_targets[training_pairs], penalty
        )
        candidate = baseline + position_design @ coefficients
        fold_groups, fold_candidate_top1, fold_candidate_rr = ranking_totals(
            rows, labels, candidate, validation_rows
        )
        _, fold_baseline_top1, fold_baseline_rr = ranking_totals(
            rows, labels, baseline, validation_rows
        )
        candidate_margin = baseline_margins[validation_pairs] + (
            pair_matrix[validation_pairs] @ coefficients
        )
        baseline_error = teacher_margins[validation_pairs] - baseline_margins[validation_pairs]
        candidate_error = teacher_margins[validation_pairs] - candidate_margin
        groups += fold_groups
        pairs += int(validation_pairs.sum())
        baseline_top1 += fold_baseline_top1
        candidate_top1 += fold_candidate_top1
        baseline_reciprocal += fold_baseline_rr
        candidate_reciprocal += fold_candidate_rr
        baseline_squared += float(baseline_error @ baseline_error)
        candidate_squared += float(candidate_error @ candidate_error)
    return {
        "fold_count": fold_count,
        "validation_groups": groups,
        "validation_pairs": pairs,
        "baseline_margin_rmse": float(np.sqrt(baseline_squared / pairs)),
        "candidate_margin_rmse": float(np.sqrt(candidate_squared / pairs)),
        "baseline_top1": baseline_top1 / groups,
        "candidate_top1": candidate_top1 / groups,
        "baseline_mrr": baseline_reciprocal / groups,
        "candidate_mrr": candidate_reciprocal / groups,
    }


def independent_validation(
    path: Path, model: dict[str, Any], coefficients: np.ndarray, margin_clip: float
) -> dict[str, Any]:
    rows, metadata = load_active_dataset(path)
    positions = [chess.Board(row["fen"]) for row in rows]
    design = np.vstack([positional_features(board) for board in positions])
    label_clip = float(model["training"]["label_clip_centipawns"])
    labels = np.clip(
        np.asarray([float(row["label"]) for row in rows]), -label_clip, label_clip
    )
    baseline = baseline_prediction(positions, model)
    _, _targets, teacher_margins, baseline_margins, _ = pairwise_design(
        rows, design, labels, baseline, margin_clip
    )
    candidate = baseline + design @ coefficients
    all_rows: np.ndarray = np.ones(len(rows), dtype=bool)
    groups, baseline_top1, baseline_rr = ranking_totals(rows, labels, baseline, all_rows)
    _, candidate_top1, candidate_rr = ranking_totals(rows, labels, candidate, all_rows)
    pair_design, _, _, _, _ = pairwise_design(rows, design, labels, baseline, margin_clip)
    return {
        "dataset_sha256": metadata["dataset_sha256"],
        "validation_groups": groups,
        "baseline_top1": baseline_top1 / groups,
        "candidate_top1": candidate_top1 / groups,
        "baseline_mrr": baseline_rr / groups,
        "candidate_mrr": candidate_rr / groups,
        "baseline_margin_rmse": rmse(teacher_margins, baseline_margins),
        "candidate_margin_rmse": rmse(
            teacher_margins, baseline_margins + pair_design @ coefficients
        ),
    }


def check_agent_consistency(sample: int = CONSISTENCY_SAMPLE) -> None:
    """Abort unless agent._positional_features matches this module's source."""
    import agent

    if not hasattr(agent, "_positional_features"):
        raise SystemExit(
            "agent.py has no _positional_features; splice the residual block first"
        )
    rng = np.random.default_rng(FOLD_SEED)
    mismatches = 0
    for _ in range(sample):
        board = chess.Board()
        for _step in range(int(rng.integers(0, 60))):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(moves[int(rng.integers(0, len(moves)))])
        ours = _positional_features_runtime(board)
        theirs = tuple(agent._positional_features(board))
        if len(ours) != len(theirs) or any(
            abs(a - b) > 1e-9 for a, b in zip(ours, theirs, strict=True)
        ):
            mismatches += 1
    if mismatches:
        raise SystemExit(
            f"agent._positional_features disagrees with the trainer on "
            f"{mismatches}/{sample} sampled positions; refusing to write weights"
        )


def _reject_leaky_dataset(metadata: dict[str, Any], label: str) -> None:
    if metadata.get("protected_opening_list_used") is not False:
        raise SystemExit(f"{label} dataset provenance is not independent")
    if metadata.get("game_grouped") is not True:
        raise SystemExit(f"{label} dataset is not game-grouped")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=Path("weights/model.json"))
    parser.add_argument("--training-dataset", type=Path)
    parser.add_argument("--validation-dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=FOLD_SEED)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--margin-clip", type=float, default=300.0)
    parser.add_argument(
        "--ridge-penalties", type=float, nargs="+", default=list(DEFAULT_RIDGE_PENALTIES)
    )
    parser.add_argument("--check-agent-consistency", action="store_true")
    args = parser.parse_args()

    if args.check_agent_consistency and not (args.training_dataset and args.output):
        check_agent_consistency()
        print("agent._positional_features matches the trainer source", flush=True)
        return

    if not (args.training_dataset and args.validation_dataset and args.output):
        parser.error("--training-dataset, --validation-dataset and --output are required")
    if any(penalty <= 0.0 for penalty in args.ridge_penalties):
        parser.error("ridge penalties must be positive")

    check_agent_consistency()
    seed = int(args.seed)

    model = json.loads(args.base_model.read_text())
    if len(model["weights"]) != 770:
        raise SystemExit("base model must be the frozen 770-weight tapered evaluator")

    rows, train_metadata = load_active_dataset(args.training_dataset)
    _reject_leaky_dataset(train_metadata, "training")
    _, validation_metadata = load_active_dataset(args.validation_dataset)
    _reject_leaky_dataset(validation_metadata, "validation")
    train_games = {int(row["game_id"]) for row in rows}
    validation_rows, _ = load_active_dataset(args.validation_dataset)
    validation_games = {int(row["game_id"]) for row in validation_rows}
    if train_games & validation_games:
        raise SystemExit("training and validation datasets share games; leakage risk")

    positions = [chess.Board(row["fen"]) for row in rows]
    design = np.vstack([positional_features(board) for board in positions])
    label_clip = float(model["training"]["label_clip_centipawns"])
    labels = np.clip(
        np.asarray([float(row["label"]) for row in rows]), -label_clip, label_clip
    )
    baseline = baseline_prediction(positions, model)
    pair_matrix, targets, teacher_margins, baseline_margins, pair_game_ids = pairwise_design(
        rows, design, labels, baseline, args.margin_clip
    )

    penalty_results = []
    for penalty in sorted(set(args.ridge_penalties)):
        validation = cross_validate(
            rows, design, labels, baseline, pair_matrix, targets,
            teacher_margins, baseline_margins, pair_game_ids, penalty, args.folds, seed,
        )
        penalty_results.append({"ridge_penalty": penalty, "cross_validation": validation})
    selected = max(
        penalty_results,
        key=lambda item: (
            item["cross_validation"]["candidate_mrr"],
            item["cross_validation"]["candidate_top1"],
            -item["cross_validation"]["candidate_margin_rmse"],
            -item["ridge_penalty"],
        ),
    )
    penalty = float(selected["ridge_penalty"])
    coefficients = fit_coefficients(pair_matrix, targets, penalty)
    independent = independent_validation(
        args.validation_dataset, model, coefficients, args.margin_clip
    )

    layout = dict(model["layout"])
    layout.update(
        {"positional_offset": 770, "positional_feature_names": list(FEATURE_NAMES)}
    )
    script = Path(__file__).resolve()
    payload = {
        "schema_version": 7,
        "model_kind": "frozen_psqt_with_positional_endgame_residual",
        "materially_drives": "all non-terminal search leaf evaluations",
        "bias": float(model["bias"]),
        "weights": [*map(float, model["weights"]), *map(float, coefficients)],
        "layout": layout,
        "residual_coefficients": dict(
            zip(FEATURE_NAMES, map(float, coefficients), strict=True)
        ),
        "residual_bounds": {
            name: [float(low), float(high)]
            for name, low, high in zip(FEATURE_NAMES, LOWER_BOUNDS, UPPER_BOUNDS, strict=True)
        },
        "training": {
            **model["training"],
            "method": "frozen 770-weight PSQT plus bounded positional/endgame residual",
            "objective": "pairwise teacher move-ranking margin (reused from champion)",
            "selection": "game-grouped cross-validation on the training set only",
            "residual_seed": int(args.seed),
            "fold_count": int(args.folds),
            "ridge_penalty": penalty,
            "ridge_penalties_considered": sorted(set(args.ridge_penalties)),
            "margin_clip_centipawns": args.margin_clip,
            "residual_script": str(script.relative_to(ROOT)),
            "residual_script_sha256": file_sha256(script),
            "base_model_sha256": file_sha256(args.base_model),
            "training_dataset_sha256": train_metadata["dataset_sha256"],
            "validation_dataset_sha256": validation_metadata["dataset_sha256"],
            "protected_opening_list_used": False,
        },
        "penalty_cross_validation": penalty_results,
        "selected_cross_validation": selected["cross_validation"],
        "independent_validation": independent,
        "feature_consistency": {"checked_positions": CONSISTENCY_SAMPLE, "mismatches": 0},
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
                "residual_coefficients": payload["residual_coefficients"],
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
