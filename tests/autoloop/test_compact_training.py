from pathlib import Path

import chess
import numpy as np

from training.train_compact_evaluator import (
    ENDGAME_BOUNDS,
    FEATURES,
    MATERIAL_PRIOR,
    MIDGAME_BOUNDS,
    coefficient_prior,
    dataset_digest,
    features,
    load_dataset,
    positional_values,
    ridge_fit,
    save_dataset,
)


def test_compact_features_have_fixed_shape_and_reverse_with_turn() -> None:
    white = chess.Board("4k3/pp6/8/3P4/8/2N5/PP6/4K3 w - - 0 1")
    black = chess.Board("4k3/pp6/8/3P4/8/2N5/PP6/4K3 b - - 0 1")
    assert features(white).shape == (FEATURES,)
    assert np.array_equal(features(white), -features(black))


def test_compact_prior_keeps_conventional_material_scale() -> None:
    prior = coefficient_prior()
    assert tuple(prior[1 : 1 + len(MATERIAL_PRIOR)]) == MATERIAL_PRIOR


def test_passed_and_isolated_pawn_features_are_detected() -> None:
    board = chess.Board("4k3/8/8/3P4/8/8/7p/4K3 w - - 0 1")
    white = positional_values(board, chess.WHITE)
    black = positional_values(board, chess.BLACK)
    assert white[1] == 1.0
    assert white[2] == 1.0
    assert black[1] == 1.0
    assert black[2] == 1.0


def test_dataset_cache_round_trip_is_digest_checked(tmp_path: Path) -> None:
    engine = tmp_path / "teacher"
    engine.write_bytes(b"fixed teacher binary")
    positions = [chess.Board(), chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")]
    labels = np.asarray([17.0, 0.0])
    digest = dataset_digest(positions, labels)
    cache = tmp_path / "dataset.json"
    save_dataset(cache, positions, labels, engine, 2500, digest)
    loaded_positions, loaded_labels, loaded_digest = load_dataset(
        cache, engine, len(positions), 2500
    )
    assert [board.fen() for board in loaded_positions] == [board.fen() for board in positions]
    assert np.array_equal(loaded_labels, labels)
    assert loaded_digest == digest


def test_fit_fixes_material_and_clips_positional_coefficients() -> None:
    design = np.zeros((12, FEATURES))
    labels = np.full(12, 50_000.0)
    coefficients = ridge_fit(design, labels, 250.0)
    assert tuple(coefficients[1 : 1 + len(MATERIAL_PRIOR)]) == MATERIAL_PRIOR
    positional = coefficients[1 + len(MATERIAL_PRIOR) :]
    for coefficient, (lower, upper) in zip(
        positional, (*MIDGAME_BOUNDS, *ENDGAME_BOUNDS), strict=True
    ):
        assert lower <= coefficient <= upper
