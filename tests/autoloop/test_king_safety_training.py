from __future__ import annotations

import chess
import numpy as np

from training.train_king_safety_residual import (
    LOWER_BOUNDS,
    UPPER_BOUNDS,
    fit_coefficients,
    king_safety_features,
    king_safety_values,
)


def test_start_position_king_safety_is_symmetric() -> None:
    features = king_safety_features(chess.Board())
    assert features.shape == (6,)
    assert np.array_equal(features, np.zeros(6))


def test_pawn_shield_and_open_files_are_counted_from_each_king() -> None:
    board = chess.Board("6k1/8/8/8/8/8/5PPP/6K1 w - - 0 1")
    white = king_safety_values(board, chess.WHITE)
    black = king_safety_values(board, chess.BLACK)
    assert white[0] == 3.0
    assert black[0] == 0.0
    assert white[2] == black[2] == 0.0


def test_distinct_attacker_is_not_double_counted_across_king_zone() -> None:
    board = chess.Board("6k1/8/8/8/8/6q1/8/6K1 w - - 0 1")
    _, attackers, _ = king_safety_values(board, chess.WHITE)
    assert attackers == 1.0


def test_king_safety_fit_obeys_conservative_coefficient_bounds() -> None:
    design = np.eye(6) * 100.0
    targets = np.asarray((10_000.0, -10_000.0, -10_000.0) * 2)
    coefficients = fit_coefficients(design, targets, 1.0)
    assert np.all(coefficients >= LOWER_BOUNDS)
    assert np.all(coefficients <= UPPER_BOUNDS)
