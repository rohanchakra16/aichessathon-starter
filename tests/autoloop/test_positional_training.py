from __future__ import annotations

import chess
import numpy as np

import controller
from training.train_positional_evaluator import (
    AGENT_FEATURE_SOURCE,
    AGENT_RESIDUAL_GLUE,
    FEATURE_NAMES,
    LOWER_BOUNDS,
    UPPER_BOUNDS,
    fit_coefficients,
    positional_features,
)

DECIDED_ENDGAME = "8/8/8/3k4/8/8/1R6/3K4 w - - 40 1"  # white a rook up, hmc 40
BALANCED_ENDGAME = "8/4k3/8/4p3/4P3/8/4K3/8 w - - 40 1"  # equal material


def test_feature_vector_shape_matches_names_and_bounds() -> None:
    assert len(FEATURE_NAMES) == len(LOWER_BOUNDS) == len(UPPER_BOUNDS)
    assert positional_features(chess.Board()).shape == (len(FEATURE_NAMES),)


def test_start_and_mirrored_positions_are_symmetric() -> None:
    assert np.array_equal(positional_features(chess.Board()), np.zeros(len(FEATURE_NAMES)))
    mirrored = chess.Board("r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1")
    assert np.allclose(positional_features(mirrored), 0.0)


def test_progress_and_tropism_gate_on_decided_reduced_endgames() -> None:
    names = list(FEATURE_NAMES)
    tropism = names.index("king_tropism_decided")
    progress = names.index("progress_gradient")

    decided = positional_features(chess.Board(DECIDED_ENDGAME))
    assert decided[tropism] != 0.0
    assert decided[progress] != 0.0

    balanced = positional_features(chess.Board(BALANCED_ENDGAME))
    assert balanced[tropism] == 0.0
    assert balanced[progress] == 0.0

    opening = positional_features(chess.Board())
    assert opening[tropism] == 0.0
    assert opening[progress] == 0.0


def test_progress_gradient_penalises_the_winning_side_as_the_clock_climbs() -> None:
    progress = list(FEATURE_NAMES).index("progress_gradient")
    low = positional_features(chess.Board("8/8/8/3k4/8/8/1R6/3K4 w - - 2 1"))[progress]
    high = positional_features(chess.Board("8/8/8/3k4/8/8/1R6/3K4 w - - 80 1"))[progress]
    assert high < low <= 0.0


def test_fit_coefficients_obeys_conservative_sign_bounds() -> None:
    design = np.eye(len(FEATURE_NAMES)) * 100.0
    targets = np.asarray([10_000.0, -10_000.0] * (len(FEATURE_NAMES) // 2))
    coefficients = fit_coefficients(design, targets, 1.0)
    assert np.all(coefficients >= LOWER_BOUNDS - 1e-9)
    assert np.all(coefficients <= UPPER_BOUNDS + 1e-9)


def test_agent_feature_source_matches_the_module_runtime() -> None:
    namespace: dict[str, object] = {"chess": chess}
    exec(AGENT_FEATURE_SOURCE, namespace)
    spliced = namespace["_positional_features"]
    assert callable(spliced)
    rng = np.random.default_rng(20260904)
    for _ in range(64):
        board = chess.Board()
        for _step in range(int(rng.integers(0, 50))):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(moves[int(rng.integers(0, len(moves)))])
        assert np.allclose(np.asarray(spliced(board)), positional_features(board))


def test_splice_into_agent_is_idempotent_and_wires_the_residual() -> None:
    # Runs against whatever agent.py the checkout has: the untouched champion
    # under `make gate`, or the already-spliced source inside a retrain
    # candidate worktree during CI. Both must behave identically.
    live = (controller.ROOT / "agent.py").read_text()
    pristine = controller.strip_residual_block(live)
    spliced = controller.splice_residual_block(
        live, AGENT_FEATURE_SOURCE, AGENT_RESIDUAL_GLUE
    )
    for source in (pristine, live, spliced):
        once = controller.splice_residual_block(
            source, AGENT_FEATURE_SOURCE, AGENT_RESIDUAL_GLUE
        )
        twice = controller.splice_residual_block(
            once, AGENT_FEATURE_SOURCE, AGENT_RESIDUAL_GLUE
        )
        assert once == twice
        assert once.count(controller.RESIDUAL_BEGIN) == 1
        assert once.count(controller.RESIDUAL_END) == 1
        assert once.count("def _positional_residual(") == 1
        assert once.count("score += _positional_residual(board)") == 1
        assert once.count("return BIAS + score") == 1
    assert "_positional_residual" not in pristine
    assert controller.strip_residual_block(spliced) == pristine
