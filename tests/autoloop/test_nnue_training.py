import chess
import numpy as np

from training.train_nnue_evaluator import (
    PADDING_INDEX,
    SPARSE_FEATURES,
    fit,
    neural_residual,
    perspective_indices,
    sparse_batch,
)


def test_sparse_encoding_is_deterministic_and_perspective_specific() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    white = perspective_indices(board, chess.WHITE)
    black = perspective_indices(board, chess.BLACK)
    assert white == perspective_indices(board, chess.WHITE)
    assert len(white) == len(black) == 32
    assert all(0 <= index < SPARSE_FEATURES for index in white + black)
    assert white != black


def test_two_perspective_network_is_exactly_antisymmetric() -> None:
    positions = [chess.Board(), chess.Board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")]
    own, opponent, mask = sparse_batch(positions)
    rng = np.random.default_rng(7)
    embedding = rng.normal(size=(SPARSE_FEATURES + 1, 3)).astype(np.float32)
    embedding[PADDING_INDEX] = 0.0
    output = rng.normal(size=3).astype(np.float32)
    forward = neural_residual(own, opponent, mask, embedding, output, 400.0)
    reverse = neural_residual(opponent, own, mask, embedding, output, 400.0)
    assert np.allclose(forward, -reverse, atol=1e-5)


def test_small_fit_is_reproducible_and_reduces_residual_error() -> None:
    positions = [
        chess.Board(),
        chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"),
        chess.Board("4k3/4p3/8/8/8/8/8/4K3 w - - 0 1"),
        chess.Board("4k3/8/8/8/8/8/3P4/4K3 b - - 0 1"),
    ]
    own, opponent, mask = sparse_batch(positions)
    targets = np.asarray([0.0, 120.0, -120.0, -80.0])
    sample_weights = np.ones(4, dtype=np.float32)
    first_embedding, first_output, losses = fit(
        own,
        opponent,
        mask,
        targets,
        sample_weights,
        hidden=2,
        epochs=20,
        batch_size=4,
        learning_rate=0.01,
        output_scale=400.0,
        l2_penalty=0.0,
    )
    second_embedding, second_output, second_losses = fit(
        own,
        opponent,
        mask,
        targets,
        sample_weights,
        hidden=2,
        epochs=20,
        batch_size=4,
        learning_rate=0.01,
        output_scale=400.0,
        l2_penalty=0.0,
    )
    assert losses[-1] < losses[0]
    assert losses == second_losses
    assert np.array_equal(first_embedding, second_embedding)
    assert np.array_equal(first_output, second_output)
