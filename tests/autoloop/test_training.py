import chess

from training.train_stockfish_evaluator import FEATURES, features, generated_positions, phase


def test_tapered_features_have_fixed_shape() -> None:
    vector = features(chess.Board())
    assert vector.shape == (FEATURES,)
    assert phase(chess.Board()) == 1.0


def test_generated_positions_are_deterministic_and_nonterminal() -> None:
    first = generated_positions(12)
    second = generated_positions(12)
    assert [board.fen() for board in first] == [board.fen() for board in second]
    assert all(not board.is_game_over(claim_draw=True) for board in first)


def test_side_to_move_features_reverse_sign_for_same_position() -> None:
    white = chess.Board("4k3/8/8/8/8/2N5/8/4K3 w - - 0 1")
    black = chess.Board("4k3/8/8/8/8/2N5/8/4K3 b - - 0 1")
    assert (features(white) == -features(black)).all()
