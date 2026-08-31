import argparse

import chess
import pytest

from training.generate_opening_book import parse_schedule, position_key


def test_position_key_ignores_only_move_clocks() -> None:
    first = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    later = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 47 92")
    assert position_key(first) == position_key(later)


def test_position_key_preserves_castling_and_en_passant_state() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    without_en_passant = board.copy(stack=False)
    without_en_passant.ep_square = None
    assert position_key(board) != position_key(without_en_passant)
    without_castling = chess.Board()
    without_castling.castling_rights = chess.BB_EMPTY
    assert position_key(chess.Board()) != position_key(without_castling)


def test_branch_schedule_is_validated() -> None:
    assert parse_schedule("4,3,2") == (4, 3, 2)
    with pytest.raises(argparse.ArgumentTypeError):
        parse_schedule("4,0,2")
