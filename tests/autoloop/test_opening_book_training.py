import argparse
from pathlib import Path

import chess
import pytest

from training.generate_opening_book import parse_schedule, payload, position_key


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


def test_payload_records_multiple_candidates_and_generation_recipe(tmp_path: Path) -> None:
    engine = tmp_path / "teacher"
    engine.write_bytes(b"fixed teacher")
    book = {position_key(chess.Board()): ["e2e4", "d2d4", "g1f3"]}
    result = payload(book, engine, 1000, (4, 2), 3, 4000)
    assert result["moves"] == book
    assert result["generation"]["candidate_moves_per_position"] == 3
    assert result["generation"]["protected_opening_list_used"] is False
