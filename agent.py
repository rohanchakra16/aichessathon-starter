"""Readable learned-evaluation chess agent for the AI Chessathon.

The shipped model is a small linear evaluator trained by
``training/train_linear_evaluator.py``. Its output is the only non-terminal
leaf evaluation used by the search, so the learned model materially determines
move selection.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import chess

MODEL_PATH = Path(__file__).with_name("weights") / "model.json"
MODEL = json.loads(MODEL_PATH.read_text())
WEIGHTS: tuple[float, ...] = tuple(float(value) for value in MODEL["weights"])
BIAS = float(MODEL["bias"])
BOOK_PATH = Path(__file__).with_name("weights") / "opening-candidates.json"
BOOK_MOVES: dict[str, list[str]] = json.loads(BOOK_PATH.read_text())["moves"]

MATE = 1_000_000.0
MAX_DEPTH = 5
QUIESCENCE_DEPTH = 2
TT_LIMIT = 50_000
TIME_CHECK_MASK = 63

_deadline = math.inf
_nodes = 0
_tt: dict[tuple[object, int], float] = {}


class SearchTimeout(Exception):
    """Internal control flow used to return the last completed iteration."""


def _position_key(board: chess.Board) -> str:
    """Book identity without move clocks, preserving legal move state."""
    return " ".join(board.fen(en_passant="fen").split()[:4])


def _features(board: chess.Board) -> tuple[float, ...]:
    """Model features from the point of view of the side to move."""
    side = board.turn
    features: list[float] = []
    for piece_type in range(chess.PAWN, chess.KING + 1):
        features.append(
            float(len(board.pieces(piece_type, side)) - len(board.pieces(piece_type, not side)))
        )
    for piece_type in range(chess.PAWN, chess.KING + 1):
        activity = 0.0
        for colour, sign in ((side, 1.0), (not side, -1.0)):
            for square in board.pieces(piece_type, colour):
                file_distance = abs(chess.square_file(square) - 3.5)
                rank_distance = abs(chess.square_rank(square) - 3.5)
                activity += sign * (3.5 - (file_distance + rank_distance) / 2.0)
        features.append(activity)
    return tuple(features)


def _model_evaluate(board: chess.Board) -> float:
    """Learned leaf evaluation; no handcrafted leaf score is mixed in."""
    pairs = zip(WEIGHTS, _features(board), strict=True)
    return BIAS + sum(weight * value for weight, value in pairs)


def _ordered_moves(
    board: chess.Board,
    principal: chess.Move | None = None,
    candidates: list[chess.Move] | None = None,
) -> list[chess.Move]:
    def priority(move: chess.Move) -> tuple[int, int, int, str]:
        victim = board.piece_type_at(move.to_square) or 0
        attacker = board.piece_type_at(move.from_square) or 0
        return (
            1 if move == principal else 0,
            1 if move.promotion else 0,
            victim * 10 - attacker,
            move.uci(),
        )

    moves = board.legal_moves if candidates is None else candidates
    return sorted(moves, key=priority, reverse=True)


def _check_time() -> None:
    global _nodes
    _nodes += 1
    if _nodes & TIME_CHECK_MASK == 0 and time.monotonic() >= _deadline:
        raise SearchTimeout


def _quiescence(board: chess.Board, alpha: float, beta: float, depth: int) -> float:
    """Resolve short capture sequences before applying the learned evaluator."""
    _check_time()
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        if outcome.winner is None:
            return 0.0
        return MATE if outcome.winner == board.turn else -MATE
    if depth == 0:
        return _model_evaluate(board)

    in_check = board.is_check()
    if not in_check:
        stand_pat = _model_evaluate(board)
        if stand_pat >= beta:
            return stand_pat
        alpha = max(alpha, stand_pat)

    moves = _ordered_moves(board)
    if not in_check:
        moves = [move for move in moves if board.is_capture(move)]
    if not moves:
        return _model_evaluate(board)

    best = -math.inf if in_check else alpha
    for move in moves:
        board.push(move)
        score = -_quiescence(board, -beta, -alpha, depth - 1)
        board.pop()
        best = max(best, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return best


def _negamax(board: chess.Board, depth: int, alpha: float, beta: float) -> float:
    _check_time()
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        if outcome.winner is None:
            return 0.0
        return MATE if outcome.winner == board.turn else -MATE
    if depth == 0:
        return _quiescence(board, alpha, beta, QUIESCENCE_DEPTH)

    key = (board._transposition_key(), depth)
    cached = _tt.get(key)
    if cached is not None:
        return cached
    best = -math.inf
    for move in _ordered_moves(board):
        board.push(move)
        score = -_negamax(board, depth - 1, -beta, -alpha)
        board.pop()
        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    if len(_tt) >= TT_LIMIT:
        _tt.clear()
    _tt[key] = best
    return best


def _root_search(
    board: chess.Board,
    depth: int,
    previous: chess.Move,
    candidates: list[chess.Move] | None,
) -> chess.Move:
    best_move = previous
    best_score = -math.inf
    alpha = -math.inf
    for move in _ordered_moves(board, previous, candidates):
        _check_time()
        board.push(move)
        score = -_negamax(board, depth - 1, -math.inf, -alpha)
        board.pop()
        if score > best_score:
            best_score = score
            best_move = move
        alpha = max(alpha, score)
    return best_move


def _budget_seconds(time_left_ms: int) -> float:
    if time_left_ms <= 5:
        return 0.0
    clock = time_left_ms / 1000.0
    return min(0.35, max(0.002, clock / 80.0), max(0.0, clock - 0.003))


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal UCI move while retaining a conservative flag margin."""
    global _deadline, _nodes
    board = chess.Board(fen)
    moves = list(board.legal_moves)
    if not moves:
        raise ValueError("get_move called for a terminal position")
    allowed = set(moves)
    book_moves = [
        chess.Move.from_uci(uci)
        for uci in BOOK_MOVES.get(_position_key(board), [])
        if chess.Move.from_uci(uci) in allowed
    ]
    root_moves = book_moves or moves
    best = root_moves[0]
    budget = _budget_seconds(time_left_ms)
    if budget == 0.0 or len(moves) == 1:
        return best.uci()

    _deadline = time.monotonic() + budget
    _nodes = 0
    for depth in range(1, MAX_DEPTH + 1):
        try:
            completed = _root_search(board, depth, best, root_moves)
        except SearchTimeout:
            break
        best = completed
        if time.monotonic() >= _deadline:
            break
    return best.uci()
