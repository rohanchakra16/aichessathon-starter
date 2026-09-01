"""Readable learned-evaluation chess agent for the AI Chessathon.

The shipped model combines the proven tapered piece-square evaluator with a
small king-aware, two-perspective neural accumulator.  The accumulator scores
both players' views with shared weights and subtracts them, preserving the
colour symmetry required by negamax search.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import chess
import numpy as np

MODEL_PATH = Path(__file__).with_name("weights") / "model.json"
_MODEL = json.loads(MODEL_PATH.read_text())
_RAW_WEIGHTS = _MODEL["weights"]
BASE_WEIGHTS = 770
WEIGHTS: tuple[float, ...] = tuple(float(value) for value in _RAW_WEIGHTS[:BASE_WEIGHTS])
BIAS = float(_MODEL["bias"])
NN_HIDDEN = int(_MODEL["layout"]["hidden"])
NN_FEATURES = (
    int(_MODEL["layout"]["king_buckets"])
    * int(_MODEL["layout"]["piece_planes"])
    * int(_MODEL["layout"]["squares"])
)
NN_END = BASE_WEIGHTS + NN_FEATURES * NN_HIDDEN
NN_EMBEDDING = np.asarray(_RAW_WEIGHTS[BASE_WEIGHTS:NN_END], dtype=np.float32).reshape(
    NN_FEATURES, NN_HIDDEN
)
NN_OUTPUT = np.asarray(_RAW_WEIGHTS[NN_END:], dtype=np.float32)
NN_SCALE = float(_MODEL["layout"]["output_scale_centipawns"])
if len(NN_OUTPUT) != NN_HIDDEN:
    raise ValueError("invalid neural model layout")
del _MODEL, _RAW_WEIGHTS

MATE = 1_000_000.0
SQUARE_FEATURES = 6 * 64
ENDGAME_OFFSET = SQUARE_FEATURES
CASTLING_OFFSET = SQUARE_FEATURES * 2
PHASE_VALUES = (0, 0, 1, 1, 2, 4, 0)
MAX_PHASE = 24
MAX_DEPTH = 8
QUIESCENCE_DEPTH = 3
TT_LIMIT = 50_000
TIME_CHECK_MASK = 63

_deadline = math.inf
_nodes = 0
_tt: dict[tuple[object, int], float] = {}


class SearchTimeout(Exception):
    """Internal control flow used to return the last completed iteration."""


def _model_evaluate(board: chess.Board) -> float:
    """Combine tapered piece-square and king-aware neural evaluations."""
    side = board.turn
    side_king = board.king(side)
    opponent_king = board.king(not side)
    if side_king is None or opponent_king is None:
        return 0.0
    side_relative_king = (
        side_king if side == chess.WHITE else chess.square_mirror(side_king)
    )
    opponent_relative_king = (
        opponent_king if side == chess.BLACK else chess.square_mirror(opponent_king)
    )
    side_mirror_files = chess.square_file(side_relative_king) >= 4
    opponent_mirror_files = chess.square_file(opponent_relative_king) >= 4
    if side_mirror_files:
        side_relative_king ^= 7
    if opponent_mirror_files:
        opponent_relative_king ^= 7
    side_bucket = chess.square_rank(side_relative_king) * 4 + chess.square_file(
        side_relative_king
    )
    opponent_bucket = chess.square_rank(opponent_relative_king) * 4 + chess.square_file(
        opponent_relative_king
    )

    midgame = 0.0
    endgame = 0.0
    phase = 0
    side_indices: list[int] = []
    opponent_indices: list[int] = []
    for colour, sign in ((side, 1.0), (not side, -1.0)):
        for piece_type in range(chess.PAWN, chess.KING + 1):
            squares = board.pieces(piece_type, colour)
            phase += PHASE_VALUES[piece_type] * len(squares)
            offset = (piece_type - 1) * 64
            for square in squares:
                relative = square if colour == chess.WHITE else chess.square_mirror(square)
                midgame += sign * WEIGHTS[offset + relative]
                endgame += sign * WEIGHTS[ENDGAME_OFFSET + offset + relative]

                side_square = square if side == chess.WHITE else chess.square_mirror(square)
                if side_mirror_files:
                    side_square ^= 7
                side_relation = 0 if colour == side else 1
                side_plane = side_relation * 6 + piece_type - 1
                side_indices.append((side_bucket * 12 + side_plane) * 64 + side_square)

                opponent_square = (
                    square if side == chess.BLACK else chess.square_mirror(square)
                )
                if opponent_mirror_files:
                    opponent_square ^= 7
                opponent_relation = 0 if colour != side else 1
                opponent_plane = opponent_relation * 6 + piece_type - 1
                opponent_indices.append(
                    (opponent_bucket * 12 + opponent_plane) * 64 + opponent_square
                )

    blend = min(1.0, phase / MAX_PHASE)
    score = blend * midgame + (1.0 - blend) * endgame
    if board.has_kingside_castling_rights(side):
        score += WEIGHTS[CASTLING_OFFSET]
    if board.has_kingside_castling_rights(not side):
        score -= WEIGHTS[CASTLING_OFFSET]
    if board.has_queenside_castling_rights(side):
        score += WEIGHTS[CASTLING_OFFSET + 1]
    if board.has_queenside_castling_rights(not side):
        score -= WEIGHTS[CASTLING_OFFSET + 1]
    side_accumulator = NN_EMBEDDING[side_indices].sum(axis=0)
    opponent_accumulator = NN_EMBEDDING[opponent_indices].sum(axis=0)
    neural = (
        np.clip(side_accumulator, 0.0, 1.0)
        - np.clip(opponent_accumulator, 0.0, 1.0)
    ) @ NN_OUTPUT
    return BIAS + score + float(neural) * NN_SCALE


def _ordered_moves(board: chess.Board, principal: chess.Move | None = None) -> list[chess.Move]:
    def priority(move: chess.Move) -> tuple[int, int, int, str]:
        victim = board.piece_type_at(move.to_square) or 0
        attacker = board.piece_type_at(move.from_square) or 0
        return (
            1 if move == principal else 0,
            1 if move.promotion else 0,
            victim * 10 - attacker,
            move.uci(),
        )

    return sorted(board.legal_moves, key=priority, reverse=True)


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


def _root_search(board: chess.Board, depth: int, previous: chess.Move) -> chess.Move:
    best_move = previous
    best_score = -math.inf
    alpha = -math.inf
    for move in _ordered_moves(board, previous):
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
    if clock >= 10.0:
        return min(2.0, max(0.68, clock / 60.0), max(0.0, clock - 0.003))
    return min(0.68, max(0.002, clock / 25.0), max(0.0, clock - 0.003))


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal UCI move while retaining a conservative flag margin."""
    global _deadline, _nodes
    board = chess.Board(fen)
    moves = list(board.legal_moves)
    if not moves:
        raise ValueError("get_move called for a terminal position")
    best = moves[0]
    budget = _budget_seconds(time_left_ms)
    if budget == 0.0 or len(moves) == 1:
        return best.uci()

    _deadline = time.monotonic() + budget
    _nodes = 0
    for depth in range(1, MAX_DEPTH + 1):
        try:
            completed = _root_search(board, depth, best)
        except SearchTimeout:
            break
        best = completed
        if time.monotonic() >= _deadline:
            break
    return best.uci()
