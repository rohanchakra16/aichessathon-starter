"""Readable learned-evaluation chess agent for the AI Chessathon.

The shipped model is a compact tapered piece-square evaluator produced by the
protected offline training pipeline. Its output is the only non-terminal leaf
evaluation used by the search, so the learned model materially determines move
selection.
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

MATE = 1_000_000.0
SQUARE_FEATURES = 6 * 64
ENDGAME_OFFSET = SQUARE_FEATURES
CASTLING_OFFSET = SQUARE_FEATURES * 2
STRATEGIC_OFFSET = CASTLING_OFFSET + 2
STRATEGIC_FEATURES = 7
ACTIVE_STRATEGIC_INDICES = (0, 1, 2, 3, 6)
STRATEGIC_CACHE_LIMIT = 20_000
PHASE_VALUES = (0, 0, 1, 1, 2, 4, 0)
MAX_PHASE = 24
MAX_DEPTH = 8
QUIESCENCE_DEPTH = 3
TT_LIMIT = 50_000
TIME_CHECK_MASK = 63

_deadline = math.inf
_nodes = 0
_tt: dict[tuple[object, int], float] = {}
_strategic_cache: dict[tuple[int, ...], tuple[float, ...]] = {}


class SearchTimeout(Exception):
    """Internal control flow used to return the last completed iteration."""


def _passed_mask(colour: chess.Color, square: chess.Square) -> int:
    direction = 1 if colour == chess.WHITE else -1
    rank = chess.square_rank(square) + direction
    file = chess.square_file(square)
    mask = 0
    while 0 <= rank < 8:
        for candidate_file in range(max(0, file - 1), min(8, file + 2)):
            mask |= chess.BB_SQUARES[chess.square(candidate_file, rank)]
        rank += direction
    return mask


PASSED_MASKS = {
    colour: tuple(_passed_mask(colour, square) for square in chess.SQUARES)
    for colour in chess.COLORS
}


def _strategic_values(board: chess.Board, colour: chess.Color) -> tuple[float, ...]:
    pawns = board.pieces(chess.PAWN, colour)
    enemy_pawns = board.pieces(chess.PAWN, not colour)
    file_counts = [len(pawns & chess.BB_FILES[file]) for file in range(8)]
    occupied_files = [count > 0 for count in file_counts]
    passed_pawns = 0
    isolated_pawns = 0
    for square in pawns:
        if not enemy_pawns & PASSED_MASKS[colour][square]:
            passed_pawns += 1
        file = chess.square_file(square)
        left = file > 0 and occupied_files[file - 1]
        right = file < 7 and occupied_files[file + 1]
        if not left and not right:
            isolated_pawns += 1
    doubled_pawns = sum(max(0, count - 1) for count in file_counts)
    bishop_pair = int(len(board.pieces(chess.BISHOP, colour)) >= 2)

    king_shield = 0
    king_square = board.king(colour)
    if king_square is not None:
        king_file = chess.square_file(king_square)
        shield_rank = chess.square_rank(king_square) + (
            1 if colour == chess.WHITE else -1
        )
        if 0 <= shield_rank < 8:
            for file in range(max(0, king_file - 1), min(8, king_file + 2)):
                king_shield += int(
                    bool(pawns & chess.BB_SQUARES[chess.square(file, shield_rank)])
                )
    return tuple(
        float(value)
        for value in (
            passed_pawns,
            isolated_pawns,
            doubled_pawns,
            bishop_pair,
            king_shield,
        )
    )


def _strategic_difference(board: chess.Board) -> tuple[float, ...]:
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    key = (
        board.pawns & white,
        board.pawns & black,
        board.bishops & white,
        board.bishops & black,
        board.kings & white,
        board.kings & black,
    )
    cached = _strategic_cache.get(key)
    if cached is None:
        white_values = _strategic_values(board, chess.WHITE)
        black_values = _strategic_values(board, chess.BLACK)
        cached = tuple(
            value - opposing
            for value, opposing in zip(white_values, black_values, strict=True)
        )
        if len(_strategic_cache) >= STRATEGIC_CACHE_LIMIT:
            _strategic_cache.clear()
        _strategic_cache[key] = cached
    if board.turn == chess.WHITE:
        return cached
    return tuple(-value for value in cached)


def _model_evaluate(board: chess.Board) -> float:
    """Taper the learned piece-square model by remaining material phase."""
    side = board.turn
    midgame = 0.0
    endgame = 0.0
    phase = 0
    for colour, sign in ((side, 1.0), (not side, -1.0)):
        for piece_type in range(chess.PAWN, chess.KING + 1):
            squares = board.pieces(piece_type, colour)
            phase += PHASE_VALUES[piece_type] * len(squares)
            offset = (piece_type - 1) * 64
            for square in squares:
                relative = square if colour == chess.WHITE else chess.square_mirror(square)
                midgame += sign * WEIGHTS[offset + relative]
                endgame += sign * WEIGHTS[ENDGAME_OFFSET + offset + relative]

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
    for feature_index, value in zip(
        ACTIVE_STRATEGIC_INDICES, _strategic_difference(board), strict=True
    ):
        score += value * (
            blend * WEIGHTS[STRATEGIC_OFFSET + feature_index]
            + (1.0 - blend)
            * WEIGHTS[STRATEGIC_OFFSET + STRATEGIC_FEATURES + feature_index]
        )
    return BIAS + score


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
