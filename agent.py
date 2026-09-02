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
PHASE_VALUES = (0, 0, 1, 1, 2, 4, 0)
MAX_PHASE = 24
MAX_DEPTH = 8
QUIESCENCE_DEPTH = 3
TT_LIMIT = 50_000
TIME_CHECK_MASK = 63
TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2
LMR_MIN_DEPTH = 3
LMR_QUIET_INDEX = 4

_deadline = math.inf
_nodes = 0
_tt: dict[tuple[object, int, int], tuple[float, int]] = {}
_history: dict[tuple[bool, int, int], int] = {}


class SearchTimeout(Exception):
    """Internal control flow used to return the last completed iteration."""


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
    return BIAS + score


def _ordered_moves(board: chess.Board, principal: chess.Move | None = None) -> list[chess.Move]:
    turn = board.turn
    history = _history

    def priority(move: chess.Move) -> tuple[int, int, int, int, str]:
        victim = board.piece_type_at(move.to_square) or 0
        attacker = board.piece_type_at(move.from_square) or 0
        return (
            1 if move == principal else 0,
            1 if move.promotion else 0,
            victim * 10 - attacker,
            history.get((turn, move.from_square, move.to_square), 0),
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

    alpha_original = alpha
    beta_original = beta
    key = (board._transposition_key(), board.halfmove_clock, depth)
    cached = _tt.get(key)
    if cached is not None:
        cached_score, flag = cached
        if flag == TT_EXACT:
            return cached_score
        if flag == TT_LOWER:
            alpha = max(alpha, cached_score)
        else:
            beta = min(beta, cached_score)
        if alpha >= beta:
            return cached_score
    best = -math.inf
    in_check = board.is_check()
    for move_index, move in enumerate(_ordered_moves(board)):
        is_quiet = move.promotion is None and not board.is_capture(move)
        reduce_quiet = (
            depth >= LMR_MIN_DEPTH
            and move_index >= LMR_QUIET_INDEX
            and math.isfinite(alpha)
            and not in_check
            and is_quiet
            and not board.gives_check(move)
        )
        board.push(move)
        if move_index == 0:
            score = -_negamax(board, depth - 1, -beta, -alpha)
        else:
            probe_depth = depth - 2 if reduce_quiet else depth - 1
            score = -_negamax(board, probe_depth, -alpha - 1.0, -alpha)
            if reduce_quiet and score > alpha:
                score = -_negamax(board, depth - 1, -alpha - 1.0, -alpha)
            if alpha < score < beta:
                score = -_negamax(board, depth - 1, -beta, -alpha)
        board.pop()
        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            if is_quiet:
                slot = (board.turn, move.from_square, move.to_square)
                _history[slot] = _history.get(slot, 0) + depth * depth
            break
    if len(_tt) >= TT_LIMIT:
        _tt.clear()
    flag = TT_EXACT
    if best <= alpha_original:
        flag = TT_UPPER
    elif best >= beta_original:
        flag = TT_LOWER
    _tt[key] = (best, flag)
    return best


def _root_search(board: chess.Board, depth: int, previous: chess.Move) -> chess.Move:
    best_move = previous
    best_score = -math.inf
    alpha = -math.inf
    for move_index, move in enumerate(_ordered_moves(board, previous)):
        _check_time()
        board.push(move)
        if move_index == 0:
            score = -_negamax(board, depth - 1, -math.inf, math.inf)
        else:
            score = -_negamax(board, depth - 1, -alpha - 1.0, -alpha)
            if score > alpha:
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
    _history.clear()
    for depth in range(1, MAX_DEPTH + 1):
        try:
            completed = _root_search(board, depth, best)
        except SearchTimeout:
            break
        best = completed
        if time.monotonic() >= _deadline:
            break
    return best.uci()
