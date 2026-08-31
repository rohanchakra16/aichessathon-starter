"""Readable learned-evaluation chess agent for the AI Chessathon.

The shipped model is a compact tapered evaluator produced by the protected
offline training pipeline. Its output is the only non-terminal leaf evaluation
used by the search, so the learned model materially determines move selection.
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
PHASE_VALUES = (0, 0, 1, 1, 2, 4, 0)
MAX_PHASE = 24
CENTER = chess.BB_D4 | chess.BB_E4 | chess.BB_D5 | chess.BB_E5
MAX_DEPTH = 5
QUIESCENCE_DEPTH = 2
TT_LIMIT = 50_000
TIME_CHECK_MASK = 63

_deadline = math.inf
_nodes = 0
_tt: dict[tuple[object, int], float] = {}


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


def _phase(board: chess.Board) -> float:
    current = sum(
        PHASE_VALUES[piece_type] * len(board.pieces(piece_type, colour))
        for piece_type in range(chess.PAWN, chess.KING + 1)
        for colour in chess.COLORS
    )
    return min(1.0, current / MAX_PHASE)


def _positional_values(board: chess.Board, colour: chess.Color) -> tuple[float, ...]:
    pawns = board.pieces(chess.PAWN, colour)
    enemy_pawns = board.pieces(chess.PAWN, not colour)
    file_counts = [len(pawns & chess.BB_FILES[file]) for file in range(8)]
    occupied_files = [count > 0 for count in file_counts]
    pawn_advancement = 0
    passed_pawns = 0
    isolated_pawns = 0
    for square in pawns:
        rank = chess.square_rank(square)
        relative_rank = rank if colour == chess.WHITE else 7 - rank
        pawn_advancement += max(0, relative_rank - 1)
        if not enemy_pawns & PASSED_MASKS[colour][square]:
            passed_pawns += 1
        file = chess.square_file(square)
        adjacent = occupied_files[max(0, file - 1) : file] + occupied_files[file + 1 : file + 2]
        if not any(adjacent):
            isolated_pawns += 1
    doubled_pawns = sum(max(0, count - 1) for count in file_counts)
    pawn_islands = sum(
        occupied and (file == 0 or not occupied_files[file - 1])
        for file, occupied in enumerate(occupied_files)
    )

    own_occupied = board.occupied_co[colour]
    mobility: list[int] = []
    attacks = 0
    for piece_type in range(chess.PAWN, chess.KING + 1):
        piece_mobility = 0
        for square in board.pieces(piece_type, colour):
            destinations = int(board.attacks(square)) & ~own_occupied
            attacks |= destinations
            if piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
                piece_mobility += destinations.bit_count()
        if piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            mobility.append(piece_mobility)

    open_files = 0
    semi_open_files = 0
    all_pawns = board.pawns
    for square in board.pieces(chess.ROOK, colour):
        file_mask = chess.BB_FILES[chess.square_file(square)]
        if not all_pawns & file_mask:
            open_files += 1
        elif not pawns & file_mask:
            semi_open_files += 1

    king_square = board.king(colour)
    king_shield = 0
    king_centrality = 0.0
    if king_square is not None:
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        shield_rank = king_rank + (1 if colour == chess.WHITE else -1)
        if 0 <= shield_rank < 8:
            for file in range(max(0, king_file - 1), min(8, king_file + 2)):
                shield_square = chess.square(file, shield_rank)
                king_shield += int(bool(pawns & chess.BB_SQUARES[shield_square]))
        king_centrality = 3.5 - (
            abs(king_file - 3.5) + abs(king_rank - 3.5)
        ) / 2.0

    return (
        float(pawn_advancement),
        float(passed_pawns),
        float(isolated_pawns),
        float(doubled_pawns),
        float(pawn_islands),
        float(len(board.pieces(chess.BISHOP, colour)) >= 2),
        *(float(value) for value in mobility),
        float(open_files),
        float(semi_open_files),
        float(king_shield),
        float(board.has_kingside_castling_rights(colour))
        + float(board.has_queenside_castling_rights(colour)),
        float(len(pawns & CENTER)),
        float((attacks & CENTER).bit_count()),
        king_centrality,
    )


def _features(board: chess.Board) -> tuple[float, ...]:
    """Compact tapered features from the point of view of the side to move."""
    side = board.turn
    material = tuple(
        float(
            len(board.pieces(piece_type, side))
            - len(board.pieces(piece_type, not side))
        )
        for piece_type in range(chess.PAWN, chess.QUEEN + 1)
    )
    positional = tuple(
        own - enemy
        for own, enemy in zip(
            _positional_values(board, side),
            _positional_values(board, not side),
            strict=True,
        )
    )
    phase = _phase(board)
    return (
        *material,
        *(value * phase for value in positional),
        *(value * (1.0 - phase) for value in positional),
    )


def _model_evaluate(board: chess.Board) -> float:
    """Learned leaf evaluation; no handcrafted leaf score is mixed in."""
    pairs = zip(WEIGHTS, _features(board), strict=True)
    return BIAS + sum(weight * value for weight, value in pairs)


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
    return min(0.35, max(0.002, clock / 80.0), max(0.0, clock - 0.003))


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
