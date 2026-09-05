"""Syzygy endgame tablebase probing (3- and 4-piece WDL+DTZ, weights/syzygy/).

Step 8 candidate: the exact-clock ladder testing repeatedly found the same
failure mode -- a position the search itself judged clearly winning
(sometimes by several pawns, once even after promoting a new queen) getting
shuffled into a threefold-repetition or fifty-move draw because the search
horizon couldn't find real technique in a simplified endgame. A tablebase
removes that failure mode entirely for any endgame it covers: it gives the
exact game-theoretic value and a move that provably makes progress, not a
heuristic guess.

Full 3-4-5-piece Syzygy WDL+DTZ is roughly 980MB -- far past the 50MB zip
cap. 3-4-piece WDL+DTZ is ~4.3MB (verified: 35 endgame classes each way,
downloaded from https://tablebase.lichess.ovh/tables/standard/3-4-5-{wdl,dtz}/
and sanity-checked against known positions before shipping), so that is what
ships. It does not cover 5-piece endings; those still rely on search + eval.

Only used at the root, only when <=4 total pieces remain -- at most once per
move, only in already-simplified endgames, so using python-chess here (never
in the search hot path) costs nothing that matters.

Move selection: maximise our own WDL category first (this alone guarantees a
won position is never turned into a draw or loss, and a draw never into a
loss, regardless of the DTZ tie-break below), then break ties by DTZ
magnitude -- smallest when winning (force real progress fastest), largest
when losing (make the opponent prove it) -- to avoid shuffling in a position
whose category already favours us.
"""

from __future__ import annotations

import os

import chess
import chess.syzygy

_TB_DIR = os.path.join(os.path.dirname(__file__), "syzygy")
_MAX_PIECES = 4

_tablebase: chess.syzygy.Tablebase | bool | None = None


def _get_tablebase() -> chess.syzygy.Tablebase | None:
    global _tablebase
    if _tablebase is None:
        try:
            _tablebase = chess.syzygy.open_tablebase(_TB_DIR)
        except Exception:
            _tablebase = False  # sentinel: unavailable, never retry
    return _tablebase or None


def probe_best_move(fen: str) -> str | None:
    """UCI move if this is a covered, cleanly-probeable tablebase position;
    None otherwise (caller falls back to the normal search)."""
    board = chess.Board(fen)
    if len(board.piece_map()) > _MAX_PIECES:
        return None
    tb = _get_tablebase()
    if tb is None:
        return None

    best_move: chess.Move | None = None
    best_wdl = -3
    best_dtz_mag = 0
    try:
        for move in board.legal_moves:
            board.push(move)
            try:
                if board.is_checkmate():
                    wdl, dtz_mag = 2, 0
                elif board.is_stalemate() or board.is_insufficient_material():
                    wdl, dtz_mag = 0, 0
                else:
                    wdl = -tb.probe_wdl(board)
                    dtz_mag = abs(tb.probe_dtz(board))
            except (KeyError, ValueError):
                board.pop()
                continue
            board.pop()

            better = (
                best_move is None
                or wdl > best_wdl
                or (wdl == best_wdl and (dtz_mag < best_dtz_mag if wdl >= 0
                                          else dtz_mag > best_dtz_mag))
            )
            if better:
                best_move, best_wdl, best_dtz_mag = move, wdl, dtz_mag
    except Exception:
        return None

    return best_move.uci() if best_move is not None else None
