"""Phineas 2 — AI Chessathon agent entry point.

Thin, judge-readable wrapper. All engine logic lives under ``weights/`` (which
ships in the zip and is import-safe): ``weights.p2core`` is the numba-jitted
bitboard move generator and make/unmake, ``weights.p2eval`` is the tapered
piece-square evaluation seeded from the trained model ``weights/model.json``,
and ``weights.p2search`` is the iterative-deepening principal-variation search.

The competition starts one process per game, so module state here is per-game
state: a fresh transposition table and the list of root positions we have been
asked about (our own move history, for repetition detection — the platform
hands us only a FEN).
"""

from __future__ import annotations

import numpy as np

from weights.p2pos import Position
from weights.p2search import Searcher

_SEARCHER = Searcher()
_SEARCHER.new_game()
_ROOT_KEYS: list[int] = []


def _warm_up() -> None:
    """Pay the numba JIT cost inside the import budget, not on the clock."""
    start = Position.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    _SEARCHER.search(start, max_depth=6, time_ms=10_000)
    _SEARCHER.new_game()
    _ROOT_KEYS.clear()


_warm_up()


def _budget_ms(time_left_ms: int, movenum: int) -> float:
    if time_left_ms <= 200:
        return 0.0
    safety = 300.0
    usable = time_left_ms - safety
    share = usable / 25.0 + 400.0
    if movenum < 30:
        share *= 1.15
    return max(20.0, min(share, usable * 0.5))


def get_move(fen: str, time_left_ms: int) -> str:
    pos = Position.from_fen(fen)
    _ROOT_KEYS.append(int(pos.zob[0]))

    prefix = np.asarray(_ROOT_KEYS[:-1], dtype=np.uint64) if len(_ROOT_KEYS) > 1 else None
    budget = _budget_ms(time_left_ms, int(pos.meta[4]))

    if budget <= 0.0:
        move, _score, _nodes = _SEARCHER.search(pos, max_depth=1, time_ms=50.0, hist_prefix=prefix)
        return move

    move, _score, _nodes = _SEARCHER.search(pos, time_ms=budget, hist_prefix=prefix)
    return move
