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

# Diagnostics from the most recent get_move call (score/nodes/depth). Not part
# of the competition contract -- get_move still returns only the UCI string --
# but harmless to keep for dev harnesses that want per-move search telemetry.
LAST_INFO: dict[str, int] = {}


def _warm_up() -> None:
    """Pay the numba JIT cost inside the import budget, not on the clock."""
    start = Position.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    _SEARCHER.search(start, max_depth=6, time_ms=10_000)
    _SEARCHER.new_game()
    _ROOT_KEYS.clear()


_warm_up()


_OVERHEAD_MS = 15.0  # measured fixed per-call cost outside the timed search
                     # (FEN parse, array setup, watchdog thread) -- subtracted
                     # from the requested search budget, never added to it.


def _budget_ms(time_left_ms: int, movenum: int) -> float:
    """Fraction of the remaining bank, with a hard reserve that scales with
    the bank itself so a fast/short time control (proportionally riskier)
    keeps a bigger relative margin than a long one. Never leans on the
    increment for safety -- it is a bonus, not part of the plan."""
    hard_reserve = max(250.0, time_left_ms * 0.03)
    if time_left_ms <= hard_reserve:
        return 0.0
    usable = time_left_ms - hard_reserve
    share = usable / 30.0
    if movenum < 25:
        share *= 1.15
    share = min(share, usable * 0.4)
    return max(15.0, share - _OVERHEAD_MS)


def get_move(fen: str, time_left_ms: int) -> str:
    pos = Position.from_fen(fen)
    _ROOT_KEYS.append(int(pos.zob[0]))

    prefix = np.asarray(_ROOT_KEYS[:-1], dtype=np.uint64) if len(_ROOT_KEYS) > 1 else None
    budget = _budget_ms(time_left_ms, int(pos.meta[4]))

    if budget <= 0.0:
        emergency = max(0.0, time_left_ms - 100.0)
        move, score, nodes, depth = _SEARCHER.search(
            pos, max_depth=1, time_ms=emergency, hist_prefix=prefix
        )
        LAST_INFO.update(score=score, nodes=nodes, depth=depth, budget_ms=int(emergency))
        return move

    move, score, nodes, depth = _SEARCHER.search(pos, time_ms=budget, hist_prefix=prefix)
    LAST_INFO.update(score=score, nodes=nodes, depth=depth, budget_ms=int(budget))
    return move
