"""Search sanity: tactical solves, nps, and a node/eval comparison vs champion."""

from __future__ import annotations

import sys
import time

from weights.p2pos import Position
from weights.p2search import Searcher

# (fen, best-move-uci alternatives, name) — mate/tactic shots, side to move wins
TACTICS = [
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4", ["f3f7"], "scholar"),
    ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", ["a1a8"], "back-rank"),
    ("2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1", ["g3g6"], "WAC.001"),
    ("8/7p/5k2/5p2/p1p2P2/Pr1pPK2/1P1R3P/8 b - - 0 1", ["b3b2"], "WAC.019"),
    ("r3qb1k/1b4p1/p2pr2p/3n4/Pnp1N1N1/6RP/1B3PP1/1B1QR1K1 w - - 0 1", ["g4h6"], "WAC.002"),
    ("r2rb1k1/pp1q1p1p/2n1p1p1/2bp4/5P2/PP1BPR1Q/1BPN2PP/R5K1 w - - 0 1", ["h3h7"], "WAC.004"),
    ("5rk1/1ppb3p/p1pb4/6q1/3P1p1r/2P1R2P/PP1BQ1P1/5RKN w - - 0 1", ["e3g3"], "WAC.008"),
    ("r1b1kb1r/3q1ppp/pBp1pn2/8/Np1P4/5N2/PPP2PPP/R2Q1RK1 w kq - 0 1", ["a4c5"], "WAC.011"),
    ("6k1/6p1/p1B1p2p/1p1qP2P/1Pb2p2/2P5/4rQ1K/R7 w - - 0 1", ["f2f4"], "WAC-fork"),
    ("r5k1/pQp2qpp/8/4pbN1/3P4/6P1/PP3P1P/R5K1 w - - 0 1", ["g5e6", "a1a8"], "WAC.014"),
]

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MID = "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP3PPP/RN1Q1RK1 w - - 0 8"
TAC = "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1"


def solve(ms: float) -> None:
    s = Searcher()
    s.new_game()
    ok = 0
    for fen, best, name in TACTICS:
        p = Position.from_fen(fen)
        t = time.perf_counter()
        mv, sc, nodes = s.search(p, time_ms=ms)
        dt = time.perf_counter() - t
        good = mv in best
        ok += good
        print(f"  {'OK ' if good else 'MISS'} {name:12s} got={mv:6s} want={'/'.join(best):10s} "
              f"score={sc:+6d} nodes={nodes:>9d} {dt:5.2f}s")
    print(f"solved {ok}/{len(TACTICS)}")


def bench(ms: float) -> None:
    s = Searcher()
    for name, fen in (("startpos", START), ("middlegame", MID), ("tactical", TAC)):
        s.new_game()
        p = Position.from_fen(fen)
        t = time.perf_counter()
        mv, sc, nodes = s.search(p, time_ms=ms)
        dt = time.perf_counter() - t
        # find reached depth
        d = 0
        s2 = Searcher()
        s2.new_game()
        for dd in range(1, 40):
            tt = time.perf_counter()
            s2.search(p, max_depth=dd, time_ms=ms)
            if time.perf_counter() - tt > ms / 1000.0 or s2.stop[0]:
                d = dd
                break
        print(f"  {name:12s} move={mv:6s} score={sc:+6d} nodes={nodes:>9d} "
              f"{dt:5.2f}s  {nodes/dt/1000:8.1f} knps  ~depth {d}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "solve"
    ms = float(sys.argv[2]) if len(sys.argv) > 2 else 2000.0
    print("warming JIT...")
    w = Searcher()
    w.new_game()
    w.search(Position.from_fen(START), max_depth=4, time_ms=5000)
    if mode in ("solve", "all"):
        solve(ms)
    if mode in ("bench", "all"):
        bench(ms)
