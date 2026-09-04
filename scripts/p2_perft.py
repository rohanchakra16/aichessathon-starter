"""Perft + differential validation for the Phineas 2 bitboard core.

Run pure-Python first:   P2_NO_NUMBA=1 python scripts/p2_perft.py
Then jitted:             python scripts/p2_perft.py
"""

from __future__ import annotations

import sys
import time

import chess

from weights import p2core as p2c
from weights.p2pos import Position

PERFT_SUITE = [
    ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
     [1, 20, 400, 8902, 197281, 4865609]),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
     [1, 48, 2039, 97862, 4085603]),
    ("position3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
     [1, 14, 191, 2812, 43238, 674624]),
    ("position4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
     [1, 6, 264, 9467, 422333]),
    ("position5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
     [1, 44, 1486, 62379, 2103487]),
    ("position6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
     [1, 46, 2079, 89890, 3894594]),
]

EDGE_CASES = [
    ("ep-white", "8/8/8/2Pp4/8/8/8/k6K w - d6 0 1"),
    ("ep-black", "K6k/8/8/8/2pP4/8/8/8 b - d3 0 1"),
    ("promo", "8/P6k/8/8/8/8/7K/8 w - - 0 1"),
    ("underpromo-cap", "n7/1P5k/8/8/8/8/7K/8 w - - 0 1"),
    ("castle-both", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
    ("castle-through-check", "r3k2r/8/8/8/8/8/6q1/R3K2R w KQkq - 0 1"),
    ("pinned", "4k3/8/8/8/8/8/3q4/3RK3 w - - 0 1"),
    ("check-evasion", "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"),
]


def run_perft() -> bool:
    ok = True
    for name, fen, expected in PERFT_SUITE:
        for depth, want in enumerate(expected):
            p = Position.from_fen(fen)
            t0 = time.perf_counter()
            got = p.perft(depth)
            dt = time.perf_counter() - t0
            tag = "OK " if got == want else "BAD"
            if got != want:
                ok = False
            nps = got / dt if dt > 0 else 0
            print(
                f"  {tag} {name:12s} d{depth}  got={got:>10d} "
                f"want={want:>10d}  {dt:7.3f}s  {nps / 1e3:8.1f}knps"
            )
    return ok


def run_diff() -> bool:
    ok = True
    for name, fen in [(n, f) for n, f, _ in PERFT_SUITE] + EDGE_CASES:
        p = Position.from_fen(fen)
        mine = set(p.legal_moves_uci())
        ref = {m.uci() for m in chess.Board(fen).legal_moves}
        if mine != ref:
            ok = False
            print(f"  BAD {name}: missing={sorted(ref - mine)} extra={sorted(mine - ref)}")
        else:
            print(f"  OK  {name}: {len(mine)} legal moves match python-chess")
    return ok


def run_diff_random(games: int = 40, max_plies: int = 40) -> bool:
    import random

    random.seed(2026)
    ok = True
    checked = 0
    for _ in range(games):
        board = chess.Board()
        for _ in range(max_plies):
            if board.is_game_over():
                break
            fen = board.fen()
            p = Position.from_fen(fen)
            mine = set(p.legal_moves_uci())
            ref = {m.uci() for m in board.legal_moves}
            checked += 1
            if mine != ref:
                ok = False
                print(f"  BAD {fen}\n      missing={sorted(ref - mine)} extra={sorted(mine - ref)}")
            board.push(random.choice(list(board.legal_moves)))
    print(f"  checked {checked} random positions: {'all match' if ok else 'MISMATCHES'}")
    return ok


def run_makeunmake(games: int = 30, max_plies: int = 60) -> bool:
    import random

    random.seed(7)
    ok = True
    for _ in range(games):
        board = chess.Board()
        p = Position.from_fen(board.fen())
        for _ in range(max_plies):
            if board.is_game_over():
                break
            n = int(p2c.gen_moves(p.bb, p.occ, p.meta, p.buf[0]))
            moves = [int(p.buf[0, i]) for i in range(n)]
            names = ("bb", "occ", "mbox", "meta", "zob")
            live = (p.bb, p.occ, p.mbox, p.meta, p.zob)
            snap = tuple(a.copy() for a in live)
            for mv in moves:
                p2c.make(p.bb, p.occ, p.mbox, p.meta, p.zob, p.u_cap, p.u_meta, p.u_zob, 0, mv)
                p2c.unmake(p.bb, p.occ, p.mbox, p.meta, p.zob, p.u_cap, p.u_meta, p.u_zob, 0, mv)
                for arr, ref, nm in zip(live, snap, names, strict=True):
                    if not (arr == ref).all():
                        ok = False
                        print(f"  BAD make/unmake {nm} on {p.to_fen()} mv={mv:#x}")
            # advance both
            legal = [m for m in board.legal_moves]
            chosen = random.choice(legal)
            board.push(chosen)
            # re-sync p from board (simplest, avoids uci->move mapping here)
            p = Position.from_fen(board.fen())
            # zobrist consistency check
            if p.zob[0] != p._zobrist_from_scratch():
                ok = False
                print(f"  BAD zobrist scratch mismatch {board.fen()}")
    print(f"  make/unmake round-trip: {'clean' if ok else 'BROKEN'}")
    return ok


if __name__ == "__main__":
    print(f"NUMBA_OK={p2c.NUMBA_OK}")
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    results = {}
    if which in ("all", "perft"):
        print("== perft ==")
        results["perft"] = run_perft()
    if which in ("all", "diff"):
        print("== differential (fixed) ==")
        results["diff"] = run_diff()
    if which in ("all", "rand"):
        print("== differential (random) ==")
        results["rand"] = run_diff_random()
    if which in ("all", "mu"):
        print("== make/unmake ==")
        results["mu"] = run_makeunmake()
    print("SUMMARY:", results)
    sys.exit(0 if all(results.values()) else 1)
