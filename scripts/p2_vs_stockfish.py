"""Phineas 2 vs a strength-limited local Stockfish, for dev calibration only.

NOT a promotion gate, NOT run at runtime, NOT shipped. Stockfish is used here
exactly as the competition rules allow for development: as an offline sparring
partner / teacher, never bundled or called at runtime by agent.py.

Distinguishes itself from an exact-clock claim by printing the time control
used; pass --base-ms 120000 --inc-ms 500 for the real competition clock (slow
— each game can take several minutes).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import chess
import chess.engine

from weights.p2pos import Position
from weights.p2search import Searcher

STOCKFISH = "/opt/homebrew/bin/stockfish"

OPENINGS = [
    "startpos",
    "1.e4 e5 2.Nf3 Nc6 3.Bb5",
    "1.d4 Nf6 2.c4 e6 3.Nc3 Bb4",
    "1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 a6",
    "1.c4 e5 2.Nc3 Nf6",
    "1.e4 e6 2.d4 d5 3.Nc3 Bb4",
]


def _board_from(opening: str) -> chess.Board:
    b = chess.Board()
    if opening == "startpos":
        return b
    for tok in opening.replace(".", ". ").split():
        if tok.endswith("."):
            continue
        b.push_san(tok)
    return b


def play_game(engine: chess.engine.SimpleEngine, p2_white: bool,
              opening: str, base_ms: int, inc_ms: int, ply_cap: int) -> tuple[str, str]:
    board = _board_from(opening)
    searcher = Searcher()
    searcher.new_game()
    root_keys: list[int] = []
    clocks = {chess.WHITE: base_ms, chess.BLACK: base_ms}
    sf_side = chess.BLACK if p2_white else chess.WHITE
    p2_side = chess.WHITE if p2_white else chess.BLACK

    while not board.is_game_over(claim_draw=True) and board.ply() < ply_cap:
        stm = board.turn
        t0 = time.monotonic()
        if stm == p2_side:
            pos = Position.from_fen(board.fen())
            root_keys.append(int(pos.zob[0]))
            import numpy as np
            prefix = (np.asarray(root_keys[:-1], dtype=np.uint64)
                      if len(root_keys) > 1 else None)
            budget = max(20.0, min(clocks[stm] / 25.0 + 400.0, clocks[stm] * 0.5))
            uci, _sc, _n = searcher.search(pos, time_ms=budget, hist_prefix=prefix)
            move = chess.Move.from_uci(uci)
        else:
            limit = chess.engine.Limit(
                white_clock=clocks[chess.WHITE] / 1000.0,
                black_clock=clocks[chess.BLACK] / 1000.0,
                white_inc=inc_ms / 1000.0, black_inc=inc_ms / 1000.0,
            )
            result = engine.play(board, limit)
            move = result.move
        elapsed = (time.monotonic() - t0) * 1000.0
        clocks[stm] -= elapsed
        if clocks[stm] < 0:
            loser = "p2" if stm == p2_side else "sf"
            return ("sf" if loser == "p2" else "p2"), "flag"
        clocks[stm] += inc_ms
        if move is None or move not in board.legal_moves:
            loser = "p2" if stm == p2_side else "sf"
            return ("sf" if loser == "p2" else "p2"), f"illegal:{move}"
        board.push(move)

    if board.is_checkmate():
        winner_colour = chess.WHITE if board.turn == chess.BLACK else chess.BLACK
        return ("p2" if winner_colour == p2_side else "sf"), "checkmate"
    outcome = board.outcome(claim_draw=True)
    return "draw", (outcome.termination.name.lower() if outcome else "adjudicated")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--elo", type=int, default=1600)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--base-ms", type=int, default=15000)
    ap.add_argument("--inc-ms", type=int, default=150)
    ap.add_argument("--ply-cap", type=int, default=300)
    args = ap.parse_args()

    if not Path(STOCKFISH).exists():
        raise SystemExit(f"stockfish not found at {STOCKFISH}")

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": args.elo, "Threads": 1})

    print(f"Phineas2 vs Stockfish 18 (UCI_Elo={args.elo}, LimitStrength)  "
          f"TC={args.base_ms}+{args.inc_ms}ms  ({'COMPETITION CLOCK' if args.base_ms >= 120000 else 'fast dev clock, not a competition-clock claim'})")

    w = d = losses = 0
    terms: dict[str, int] = {}
    try:
        for i in range(args.pairs):
            opening = OPENINGS[i % len(OPENINGS)]
            for p2_white in (True, False):
                res, term = play_game(engine, p2_white, opening, args.base_ms, args.inc_ms, args.ply_cap)
                terms[term] = terms.get(term, 0) + 1
                if res == "draw":
                    d += 1
                    s = "="
                elif res == "p2":
                    w += 1
                    s = "W"
                else:
                    losses += 1
                    s = "L"
                print(f"  {opening[:22]:22s} p2={'W' if p2_white else 'B'}  {s}  ({term})")
    finally:
        engine.quit()

    n = w + d + losses
    score = (w + 0.5 * d) / n if n else 0.0
    print(f"\nP2 vs Stockfish(elo={args.elo}): +{w} ={d} -{losses}   score {score:.3f}   ({n} games)")
    print("terminations:", terms)


if __name__ == "__main__":
    main()
