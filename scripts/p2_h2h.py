"""Head-to-head: Phineas 2 (this worktree's agent.py) vs the champion.

Dev harness only — NOT a promotion gate. Fast time control by default; pass
--base-ms/--inc-ms to lengthen. Champion reference is loaded from
/tmp/claude-501/p2champ/agent_champ.py (staged from main:agent.py).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import chess

WT = Path(__file__).resolve().parent.parent
CHAMP = Path("/tmp/claude-501/p2champ/agent_champ.py")

OPENINGS = [
    "startpos",
    "1.e4 e5 2.Nf3 Nc6 3.Bb5",  # ruy
    "1.d4 Nf6 2.c4 e6 3.Nc3 Bb4",  # nimzo
    "1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 a6",  # najdorf
    "1.c4 e5 2.Nc3 Nf6",  # english
    "1.e4 e6 2.d4 d5 3.Nc3 Bb4",  # french winawer
    "1.d4 d5 2.c4 c6 3.Nf3 Nf6 4.Nc3 dxc4",  # slav
    "1.e4 c6 2.d4 d5 3.e5",  # caro advance
]


def _load(path: Path, name: str):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _board_from(opening: str) -> chess.Board:
    b = chess.Board()
    if opening == "startpos":
        return b
    for tok in opening.replace(".", ". ").split():
        if tok.endswith("."):
            continue
        b.push_san(tok)
    return b


def play(white_fn, black_fn, board: chess.Board, base_ms: int, inc_ms: int,
         ply_cap: int = 300) -> str:
    clocks = {chess.WHITE: base_ms, chess.BLACK: base_ms}
    while not board.is_game_over(claim_draw=True) and board.ply() < ply_cap:
        stm = board.turn
        fn = white_fn if stm == chess.WHITE else black_fn
        t0 = time.monotonic()
        try:
            uci = fn(board.fen(), int(clocks[stm]))
        except Exception as exc:
            return "black" if stm == chess.WHITE else "white", f"crash:{exc!r}"
        elapsed = (time.monotonic() - t0) * 1000.0
        clocks[stm] -= elapsed
        if clocks[stm] < 0:
            return ("black" if stm == chess.WHITE else "white"), "flag"
        clocks[stm] += inc_ms
        try:
            move = chess.Move.from_uci(uci)
        except Exception:
            return ("black" if stm == chess.WHITE else "white"), f"malformed:{uci}"
        if move not in board.legal_moves:
            return ("black" if stm == chess.WHITE else "white"), f"illegal:{uci}"
        board.push(move)

    if board.is_checkmate():
        return ("white" if board.turn == chess.BLACK else "black"), "checkmate"
    return "draw", (board.outcome(claim_draw=True).termination.name.lower()
                    if board.outcome(claim_draw=True) else "adjudicated")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--base-ms", type=int, default=8000)
    ap.add_argument("--inc-ms", type=int, default=80)
    args = ap.parse_args()

    p2 = _load(WT / "agent.py", "agent_p2")
    champ = _load(CHAMP, "agent_champ")

    w = d = ll = 0
    terms: dict[str, int] = {}
    for i in range(args.pairs):
        opening = OPENINGS[i % len(OPENINGS)]
        for p2_white in (True, False):
            board = _board_from(opening)
            wf, bf = (p2.get_move, champ.get_move) if p2_white else (champ.get_move, p2.get_move)
            res, term = play(wf, bf, board, args.base_ms, args.inc_ms)
            terms[term] = terms.get(term, 0) + 1
            if res == "draw":
                d += 1
                s = "="
            elif (res == "white") == p2_white:
                w += 1
                s = "W"
            else:
                ll += 1
                s = "L"
            print(f"  {opening[:22]:22s} p2={'W' if p2_white else 'B'}  {s}  ({term})")
    n = w + d + ll
    score = (w + 0.5 * d) / n if n else 0.0
    print(f"\nP2 vs champion: +{w} ={d} -{ll}   score {score:.3f}   ({n} games)")
    print("terminations:", terms)


if __name__ == "__main__":
    main()
