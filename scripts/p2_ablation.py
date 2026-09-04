"""Ablation: candidate agent.py/weights/ vs a frozen baseline checkout.

Both sides are Phineas 2 (never a third-party engine), each in its own fresh
worker subprocess pointed at a different target directory -- typically the
live worktree (candidate) vs an isolated `git worktree add --detach <tag>`
checkout (baseline). This is the cheap, fast comparison the user's protocol
calls an "ablation": not a Stockfish match, just candidate vs baseline, so a
strength delta shows up in far fewer games than a Stockfish-anchored one
would need. Follow a promising ablation with a real Stockfish exact-clock
batch before trusting the result; this script alone does not.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import chess
import chess.pgn
from p2_exact_clock_validation import OPENINGS, Worker, _board_from, _score_ci95


def play_game(cand_white: bool, opening: str, base_ms: int, inc_ms: int,
              ply_cap: int, cand_dir: Path, base_dir: Path) -> dict:
    board = _board_from(opening)
    w_worker = Worker(cand_dir if cand_white else base_dir)
    b_worker = Worker(base_dir if cand_white else cand_dir)
    clocks = {chess.WHITE: float(base_ms), chess.BLACK: float(base_ms)}
    moves: list[dict] = []
    game = chess.pgn.Game()
    game.headers["White"] = "candidate" if cand_white else "baseline"
    game.headers["Black"] = "baseline" if cand_white else "candidate"
    node = game
    for opening_move in board.move_stack:
        node = node.add_variation(opening_move)
    result, termination = "draw", "unknown"

    try:
        while not board.is_game_over(claim_draw=True) and board.ply() < ply_cap:
            stm = board.turn
            worker = w_worker if stm == chess.WHITE else b_worker
            side = "cand" if (stm == chess.WHITE) == cand_white else "base"
            t0 = time.monotonic()
            uci, info = worker.get_move(board.fen(), int(clocks[stm]))
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            if uci is None:
                termination = f"crash:{info.get('error')}"
                result = "black" if stm == chess.WHITE else "white"
                break
            move = chess.Move.from_uci(uci)
            moves.append({"ply": board.ply(), "side": side, "uci": uci, **info})

            clocks[stm] -= elapsed_ms
            if clocks[stm] < 0:
                termination = "flag"
                result = "black" if stm == chess.WHITE else "white"
                break
            clocks[stm] += inc_ms

            if move not in board.legal_moves:
                termination = f"illegal:{uci}"
                result = "black" if stm == chess.WHITE else "white"
                break
            node = node.add_variation(move)
            board.push(move)
        else:
            if board.is_checkmate():
                result = "white" if board.turn == chess.BLACK else "black"
                termination = "checkmate"
            elif board.ply() >= ply_cap:
                result, termination = "draw", "ply_cap_adjudicated"
            else:
                outcome = board.outcome(claim_draw=True)
                result = "draw"
                termination = outcome.termination.name.lower() if outcome else "adjudicated"
    finally:
        w_worker.close()
        b_worker.close()

    game.headers["Result"] = {"white": "1-0", "black": "0-1", "draw": "1/2-1/2"}[result]
    cand_won = (result == "white") == cand_white
    cand_outcome = "cand" if cand_won else ("draw" if result == "draw" else "base")
    return {
        "opening": opening, "cand_white": cand_white, "result": result,
        "termination": termination, "cand_outcome": cand_outcome,
        "plies": board.ply(), "moves": moves, "pgn": str(game), "final_fen": board.fen(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-dir", type=Path, required=True)
    ap.add_argument("--baseline-dir", type=Path, required=True)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--base-ms", type=int, default=20_000)
    ap.add_argument("--inc-ms", type=int, default=200)
    ap.add_argument("--ply-cap", type=int, default=300)
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/claude-501/ablation"))
    ap.add_argument("--tag", type=str, default="ablation")
    args = ap.parse_args()

    out_dir = args.out_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"ablation: {args.candidate_dir} (candidate) vs {args.baseline_dir} (baseline)")
    print(f"{args.base_ms}+{args.inc_ms}ms, {args.pairs} pairs "
          f"({args.pairs * 2} games) -- NOT the exact competition clock unless "
          f"base_ms=120000 and inc_ms=500\n")

    games: list[dict] = []
    for i in range(args.pairs):
        opening, name = OPENINGS[i % len(OPENINGS)]
        for cand_white in (True, False):
            g = play_game(cand_white, opening, args.base_ms, args.inc_ms,
                           args.ply_cap, args.candidate_dir, args.baseline_dir)
            games.append(g)
            idx = len(games)
            (out_dir / f"game{idx:03d}.pgn").write_text(g["pgn"])
            print(f"  [{idx:2d}] {name[:20]:20s} cand={'W' if cand_white else 'B'}  "
                  f"{g['cand_outcome']:4s} ({g['termination']}, {g['plies']} plies)")

    w = sum(1 for g in games if g["cand_outcome"] == "cand")
    d = sum(1 for g in games if g["cand_outcome"] == "draw")
    losses = sum(1 for g in games if g["cand_outcome"] == "base")
    n = len(games)
    score, ci_lo, ci_hi = _score_ci95(w, d, n)
    failures = [g for g in games if "crash" in g["termination"] or "illegal" in g["termination"]
                or g["termination"] == "flag"]

    summary = {
        "candidate_dir": str(args.candidate_dir), "baseline_dir": str(args.baseline_dir),
        "base_ms": args.base_ms, "inc_ms": args.inc_ms, "n_games": n,
        "w": w, "d": d, "l": losses, "score": score, "ci95": [ci_lo, ci_hi],
        "failures": [{"idx": i, "termination": g["termination"]}
                     for i, g in enumerate(games) if g in failures],
        "generated_at": datetime.now(UTC).isoformat(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "games.json").write_text(json.dumps(games, indent=2))

    print(f"\n=== candidate vs baseline: +{w} ={d} -{losses}   score={score:.3f}  "
          f"95% CI [{ci_lo:.3f}, {ci_hi:.3f}]  n={n} ===")
    print(f"failures: {len(failures)}")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
