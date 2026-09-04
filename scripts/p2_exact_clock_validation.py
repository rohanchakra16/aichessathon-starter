"""Exact-clock validation: Phineas 2 vs a strength-limited local Stockfish.

Dev-only calibration tool, not a promotion gate, not shipped, not run at
runtime. Stockfish is used exactly as the rules permit for development: an
offline sparring/teacher engine, never bundled into or called by agent.py.

Each game runs Phineas 2 in its own fresh subprocess (scripts/p2_worker.py),
mirroring the real "one process per game" competition protocol, including
paying the numba JIT cost once at the start of the game rather than on the
clock. Openings are fixed and preregistered in OPENINGS below -- the same
list is reused for every run so results are comparable and nothing is
cherry-picked after the fact. Saves one PGN per game plus a single JSON
summary with every move's timing/depth/score and any failures.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import chess
import chess.engine
import chess.pgn

WT = Path(__file__).resolve().parent.parent
STOCKFISH = "/opt/homebrew/bin/stockfish"
VENV_PY = (
    "/Users/phantomvenom/Documents/Codex/2026-08-30/referenced-chatgpt-conversation-this-is-an/"
    "outputs/aichessathon-starter/.venv/bin/python"
)

# Preregistered, fixed for every run of this script -- do not edit between a
# screening run and the confirmation run it is meant to be compared against.
OPENINGS = [
    ("startpos", "Start position"),
    ("1.e4 e5 2.Nf3 Nc6 3.Bb5", "Ruy Lopez"),
    ("1.d4 Nf6 2.c4 e6 3.Nc3 Bb4", "Nimzo-Indian"),
    ("1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 a6", "Najdorf Sicilian"),
    ("1.c4 e5 2.Nc3 Nf6", "English"),
    ("1.e4 e6 2.d4 d5 3.Nc3 Bb4", "French Winawer"),
    ("1.d4 d5 2.c4 c6 3.Nf3 Nf6 4.Nc3 dxc4", "Slav"),
    ("1.e4 c6 2.d4 d5 3.e5", "Caro-Kann Advance"),
    ("1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6", "King's Indian"),
    ("1.Nf3 d5 2.g3 Nf6 3.Bg2 e6", "Reti/Catalan-ish"),
    ("1.e4 e5 2.Nf3 Nf6", "Petrov"),
    ("1.d4 f5", "Dutch"),
]


def _board_from(opening: str) -> chess.Board:
    b = chess.Board()
    if opening != "startpos":
        for tok in opening.replace(".", ". ").split():
            if not tok.endswith("."):
                b.push_san(tok)
    return b


class Worker:
    def __init__(self, target_dir: Path = WT) -> None:
        t0 = time.monotonic()
        self.proc = subprocess.Popen(
            [VENV_PY, "-u", str(WT / "scripts" / "p2_worker.py")],
            cwd=str(target_dir), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        ready = self.proc.stdout.readline().strip()
        self.import_time_s = time.monotonic() - t0
        if ready != "READY":
            err = self.proc.stderr.read()
            raise RuntimeError(f"worker failed to start: {ready!r} stderr={err[-2000:]}")

    def get_move(self, fen: str, time_left_ms: int) -> tuple[str | None, dict]:
        self.proc.stdin.write(f"{fen}\t{int(time_left_ms)}\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read()
            return None, {"error": f"worker died, stderr={err[-2000:]}"}
        line = line.strip()
        if line.startswith("ERROR"):
            return None, {"error": line}
        uci, score, nodes, depth, elapsed_ms = line.split("\t")
        return uci, {
            "score": int(score), "nodes": int(nodes),
            "depth": int(depth), "elapsed_ms": float(elapsed_ms),
        }

    def close(self) -> None:
        try:
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def play_game(sf: chess.engine.SimpleEngine, p2_white: bool, opening: str,
              base_ms: int, inc_ms: int, ply_cap: int, target_dir: Path = WT) -> dict:
    board = _board_from(opening)
    worker = Worker(target_dir)
    p2_side = chess.WHITE if p2_white else chess.BLACK
    clocks = {chess.WHITE: float(base_ms), chess.BLACK: float(base_ms)}
    moves: list[dict] = []
    game = chess.pgn.Game()
    game.headers["White"] = "Phineas2" if p2_white else "Stockfish18(limited)"
    game.headers["Black"] = "Stockfish18(limited)" if p2_white else "Phineas2"
    game.headers["TimeControl"] = f"{base_ms // 1000}+{inc_ms / 1000.0:g}"
    node = game
    for opening_move in board.move_stack:  # seed the PGN tree with the opening
        node = node.add_variation(opening_move)
    result, termination = "draw", "unknown"

    try:
        while not board.is_game_over(claim_draw=True) and board.ply() < ply_cap:
            stm = board.turn
            t0 = time.monotonic()
            if stm == p2_side:
                uci, info = worker.get_move(board.fen(), int(clocks[stm]))
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                if uci is None:
                    result = "black" if p2_side == chess.WHITE else "white"
                    termination = f"crash:{info.get('error')}"
                    break
                move = chess.Move.from_uci(uci)
                moves.append({"ply": board.ply(), "side": "p2", "uci": uci, **info})
            else:
                limit = chess.engine.Limit(
                    white_clock=clocks[chess.WHITE] / 1000.0,
                    black_clock=clocks[chess.BLACK] / 1000.0,
                    white_inc=inc_ms / 1000.0, black_inc=inc_ms / 1000.0,
                )
                sf_result = sf.play(board, limit)
                move = sf_result.move
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                moves.append({"ply": board.ply(), "side": "sf",
                             "uci": move.uci() if move else None, "elapsed_ms": elapsed_ms})

            clocks[stm] -= elapsed_ms
            if clocks[stm] < 0:
                termination = "flag"
                result = "black" if stm == chess.WHITE else "white"
                break
            clocks[stm] += inc_ms

            if move is None or move not in board.legal_moves:
                termination = f"illegal:{move}"
                result = "black" if stm == chess.WHITE else "white"
                break
            node = node.add_variation(move)
            board.push(move)
        else:
            if board.is_checkmate():
                result = "white" if board.turn == chess.BLACK else "black"
                termination = "checkmate"
            elif board.ply() >= ply_cap:
                result = "draw"
                termination = "ply_cap_adjudicated"
            else:
                outcome = board.outcome(claim_draw=True)
                result = "draw"
                termination = outcome.termination.name.lower() if outcome else "adjudicated"
    finally:
        worker.close()

    game.headers["Result"] = {"white": "1-0", "black": "0-1", "draw": "1/2-1/2"}[result]
    p2_won = (result == "white" and p2_side == chess.WHITE) or \
             (result == "black" and p2_side == chess.BLACK)
    p2_outcome = "p2" if p2_won else ("draw" if result == "draw" else "sf")

    return {
        "opening": opening, "p2_white": p2_white, "result": result,
        "termination": termination, "p2_outcome": p2_outcome,
        "worker_import_s": worker.import_time_s, "plies": board.ply(),
        "moves": moves, "pgn": str(game),
        "final_fen": board.fen(),
    }


def _score_ci95(w: int, d: int, n: int) -> tuple[float, float, float]:
    """Score and a 95% normal-approx CI on the score (draws count 0.5)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    score = (w + 0.5 * d) / n
    var = sum(
        ((1.0 if i < w else (0.5 if i < w + d else 0.0)) - score) ** 2 for i in range(n)
    ) / n
    se = (var / n) ** 0.5
    return score, max(0.0, score - 1.96 * se), min(1.0, score + 1.96 * se)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--elo", type=int, default=1800)
    ap.add_argument("--pairs", type=int, default=7)
    ap.add_argument("--base-ms", type=int, default=120_000)
    ap.add_argument("--inc-ms", type=int, default=500)
    ap.add_argument("--ply-cap", type=int, default=300)
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/claude-501/exact_clock"))
    ap.add_argument("--tag", type=str, default="run")
    ap.add_argument("--target-dir", type=Path, default=WT,
                     help="directory whose agent.py/weights/ are under test "
                          "(defaults to the live worktree; point this at a "
                          "separate frozen-tag checkout to test a pinned baseline "
                          "without any risk of a concurrent code edit contaminating "
                          "an in-flight run)")
    args = ap.parse_args()

    if not Path(STOCKFISH).exists():
        raise SystemExit(f"stockfish not found at {STOCKFISH}")

    out_dir = args.out_dir / f"{args.tag}_elo{args.elo}"
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": args.elo, "Threads": 1})

    is_exact_clock = args.base_ms == 120_000 and args.inc_ms == 500
    clock_claim = (
        "COMPETITION CLOCK (120000+500)" if is_exact_clock
        else f"NON-STANDARD CLOCK {args.base_ms}+{args.inc_ms} -- not a 120+0.5 claim"
    )
    print(f"Phineas2 vs Stockfish18(elo={args.elo}, limited)  {clock_claim}")
    print(f"target: {args.target_dir}")
    print(f"{args.pairs} pairs ({args.pairs * 2} games), openings cycle over "
          f"{min(args.pairs, len(OPENINGS))} preregistered entries\n")

    games: list[dict] = []
    try:
        for i in range(args.pairs):
            opening, name = OPENINGS[i % len(OPENINGS)]
            for p2_white in (True, False):
                g = play_game(engine, p2_white, opening, args.base_ms, args.inc_ms,
                               args.ply_cap, args.target_dir)
                games.append(g)
                idx = len(games)
                (out_dir / f"game{idx:03d}.pgn").write_text(g["pgn"])
                print(f"  [{idx:2d}] {name[:20]:20s} p2={'W' if p2_white else 'B'}  "
                      f"{g['p2_outcome']:4s} ({g['termination']}, {g['plies']} plies, "
                      f"import {g['worker_import_s']:.1f}s)")
    finally:
        engine.quit()

    w = sum(1 for g in games if g["p2_outcome"] == "p2")
    d = sum(1 for g in games if g["p2_outcome"] == "draw")
    losses = sum(1 for g in games if g["p2_outcome"] == "sf")
    n = len(games)
    score, ci_lo, ci_hi = _score_ci95(w, d, n)

    by_colour = {}
    for colour, label in ((True, "white"), (False, "black")):
        sub = [g for g in games if g["p2_white"] == colour]
        cw = sum(1 for g in sub if g["p2_outcome"] == "p2")
        cd = sum(1 for g in sub if g["p2_outcome"] == "draw")
        cl = sum(1 for g in sub if g["p2_outcome"] == "sf")
        by_colour[label] = {"w": cw, "d": cd, "l": cl, "n": len(sub)}

    p2_move_times = [m["elapsed_ms"] for g in games for m in g["moves"] if m["side"] == "p2"]
    p2_depths = [m["depth"] for g in games for m in g["moves"] if m["side"] == "p2"]
    failures = [g for g in games if "crash" in g["termination"] or "illegal" in g["termination"]
                or g["termination"] == "flag"]
    winning_draws = [
        g for g in games
        if g["p2_outcome"] == "draw"
        and any(m["side"] == "p2" and m.get("score", 0) > 150 for m in g["moves"][-12:])
    ]

    summary = {
        "elo": args.elo, "base_ms": args.base_ms, "inc_ms": args.inc_ms,
        "n_games": n, "w": w, "d": d, "l": losses, "score": score,
        "ci95": [ci_lo, ci_hi], "by_colour": by_colour,
        "avg_move_ms": sum(p2_move_times) / len(p2_move_times) if p2_move_times else 0,
        "max_move_ms": max(p2_move_times) if p2_move_times else 0,
        "avg_depth": sum(p2_depths) / len(p2_depths) if p2_depths else 0,
        "min_depth": min(p2_depths) if p2_depths else 0,
        "max_depth": max(p2_depths) if p2_depths else 0,
        "failures": [{"idx": i, "opening": g["opening"], "p2_white": g["p2_white"],
                     "termination": g["termination"]} for i, g in enumerate(games)
                     if g in failures],
        "winning_position_draws": [{"idx": i, "opening": g["opening"]}
                                   for i, g in enumerate(games) if g in winning_draws],
        "generated_at": datetime.now(UTC).isoformat(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "games.json").write_text(json.dumps(games, indent=2))

    print(f"\n=== Phineas2 vs Stockfish(elo={args.elo}) @ {args.base_ms}+{args.inc_ms} ===")
    print(f"  +{w} ={d} -{losses}   score={score:.3f}  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]  n={n}")
    print(f"  by colour: white {by_colour['white']}  black {by_colour['black']}")
    print(f"  move time: avg {summary['avg_move_ms']:.0f}ms  max {summary['max_move_ms']:.0f}ms")
    print(f"  depth: avg {summary['avg_depth']:.1f}  range "
          f"[{summary['min_depth']},{summary['max_depth']}]")
    print(f"  failures: {len(failures)}   winning-position draws: {len(winning_draws)}")
    print(f"  saved to {out_dir}")


if __name__ == "__main__":
    main()
