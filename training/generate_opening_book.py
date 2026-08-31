#!/usr/bin/env python3
"""Generate a deterministic compact opening book from an offline engine.

The tree begins only from the standard initial position and follows a recorded
branch schedule. It never reads the protected benchmark opening list. The
external engine and this generator are development-only and are not packaged.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import deque
from pathlib import Path
from typing import Any

import chess
import chess.engine

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_stockfish_evaluator import file_sha256  # noqa: E402


def position_key(board: chess.Board) -> str:
    """Position identity without move clocks, preserving legal move state."""
    return " ".join(board.fen(en_passant="fen").split()[:4])


def parse_schedule(raw: str) -> tuple[int, ...]:
    try:
        schedule = tuple(int(value) for value in raw.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("schedule must contain comma-separated integers") from exc
    if not schedule or any(value < 1 for value in schedule):
        raise argparse.ArgumentTypeError("schedule entries must be positive")
    return schedule


def generate(
    engine_path: Path,
    nodes: int,
    schedule: tuple[int, ...],
    candidate_moves: int,
    maximum_positions: int,
    progress_every: int,
) -> dict[str, list[str]]:
    queue: deque[tuple[chess.Board, int]] = deque([(chess.Board(), 0)])
    queued = {position_key(queue[0][0])}
    book: dict[str, list[str]] = {}
    with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
        engine.configure({"Threads": 1, "Hash": 64})
        while queue and len(book) < maximum_positions:
            board, ply = queue.popleft()
            if board.is_game_over(claim_draw=True) or ply >= len(schedule):
                continue
            legal_count = board.legal_moves.count()
            analysis_count = min(max(schedule[ply], candidate_moves), legal_count)
            analyses = engine.analyse(
                board,
                chess.engine.Limit(nodes=nodes),
                multipv=analysis_count,
            )
            if not isinstance(analyses, list):
                analyses = [analyses]
            ranked_moves = [
                information["pv"][0]
                for information in analyses
                if information.get("pv")
            ]
            if not ranked_moves:
                continue
            key = position_key(board)
            book[key] = [move.uci() for move in ranked_moves[:candidate_moves]]
            if ply + 1 < len(schedule):
                for move in ranked_moves[: schedule[ply]]:
                    child = board.copy(stack=False)
                    child.push(move)
                    child_key = position_key(child)
                    if child_key not in queued:
                        queued.add(child_key)
                        queue.append((child, ply + 1))
            if progress_every and len(book) % progress_every == 0:
                print(f"generated {len(book)} book positions", flush=True)
    return book


def payload(
    book: dict[str, list[str]],
    engine_path: Path,
    nodes: int,
    schedule: tuple[int, ...],
    candidate_moves: int,
    maximum_positions: int,
) -> dict[str, Any]:
    script = Path(__file__)
    return {
        "schema_version": 1,
        "kind": "offline_teacher_opening_book",
        "position_key": "FEN piece placement, turn, castling, and en-passant fields",
        "generation": {
            "root": chess.STARTING_FEN,
            "nodes_per_position": nodes,
            "branch_schedule": list(schedule),
            "candidate_moves_per_position": candidate_moves,
            "maximum_positions": maximum_positions,
            "script": str(script.relative_to(Path.cwd())),
            "script_sha256": file_sha256(script),
            "teacher_name": "Stockfish 18",
            "teacher_binary_sha256": file_sha256(engine_path),
            "protected_opening_list_used": False,
        },
        "positions": len(book),
        "moves": book,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=1000)
    parser.add_argument("--branch-schedule", type=parse_schedule, default=(4, 4, 4, 4, 2, 2, 2, 2))
    parser.add_argument("--candidate-moves", type=int, default=3)
    parser.add_argument("--maximum-positions", type=int, default=4000)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--output", type=Path, default=Path("weights/opening-book.json"))
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()
    if args.nodes < 1 or args.maximum_positions < 1 or args.candidate_moves < 1:
        parser.error("nodes, candidate moves, and maximum positions must be positive")
    discovered = shutil.which("stockfish") if args.engine is None else str(args.engine)
    if discovered is None:
        parser.error("Stockfish is required for offline book generation; pass --engine")
    engine_path = Path(discovered).resolve()
    book = generate(
        engine_path,
        args.nodes,
        args.branch_schedule,
        args.candidate_moves,
        args.maximum_positions,
        args.progress_every,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            payload(
                book,
                engine_path,
                args.nodes,
                args.branch_schedule,
                args.candidate_moves,
                args.maximum_positions,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps({"positions": len(book), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
