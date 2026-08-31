#!/usr/bin/env python3
"""Generate an independent, near-level confirmation opening set.

This script never reads the promotion arena openings. It samples among a
development-only teacher's top moves from the standard initial position and
keeps only final positions the teacher still considers close to equal.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import chess
import chess.engine

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_stockfish_evaluator import file_sha256  # noqa: E402

SEED = 2026083102
SELECTION_WEIGHTS = (0.68, 0.24, 0.08)


def choose_analysis(analyses: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    return rng.choices(
        analyses,
        weights=SELECTION_WEIGHTS[: len(analyses)],
        k=1,
    )[0]


def generate(
    count: int,
    plies: int,
    nodes: int,
    final_nodes: int,
    maximum_abs_score: int,
    engine_path: Path,
) -> dict[str, Any]:
    rng = random.Random(SEED)
    openings: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    attempts = 0
    with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
        engine.configure({"Threads": 1, "Hash": 64})
        while len(openings) < count:
            attempts += 1
            board = chess.Board()
            moves: list[str] = []
            for _ in range(plies):
                analyses = engine.analyse(
                    board,
                    chess.engine.Limit(nodes=nodes),
                    multipv=min(3, board.legal_moves.count()),
                )
                if not isinstance(analyses, list):
                    analyses = [analyses]
                usable = [item for item in analyses if item.get("pv")]
                if not usable:
                    break
                move = choose_analysis(usable, rng)["pv"][0]
                moves.append(move.uci())
                board.push(move)
            sequence = tuple(moves)
            if len(sequence) != plies or sequence in seen or board.is_game_over():
                continue
            information = engine.analyse(board, chess.engine.Limit(nodes=final_nodes))
            score = information["score"].white().score(mate_score=10_000)
            if score is None or abs(score) > maximum_abs_score:
                continue
            seen.add(sequence)
            openings.append(
                {
                    "id": f"confirmation-{len(openings) + 1:02d}",
                    "moves": list(sequence),
                }
            )
    return {
        "schema_version": 1,
        "kind": "independent_engine_guided_confirmation_openings",
        "seed": SEED,
        "attempts": attempts,
        "opening_plies": plies,
        "nodes_per_move": nodes,
        "final_validation_nodes": final_nodes,
        "maximum_absolute_final_score_centipawns": maximum_abs_score,
        "selection_weights": list(SELECTION_WEIGHTS),
        "teacher_name": "Stockfish 18",
        "teacher_binary_sha256": file_sha256(engine_path),
        "promotion_opening_list_used": False,
        "openings": openings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--plies", type=int, default=10)
    parser.add_argument("--nodes", type=int, default=2500)
    parser.add_argument("--final-nodes", type=int, default=5000)
    parser.add_argument("--maximum-abs-score", type=int, default=60)
    parser.add_argument("--engine", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".autoloop/protected/confirmation-openings.json"),
    )
    args = parser.parse_args()
    if (
        args.count < 1
        or args.plies < 1
        or args.nodes < 1
        or args.final_nodes < 1
        or args.maximum_abs_score < 0
    ):
        parser.error("generation arguments must be positive")
    discovered = shutil.which("stockfish") if args.engine is None else str(args.engine)
    if discovered is None:
        parser.error("Stockfish is required for offline opening generation; pass --engine")
    engine_path = Path(discovered).resolve()
    payload = generate(
        args.count,
        args.plies,
        args.nodes,
        args.final_nodes,
        args.maximum_abs_score,
        engine_path,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "openings": len(payload["openings"]),
                "attempts": payload["attempts"],
                "teacher_binary_sha256": payload["teacher_binary_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
