#!/usr/bin/env python3
"""Summarize tactical shape and regret in a retained hard-position dataset."""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import chess
import chess.pgn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_active_move_ordering import move_between  # noqa: E402
from training.train_active_residual_evaluator import load_active_dataset  # noqa: E402


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lower = int(index)
    upper = min(len(ordered) - 1, lower + 1)
    blend = index - lower
    return ordered[lower] * (1.0 - blend) + ordered[upper] * blend


def move_kind(board: chess.Board, move: chess.Move) -> str:
    if move.promotion:
        return "promotion"
    if board.is_capture(move):
        return "capture_check" if board.gives_check(move) else "capture"
    if board.gives_check(move):
        return "quiet_check"
    if board.is_castling(move):
        return "castle"
    return "quiet"


def played_moves(payload: dict[str, Any]) -> dict[tuple[int, int], str]:
    result: dict[tuple[int, int], str] = {}
    for retained in payload.get("games", []):
        game_id = int(retained["game_id"])
        game = chess.pgn.read_game(io.StringIO(str(retained["pgn"])))
        if game is None:
            raise ValueError(f"could not parse retained game {game_id}")
        board = game.board()
        for move in game.mainline_moves():
            result[(game_id, board.ply())] = move.uci()
            board.push(move)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    if args.top < 1:
        parser.error("top must be positive")
    payload = json.loads(args.dataset.read_text())
    rows, _ = load_active_dataset(args.dataset)
    actual = played_moves(payload)
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["game_id"]), int(row["parent_ply"])), []).append(row)
    contexts: list[dict[str, Any]] = []
    for key, group in grouped.items():
        parent = next(row for row in group if row["source"] == "parent")
        best = next(row for row in group if row["teacher_rank"] == 1)
        board = chess.Board(str(parent["fen"]))
        best_move = move_between(board.fen(), str(best["fen"]))
        actual_uci = actual.get(key)
        if actual_uci is None:
            raise ValueError(f"retained game lacks sampled move {key}")
        actual_move = chess.Move.from_uci(actual_uci)
        contexts.append(
            {
                "game_id": key[0],
                "ply": key[1],
                "regret": float(parent["regret"]),
                "in_check": board.is_check(),
                "legal_moves": board.legal_moves.count(),
                "actual_move": actual_move.uci(),
                "actual_san": board.san(actual_move),
                "actual_kind": move_kind(board, actual_move),
                "teacher_move": best_move.uci(),
                "teacher_san": board.san(best_move),
                "teacher_kind": move_kind(board, best_move),
                "fen": board.fen(),
            }
        )
    regrets = [float(context["regret"]) for context in contexts]
    serious = [context for context in contexts if float(context["regret"]) >= 100.0]
    summary = {
        "schema_version": 1,
        "dataset_sha256": payload["dataset_sha256"],
        "contexts": len(contexts),
        "regret_cp": {
            "mean": sum(regrets) / len(regrets),
            "median": percentile(regrets, 0.5),
            "p75": percentile(regrets, 0.75),
            "p90": percentile(regrets, 0.9),
            "p95": percentile(regrets, 0.95),
            "maximum": max(regrets),
            "at_least_100": sum(regret >= 100.0 for regret in regrets),
            "at_least_300": sum(regret >= 300.0 for regret in regrets),
            "at_least_1000": sum(regret >= 1000.0 for regret in regrets),
        },
        "serious_error_shape": {
            "contexts": len(serious),
            "in_check": sum(bool(context["in_check"]) for context in serious),
            "actual_move_kinds": dict(Counter(str(row["actual_kind"]) for row in serious)),
            "teacher_move_kinds": dict(Counter(str(row["teacher_kind"]) for row in serious)),
        },
        "largest_errors": sorted(
            contexts,
            key=lambda context: (
                -float(context["regret"]),
                int(context["game_id"]),
                int(context["ply"]),
            ),
        )[: args.top],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
