#!/usr/bin/env python3
"""Measure full-agent move agreement with grouped MultiPV teacher decisions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import chess

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.rules import INIT_BUDGET_S  # noqa: E402
from harness.sandbox import AgentFailure, local  # noqa: E402
from training.train_active_move_ordering import move_between  # noqa: E402
from training.train_active_residual_evaluator import load_active_dataset  # noqa: E402


def contexts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["game_id"]), int(row["parent_ply"])), []).append(row)
    result: list[dict[str, Any]] = []
    for (game_id, parent_ply), group in grouped.items():
        parent = next(row for row in group if row["source"] == "parent")
        ranked = sorted(
            (row for row in group if row["teacher_rank"] is not None),
            key=lambda row: int(row["teacher_rank"]),
        )
        if not ranked:
            continue
        parent_fen = str(parent["fen"])
        moves = [move_between(parent_fen, str(row["fen"])).uci() for row in ranked]
        result.append(
            {
                "game_id": game_id,
                "parent_ply": parent_ply,
                "fen": parent_fen,
                "regret": float(parent["regret"]),
                "teacher_moves": moves,
            }
        )
    return sorted(
        result,
        key=lambda row: (-float(row["regret"]), int(row["game_id"]), int(row["parent_ply"])),
    )


def evaluate_agent(
    root: Path, positions: list[dict[str, Any]], time_left_ms: int
) -> dict[str, Any]:
    agent = local(root.resolve())
    decisions: list[dict[str, Any]] = []
    failures: list[str] = []
    try:
        agent.start(INIT_BUDGET_S)
        for context in positions:
            fen = str(context["fen"])
            try:
                uci = agent.move(fen, time_left_ms)
                move = chess.Move.from_uci(uci)
                if move not in chess.Board(fen).legal_moves:
                    raise ValueError(f"illegal move {uci}")
            except (AgentFailure, ValueError, chess.InvalidMoveError) as error:
                failures.append(
                    f"game={context['game_id']} ply={context['parent_ply']}: {error}"
                )
                continue
            teacher_moves = list(context["teacher_moves"])
            rank = teacher_moves.index(uci) + 1 if uci in teacher_moves else None
            decisions.append(
                {
                    "game_id": context["game_id"],
                    "parent_ply": context["parent_ply"],
                    "regret": context["regret"],
                    "move": uci,
                    "teacher_rank": rank,
                    "teacher_moves": teacher_moves,
                    "fen": fen,
                }
            )
    finally:
        agent.stop()
    count = len(positions)
    top_one = sum(decision["teacher_rank"] == 1 for decision in decisions)
    top_three = sum(
        decision["teacher_rank"] is not None and int(decision["teacher_rank"]) <= 3
        for decision in decisions
    )
    return {
        "root": str(root),
        "positions": count,
        "time_left_ms": time_left_ms,
        "failures": failures,
        "top1": top_one / count,
        "top3": top_three / count,
        "unlisted": sum(decision["teacher_rank"] is None for decision in decisions),
        "decisions": decisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-root", type=Path, action="append", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--time-left-ms", type=int, default=3000)
    parser.add_argument("--maximum-contexts", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.time_left_ms < 1:
        parser.error("time left must be positive")
    rows, metadata = load_active_dataset(args.dataset)
    positions = contexts(rows)
    if args.maximum_contexts is not None:
        if args.maximum_contexts < 1:
            parser.error("maximum contexts must be positive")
        positions = positions[: args.maximum_contexts]
    payload = {
        "schema_version": 1,
        "dataset_sha256": metadata["dataset_sha256"],
        "agents": [
            evaluate_agent(root, positions, args.time_left_ms) for root in args.agent_root
        ],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
