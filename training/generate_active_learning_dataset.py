#!/usr/bin/env python3
"""Generate game-grouped positions where the current champion disagrees with a teacher.

The champion supplies realistic trajectories from independently sampled openings.
Stockfish is used only offline to rank moves and label the parent/child positions.
No protected promotion or confirmation opening file is read.
"""

from __future__ import annotations

import argparse
import hashlib
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

from harness.rules import INIT_BUDGET_S  # noqa: E402
from harness.sandbox import AgentFailure, local  # noqa: E402
from training.train_stockfish_evaluator import file_sha256  # noqa: E402

ACTIVE_SEED = 2026090101
OPENING_WEIGHTS = (0.65, 0.25, 0.10)


def score_cp(board: chess.Board, information: dict[str, Any]) -> float:
    score = information["score"].pov(board.turn).score(mate_score=10_000)
    if score is None:
        raise RuntimeError(f"teacher produced no score for {board.fen()}")
    return float(max(-2_000, min(2_000, score)))


def choose_teacher_move(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    rng: random.Random,
    nodes: int,
) -> chess.Move:
    count = min(len(OPENING_WEIGHTS), board.legal_moves.count())
    analyses = engine.analyse(board, chess.engine.Limit(nodes=nodes), multipv=count)
    if not isinstance(analyses, list):
        analyses = [analyses]
    usable = [information for information in analyses if information.get("pv")]
    if not usable:
        raise RuntimeError(f"teacher produced no opening move for {board.fen()}")
    choice = rng.choices(range(len(usable)), weights=OPENING_WEIGHTS[: len(usable)], k=1)[0]
    return usable[choice]["pv"][0]


def opening_board(
    engine: chess.engine.SimpleEngine,
    rng: random.Random,
    nodes: int,
    plies: int,
) -> chess.Board:
    board = chess.Board()
    for _ in range(plies):
        if board.is_game_over(claim_draw=True):
            break
        board.push(choose_teacher_move(board, engine, rng, nodes))
    return board


def champion_trajectory(
    champion_root: Path,
    board: chess.Board,
    game_id: int,
    maximum_plies: int,
    time_left_ms: int,
    sample_stride: int,
) -> list[dict[str, Any]]:
    agents = {chess.WHITE: local(champion_root), chess.BLACK: local(champion_root)}
    contexts: list[dict[str, Any]] = []
    try:
        for agent in agents.values():
            agent.start(INIT_BUDGET_S)
        while not board.is_game_over(claim_draw=True) and board.ply() < maximum_plies:
            mover = board.turn
            fen = board.fen()
            uci = agents[mover].move(fen, time_left_ms)
            try:
                move = chess.Move.from_uci(uci)
            except chess.InvalidMoveError as error:
                raise RuntimeError(f"champion returned malformed move {uci!r}") from error
            if move not in board.legal_moves:
                raise RuntimeError(f"champion returned illegal move {uci!r} for {fen}")
            if board.ply() % sample_stride == 0:
                contexts.append(
                    {
                        "game_id": game_id,
                        "ply": board.ply(),
                        "fen": fen,
                        "champion_move": move.uci(),
                    }
                )
            board.push(move)
    except AgentFailure as error:
        raise RuntimeError(f"champion failed during active trajectory: {error.reason}") from error
    finally:
        for agent in agents.values():
            agent.stop()
    return contexts


def annotate_context(
    context: dict[str, Any],
    engine: chess.engine.SimpleEngine,
    nodes: int,
    multipv: int,
) -> dict[str, Any]:
    board = chess.Board(context["fen"])
    count = min(multipv, board.legal_moves.count())
    analyses = engine.analyse(board, chess.engine.Limit(nodes=nodes), multipv=count)
    if not isinstance(analyses, list):
        analyses = [analyses]
    usable = [information for information in analyses if information.get("pv")]
    if not usable:
        raise RuntimeError(f"teacher produced no active-learning line for {board.fen()}")
    champion_move = chess.Move.from_uci(context["champion_move"])
    by_move = {information["pv"][0]: information for information in usable}
    champion_information = by_move.get(champion_move)
    if champion_information is None:
        forced = engine.analyse(
            board,
            chess.engine.Limit(nodes=nodes),
            root_moves=[champion_move],
        )
        if isinstance(forced, list):
            forced = forced[0]
        champion_information = forced
    best_score = score_cp(board, usable[0])
    champion_score = score_cp(board, champion_information)
    return {
        **context,
        "best_score": best_score,
        "champion_score": champion_score,
        "regret": max(0.0, best_score - champion_score),
        "teacher_lines": [
            {"move": information["pv"][0].uci(), "score": score_cp(board, information)}
            for information in usable
        ],
    }


def select_contexts(
    contexts: list[dict[str, Any]], count: int, rng: random.Random
) -> list[dict[str, Any]]:
    if count >= len(contexts):
        return contexts
    by_game: dict[int, list[dict[str, Any]]] = {}
    for context in contexts:
        by_game.setdefault(int(context["game_id"]), []).append(context)
    coverage: list[dict[str, Any]] = []
    if count >= len(by_game):
        for game_id in sorted(by_game):
            choices = sorted(by_game[game_id], key=lambda item: int(item["ply"]))
            coverage.append(rng.choice(choices))
    coverage_keys = {(item["game_id"], item["ply"]) for item in coverage}
    remaining_count = count - len(coverage)
    active_count = remaining_count // 2
    ordered = sorted(
        [
            item
            for item in contexts
            if (item["game_id"], item["ply"]) not in coverage_keys
        ],
        key=lambda item: (-float(item["regret"]), int(item["game_id"]), int(item["ply"])),
    )
    active = ordered[:active_count]
    active_keys = coverage_keys | {(item["game_id"], item["ply"]) for item in active}
    exploration = [
        item for item in contexts if (item["game_id"], item["ply"]) not in active_keys
    ]
    rng.shuffle(exploration)
    selected = coverage + active + exploration[: count - len(coverage) - active_count]
    return sorted(selected, key=lambda item: (int(item["game_id"]), int(item["ply"])))


def rows_from_contexts(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for context in contexts:
        group = int(context["game_id"])
        parent = chess.Board(context["fen"])
        candidates = [
            ("parent", parent.fen(), float(context["best_score"]), None),
        ]
        teacher_moves: set[str] = set()
        for rank, line in enumerate(context["teacher_lines"], 1):
            move = chess.Move.from_uci(line["move"])
            child = parent.copy(stack=False)
            child.push(move)
            teacher_moves.add(move.uci())
            candidates.append((f"teacher_child_{rank}", child.fen(), -float(line["score"]), rank))
        champion_move = str(context["champion_move"])
        if champion_move not in teacher_moves:
            child = parent.copy(stack=False)
            child.push_uci(champion_move)
            candidates.append(
                ("champion_child", child.fen(), -float(context["champion_score"]), None)
            )
        for source, fen, label, rank in candidates:
            key = (group, fen)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "game_id": group,
                    "parent_ply": int(context["ply"]),
                    "source": source,
                    "teacher_rank": rank,
                    "regret": float(context["regret"]),
                    "fen": fen,
                    "label": label,
                }
            )
    return rows


def dataset_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (
                f"{row['game_id']}\t{row['parent_ply']}\t{row['source']}\t"
                f"{row['fen']}\t{float(row['label']):.1f}\n"
            ).encode()
        )
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--champion-root", type=Path, default=ROOT)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--opening-plies", type=int, default=8)
    parser.add_argument("--maximum-plies", type=int, default=72)
    parser.add_argument("--champion-time-left-ms", type=int, default=1000)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--opening-nodes", type=int, default=1000)
    parser.add_argument("--label-nodes", type=int, default=2500)
    parser.add_argument("--multipv", type=int, default=3)
    parser.add_argument("--selected-contexts", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=8)
    args = parser.parse_args()
    if min(
        args.games,
        args.opening_plies,
        args.maximum_plies,
        args.champion_time_left_ms,
        args.sample_stride,
        args.opening_nodes,
        args.label_nodes,
        args.multipv,
        args.selected_contexts,
    ) < 1:
        parser.error("all numeric settings must be positive")
    if args.opening_plies >= args.maximum_plies:
        parser.error("opening plies must be less than maximum plies")
    discovered = shutil.which("stockfish") if args.engine is None else str(args.engine)
    if discovered is None:
        parser.error("Stockfish is required for offline labels; pass --engine")
    engine_path = Path(discovered).resolve()
    champion_root = args.champion_root.resolve()
    rng = random.Random(ACTIVE_SEED)
    contexts: list[dict[str, Any]] = []
    with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
        engine.configure({"Threads": 1, "Hash": 64})
        for game_id in range(args.games):
            board = opening_board(engine, rng, args.opening_nodes, args.opening_plies)
            contexts.extend(
                champion_trajectory(
                    champion_root,
                    board,
                    game_id,
                    args.maximum_plies,
                    args.champion_time_left_ms,
                    args.sample_stride,
                )
            )
            if args.progress_every and (game_id + 1) % args.progress_every == 0:
                print(
                    f"generated {game_id + 1}/{args.games} champion games; "
                    f"{len(contexts)} sampled positions",
                    flush=True,
                )
        annotated: list[dict[str, Any]] = []
        for index, context in enumerate(contexts, 1):
            annotated.append(annotate_context(context, engine, args.label_nodes, args.multipv))
            if args.progress_every and index % (args.progress_every * 16) == 0:
                print(f"annotated {index}/{len(contexts)} positions", flush=True)
    selected = select_contexts(annotated, args.selected_contexts, rng)
    rows = rows_from_contexts(selected)
    digest = dataset_digest(rows)
    payload = {
        "schema_version": 1,
        "kind": "champion_disagreement_active_learning_dataset",
        "seed": ACTIVE_SEED,
        "games": args.games,
        "opening_plies": args.opening_plies,
        "maximum_plies": args.maximum_plies,
        "champion_time_left_ms": args.champion_time_left_ms,
        "sample_stride": args.sample_stride,
        "opening_nodes": args.opening_nodes,
        "label_nodes": args.label_nodes,
        "multipv": args.multipv,
        "selected_contexts": len(selected),
        "sampled_contexts": len(contexts),
        "rows_count": len(rows),
        "opening_selection_weights": list(OPENING_WEIGHTS),
        "teacher_name": "Stockfish 18",
        "teacher_binary_sha256": file_sha256(engine_path),
        "champion_agent_sha256": file_sha256(champion_root / "agent.py"),
        "champion_model_sha256": file_sha256(champion_root / "weights/model.json"),
        "protected_opening_list_used": False,
        "game_grouped": True,
        "dataset_sha256": digest,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    regrets = [float(item["regret"]) for item in selected]
    print(
        json.dumps(
            {
                "games": args.games,
                "sampled_contexts": len(contexts),
                "selected_contexts": len(selected),
                "rows": len(rows),
                "mean_selected_regret": sum(regrets) / len(regrets),
                "dataset_sha256": digest,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
