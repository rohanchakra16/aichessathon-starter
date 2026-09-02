#!/usr/bin/env python3
"""Mine and deeply label positions from champion games against Stockfish.

The match opponent is strength-limited and plays a decrementing Chessathon
clock.  After every game, the same development-only engine is restored to full
strength and used at a fixed node budget to measure the champion's move regret
and label MultiPV child positions.  Neither the engine nor this dataset is part
of a submission.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import chess
import chess.engine
import chess.pgn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.referee import PIECE_VALUES  # noqa: E402
from harness.rules import INIT_BUDGET_S  # noqa: E402
from harness.sandbox import AgentFailure, local  # noqa: E402
from training.generate_active_learning_dataset import (  # noqa: E402
    annotate_context,
    dataset_digest,
    rows_from_contexts,
    select_contexts,
)
from training.train_stockfish_evaluator import file_sha256  # noqa: E402

SEED = 2026090201
OPENING_WEIGHTS = (0.50, 0.25, 0.15, 0.07, 0.03)


def score_from_result(result: str, colour: chess.Color) -> float:
    if result == "1/2-1/2":
        return 0.5
    return float((result == "1-0") == (colour == chess.WHITE))


def opening_board(
    engine: chess.engine.SimpleEngine,
    rng: random.Random,
    nodes: int,
    plies: int,
) -> chess.Board:
    """Sample a plausible opening without reading any protected gate opening."""
    board = chess.Board()
    for _ in range(plies):
        if board.is_game_over(claim_draw=True):
            break
        count = min(len(OPENING_WEIGHTS), board.legal_moves.count())
        analyses = engine.analyse(board, chess.engine.Limit(nodes=nodes), multipv=count)
        if not isinstance(analyses, list):
            analyses = [analyses]
        moves = [information["pv"][0] for information in analyses if information.get("pv")]
        if not moves:
            raise RuntimeError(f"teacher produced no opening move for {board.fen()}")
        index = rng.choices(range(len(moves)), weights=OPENING_WEIGHTS[: len(moves)], k=1)[0]
        board.push(moves[index])
    return board


def material_result(board: chess.Board) -> str:
    balance = sum(
        value
        * (
            len(board.pieces(piece_type, chess.WHITE))
            - len(board.pieces(piece_type, chess.BLACK))
        )
        for piece_type, value in PIECE_VALUES.items()
    )
    if balance > 0:
        return "1-0"
    if balance < 0:
        return "0-1"
    return "1/2-1/2"


def render_pgn(board: chess.Board, result: str, termination: str, game_id: int) -> str:
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "Phineas Stockfish hard-position mining"
    game.headers["Round"] = str(game_id + 1)
    game.headers["Result"] = result
    game.headers["Termination"] = termination
    return str(game)


def engine_limit(
    board: chess.Board,
    white_clock_ms: float,
    black_clock_ms: float,
    increment_ms: int,
) -> chess.engine.Limit:
    return chess.engine.Limit(
        white_clock=max(0.001, white_clock_ms / 1000.0),
        black_clock=max(0.001, black_clock_ms / 1000.0),
        white_inc=increment_ms / 1000.0,
        black_inc=increment_ms / 1000.0,
    )


def play_game(
    engine: chess.engine.SimpleEngine,
    champion_root: Path,
    opening: chess.Board,
    game_id: int,
    champion_colour: chess.Color,
    base_ms: int,
    increment_ms: int,
    maximum_plies: int,
    sample_stride: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    board = opening.copy(stack=True)
    clock = {chess.WHITE: float(base_ms), chess.BLACK: float(base_ms)}
    contexts: list[dict[str, Any]] = []
    think_times: list[float] = []
    decision_index = 0
    result = "*"
    termination = "unknown"
    failure: str | None = None
    champion = local(champion_root)
    try:
        champion.start(INIT_BUDGET_S)
        while not board.is_game_over(claim_draw=True) and board.ply() < maximum_plies:
            mover = board.turn
            fen = board.fen()
            started = time.monotonic()
            if mover == champion_colour:
                uci = champion.move(fen, max(1, int(clock[mover])))
                elapsed_ms = (time.monotonic() - started) * 1000.0
                try:
                    move = chess.Move.from_uci(uci)
                except chess.InvalidMoveError as error:
                    raise RuntimeError(f"champion returned malformed move {uci!r}") from error
                if move not in board.legal_moves:
                    raise RuntimeError(f"champion returned illegal move {uci!r} for {fen}")
                if decision_index % sample_stride == 0:
                    contexts.append(
                        {
                            "game_id": game_id,
                            "ply": board.ply(),
                            "fen": fen,
                            "champion_move": move.uci(),
                        }
                    )
                decision_index += 1
                think_times.append(elapsed_ms)
            else:
                played = engine.play(
                    board,
                    engine_limit(board, clock[chess.WHITE], clock[chess.BLACK], increment_ms),
                    game=game_id,
                )
                elapsed_ms = (time.monotonic() - started) * 1000.0
                move = played.move
                if move is None:
                    raise RuntimeError(f"opponent produced no move for {fen}")
            clock[mover] -= elapsed_ms
            if clock[mover] <= 0.0:
                result = "0-1" if mover == chess.WHITE else "1-0"
                termination = "champion_flag" if mover == champion_colour else "opponent_flag"
                failure = termination if mover == champion_colour else None
                break
            board.push(move)
            clock[mover] += increment_ms
        else:
            if board.is_game_over(claim_draw=True):
                result = board.result(claim_draw=True)
                outcome = board.outcome(claim_draw=True)
                termination = (
                    outcome.termination.name.lower() if outcome is not None else "game_over"
                )
            else:
                result = material_result(board)
                termination = "material_adjudication"
    except (AgentFailure, RuntimeError) as error:
        result = "0-1" if champion_colour == chess.WHITE else "1-0"
        termination = "champion_failure"
        failure = str(error)
    finally:
        champion.stop()
    return (
        {
            "game_id": game_id,
            "champion_colour": "white" if champion_colour == chess.WHITE else "black",
            "result": result,
            "champion_score": score_from_result(result, champion_colour),
            "termination": termination,
            "failure": failure,
            "plies": board.ply(),
            "final_clock_ms": {
                "white": clock[chess.WHITE],
                "black": clock[chess.BLACK],
            },
            "mean_champion_think_ms": (
                sum(think_times) / len(think_times) if think_times else 0.0
            ),
            "maximum_champion_think_ms": max(think_times, default=0.0),
            "pgn": render_pgn(board, result, termination, game_id),
        },
        contexts,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--champion-root", type=Path, default=ROOT)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--opponent-elo", type=int, default=1600)
    parser.add_argument("--opening-plies", type=int, default=8)
    parser.add_argument("--opening-nodes", type=int, default=2000)
    parser.add_argument("--base-ms", type=int, default=120_000)
    parser.add_argument("--increment-ms", type=int, default=500)
    parser.add_argument("--maximum-plies", type=int, default=300)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--label-nodes", type=int, default=20_000)
    parser.add_argument("--multipv", type=int, default=5)
    parser.add_argument("--selected-contexts", type=int, default=256)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=1)
    args = parser.parse_args()
    numeric = (
        args.games,
        args.opponent_elo,
        args.opening_plies,
        args.opening_nodes,
        args.base_ms,
        args.increment_ms,
        args.maximum_plies,
        args.sample_stride,
        args.label_nodes,
        args.multipv,
        args.selected_contexts,
    )
    if min(numeric) < 1:
        parser.error("all numeric settings must be positive")
    if args.opening_plies >= args.maximum_plies:
        parser.error("opening plies must be less than maximum plies")
    discovered = shutil.which("stockfish") if args.engine is None else str(args.engine)
    if discovered is None:
        parser.error("Stockfish is required for development labels; pass --engine")
    engine_path = Path(discovered).resolve()
    champion_root = args.champion_root.resolve()
    rng = random.Random(args.seed)
    games: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    openings: list[chess.Board] = []
    with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
        engine.configure({"Threads": 1, "Hash": 128, "UCI_LimitStrength": False})
        for _ in range(args.games):
            openings.append(opening_board(engine, rng, args.opening_nodes, args.opening_plies))
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": args.opponent_elo})
        for game_id, opening in enumerate(openings):
            game, sampled = play_game(
                engine,
                champion_root,
                opening,
                game_id,
                chess.WHITE if game_id % 2 == 0 else chess.BLACK,
                args.base_ms,
                args.increment_ms,
                args.maximum_plies,
                args.sample_stride,
            )
            games.append(game)
            contexts.extend(sampled)
            if args.progress_every and (game_id + 1) % args.progress_every == 0:
                print(
                    f"played {game_id + 1}/{args.games}: score={game['champion_score']}; "
                    f"sampled={len(contexts)}",
                    flush=True,
                )
        if any(game["failure"] for game in games):
            failures = [game["failure"] for game in games if game["failure"]]
            raise RuntimeError(f"champion reliability failure during mining: {failures}")
        engine.configure({"UCI_LimitStrength": False})
        annotated: list[dict[str, Any]] = []
        for index, context in enumerate(contexts, 1):
            annotated.append(annotate_context(context, engine, args.label_nodes, args.multipv))
            if args.progress_every and index % max(1, args.progress_every * 16) == 0:
                print(f"deep-labelled {index}/{len(contexts)} decisions", flush=True)
    selected = select_contexts(annotated, min(args.selected_contexts, len(annotated)), rng)
    rows = rows_from_contexts(selected)
    digest = dataset_digest(rows)
    scores = [float(game["champion_score"]) for game in games]
    regrets = [float(context["regret"]) for context in selected]
    payload = {
        "schema_version": 1,
        "kind": "champion_disagreement_active_learning_dataset",
        "trajectory_source": "real_clock_match_against_strength_limited_stockfish",
        "seed": args.seed,
        "games_count": len(games),
        "opponent_elo": args.opponent_elo,
        "opening_plies": args.opening_plies,
        "opening_nodes": args.opening_nodes,
        "base_ms": args.base_ms,
        "increment_ms": args.increment_ms,
        "maximum_plies": args.maximum_plies,
        "sample_stride": args.sample_stride,
        "label_nodes": args.label_nodes,
        "multipv": args.multipv,
        "sampled_contexts": len(contexts),
        "selected_contexts": len(selected),
        "rows_count": len(rows),
        "match_score": sum(scores) / len(scores),
        "mean_selected_regret": sum(regrets) / len(regrets),
        "teacher_name": "Stockfish 18",
        "teacher_binary_sha256": file_sha256(engine_path),
        "script_sha256": file_sha256(Path(__file__)),
        "champion_agent_sha256": file_sha256(champion_root / "agent.py"),
        "champion_model_sha256": file_sha256(champion_root / "weights/model.json"),
        "protected_opening_list_used": False,
        "game_grouped": True,
        "dataset_sha256": digest,
        "games": games,
        "annotated_contexts": selected,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "games": len(games),
                "match_score": payload["match_score"],
                "sampled_contexts": len(contexts),
                "selected_contexts": len(selected),
                "rows": len(rows),
                "mean_selected_regret": payload["mean_selected_regret"],
                "dataset_sha256": digest,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
