#!/usr/bin/env python3
"""Frozen-opening, paired-colour arena built around the unmodified harness Agent."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import chess  # noqa: E402
import chess.pgn  # noqa: E402
from artifact import build_deterministic  # noqa: E402

from harness.package import DEFAULT_INCLUDES  # noqa: E402
from harness.referee import FAILED_TERMINATIONS, PIECE_VALUES  # noqa: E402
from harness.rules import INIT_BUDGET_S  # noqa: E402
from harness.sandbox import Agent, AgentFailure, local  # noqa: E402

Result = Literal["white", "black", "draw", "void"]
RESULT_HEADERS = {"white": "1-0", "black": "0-1", "draw": "1/2-1/2", "void": "*"}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def opening_fen(moves: list[str]) -> str:
    board = chess.Board()
    for uci in moves:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"illegal frozen opening move {uci} after {board.fen()}")
        board.push(move)
    return board.fen()


def adjudicate(board: chess.Board) -> Result:
    balance = sum(
        value * (len(board.pieces(piece, chess.WHITE)) - len(board.pieces(piece, chess.BLACK)))
        for piece, value in PIECE_VALUES.items()
    )
    if balance > 0:
        return "white"
    if balance < 0:
        return "black"
    return "draw"


def opponent_wins(mover: chess.Color) -> Result:
    return "black" if mover == chess.WHITE else "white"


def outcome_from_board(board: chess.Board) -> Result:
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return "draw"
    return "white" if outcome.winner == chess.WHITE else "black"


def wilson_score_interval(
    wins: int, draws: int, losses: int, z: float
) -> tuple[float, float]:
    """Wilson interval after representing each draw as one half-point trial."""
    games = wins + draws + losses
    if games == 0:
        return 0.0, 1.0
    trials = 2 * games
    successes = 2 * wins + draws
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, centre - radius), min(1.0, centre + radius)


def statistical_decision(
    wins: int, draws: int, losses: int, settings: dict[str, Any]
) -> tuple[str, float, float]:
    lower, upper = wilson_score_interval(
        wins, draws, losses, float(settings["confidence_z"])
    )
    games = wins + draws + losses
    if games < int(settings["minimum_games"]):
        return "continue", lower, upper
    null_score = float(settings["null_score"])
    if lower > null_score:
        return "accept", lower, upper
    if upper < null_score:
        return "reject", lower, upper
    if games >= int(settings["games"]):
        return "inconclusive", lower, upper
    return "continue", lower, upper


def render_pgn(board: chess.Board, result: Result, termination: str, opening_id: str) -> str:
    game = chess.pgn.Game.from_board(board)
    game.headers["Result"] = RESULT_HEADERS[result]
    game.headers["Termination"] = termination
    game.headers["Opening"] = opening_id
    return str(game)


def play_from_fen(
    white: Agent,
    black: Agent,
    fen: str,
    opening_id: str,
    base_ms: int,
    increment_ms: int,
    ply_cap: int,
) -> dict[str, Any]:
    board = chess.Board(fen)
    agents = {chess.WHITE: white, chess.BLACK: black}
    clock = {chess.WHITE: float(base_ms), chess.BLACK: float(base_ms)}
    result: Result = "void"
    termination = "both_failed"
    failed_colour: str | None = None
    try:
        failures: dict[chess.Color, str] = {}
        for colour, agent in agents.items():
            try:
                agent.start(INIT_BUDGET_S)
            except AgentFailure as failure:
                failures[colour] = failure.reason
        if len(failures) == 2:
            result, termination = "void", "both_failed"
            failed_colour = "both"
        elif chess.WHITE in failures:
            result, termination = "black", failures[chess.WHITE]
            failed_colour = "white"
        elif chess.BLACK in failures:
            result, termination = "white", failures[chess.BLACK]
            failed_colour = "black"
        else:
            while True:
                finish = board.outcome(claim_draw=True)
                if finish is not None:
                    result = outcome_from_board(board)
                    termination = finish.termination.name.lower()
                    break
                if len(board.move_stack) >= ply_cap:
                    result, termination = adjudicate(board), "adjudication"
                    break
                mover = board.turn
                started = time.monotonic()
                try:
                    uci = agents[mover].move(board.fen(), int(clock[mover]))
                except AgentFailure as failure:
                    result, termination = opponent_wins(mover), failure.reason
                    failed_colour = "white" if mover == chess.WHITE else "black"
                    break
                clock[mover] -= (time.monotonic() - started) * 1000.0
                if clock[mover] < 0:
                    result, termination = opponent_wins(mover), "flag"
                    failed_colour = "white" if mover == chess.WHITE else "black"
                    break
                try:
                    move = chess.Move.from_uci(uci)
                except chess.InvalidMoveError:
                    result, termination = opponent_wins(mover), "illegal"
                    failed_colour = "white" if mover == chess.WHITE else "black"
                    break
                if move not in board.legal_moves:
                    result, termination = opponent_wins(mover), "illegal"
                    failed_colour = "white" if mover == chess.WHITE else "black"
                    break
                board.push(move)
                clock[mover] += increment_ms
    finally:
        white.stop()
        black.stop()
    return {
        "result": result,
        "termination": termination,
        "failed_colour": failed_colour,
        "pgn": render_pgn(board, result, termination, opening_id),
    }


def extract_submission(source: Path, destination: Path) -> None:
    archive = destination.with_suffix(".zip")
    build_deterministic(source, archive, DEFAULT_INCLUDES)
    destination.mkdir()
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    policy = load(args.policy)
    settings = policy["arena"]
    openings = load(ROOT / settings["openings_file"])["openings"]
    games: list[dict[str, Any]] = []
    wins = draws = losses = 0
    failures: list[str] = []
    incumbent_failures: list[str] = []
    decision = "continue"
    lower, upper = 0.0, 1.0
    with tempfile.TemporaryDirectory(prefix="chessathon-arena-") as temporary:
        temp = Path(temporary)
        candidate = temp / "candidate"
        champion = temp / "champion"
        extract_submission(args.candidate.resolve(), candidate)
        extract_submission(args.champion.resolve(), champion)
        for opening in openings:
            fen = opening_fen(opening["moves"])
            for candidate_white in (True, False):
                white, black = (
                    (candidate, champion) if candidate_white else (champion, candidate)
                )
                game = play_from_fen(
                    local(white),
                    local(black),
                    fen,
                    opening["id"],
                    settings["base_ms"],
                    settings["increment_ms"],
                    settings["ply_cap"],
                )
                game["candidate_colour"] = "white" if candidate_white else "black"
                games.append(game)
                if game["termination"] in FAILED_TERMINATIONS:
                    failure = f"{opening['id']}:{game['candidate_colour']}:{game['termination']}"
                    if game["failed_colour"] in (game["candidate_colour"], "both"):
                        failures.append(failure)
                    else:
                        incumbent_failures.append(failure)
                if game["result"] in ("draw", "void"):
                    draws += 1
                elif (game["result"] == "white") == candidate_white:
                    wins += 1
                else:
                    losses += 1
            completed = len(games)
            if completed % int(settings["batch_games"]) == 0:
                decision, lower, upper = statistical_decision(wins, draws, losses, settings)
                if decision != "continue" or failures or incumbent_failures:
                    break
        if decision == "continue":
            decision, lower, upper = statistical_decision(wins, draws, losses, settings)
    score = (wins + draws / 2.0) / len(games)
    result = {
        "schema_version": 1,
        "passed": not failures and not incumbent_failures,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "failures": failures,
        "incumbent_failures": incumbent_failures,
        "statistical_decision": decision,
        "confidence_interval": {"lower": lower, "upper": upper},
        "games": games,
        "settings": settings,
        "duration_seconds": round(time.monotonic() - started, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary = {key: result[key] for key in ("passed", "wins", "draws", "losses", "score")}
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
