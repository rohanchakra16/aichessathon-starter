#!/usr/bin/env python3
"""Build a game-grouped master-position corpus and label it offline.

Human games provide realistic and diverse trajectories, but their moves and
results are not used as training targets.  A development-only teacher labels
the selected positions at a fixed node budget.  Neither the PGN nor the
teacher is packaged with the competition agent.
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
import chess.pgn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_selfplay_evaluator import dataset_digest  # noqa: E402
from training.train_stockfish_evaluator import file_sha256  # noqa: E402

SEED = 2026090104
DEFAULT_SOURCE_URL = "https://theweekinchess.com/zips/twic1660g.zip"


def position_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (
                f"{row['game_id']}\t{row['ply']}\t{row['source']}\t"
                f"{row['fen']}\n"
            ).encode()
        )
    return digest.hexdigest()


def _position_source(board: chess.Board) -> str:
    if board.is_check() or any(
        board.is_capture(move) or move.promotion or board.gives_check(move)
        for move in board.legal_moves
    ):
        return "historical_tactical"
    return "historical_quiet"


def game_positions(
    game: chess.pgn.Game,
    game_id: int,
    rng: random.Random,
    maximum_ply: int,
    positions_per_game: int,
) -> list[dict[str, Any]]:
    board = game.board()
    candidates: list[dict[str, Any]] = []
    for move in game.mainline_moves():
        if move not in board.legal_moves:
            return []
        board.push(move)
        ply = board.ply()
        if ply < 8 or ply > maximum_ply or board.is_game_over(claim_draw=True):
            continue
        candidates.append(
            {
                "game_id": game_id,
                "ply": ply,
                "source": _position_source(board),
                "fen": board.fen(),
                "event": game.headers.get("Event", "?"),
                "result": game.headers.get("Result", "*"),
            }
        )
    if not candidates:
        return []

    selected: list[dict[str, Any]] = []
    tactical = [row for row in candidates if row["source"] == "historical_tactical"]
    quiet = [row for row in candidates if row["source"] == "historical_quiet"]
    for group in (tactical, quiet):
        if group and len(selected) < positions_per_game:
            selected.append(rng.choice(group))
    remaining = [row for row in candidates if row not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: positions_per_game - len(selected)])
    return sorted(selected, key=lambda row: int(row["ply"]))


def generate_positions(
    pgn_path: Path,
    examples: int,
    maximum_ply: int,
    positions_per_game: int,
) -> tuple[list[dict[str, Any]], int]:
    rng = random.Random(SEED)
    rows: list[dict[str, Any]] = []
    source_games = 0
    with pgn_path.open(errors="replace") as stream:
        while len(rows) < examples:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            source_games += 1
            if game.headers.get("Variant", "Standard") not in {"Standard", "?"}:
                continue
            chosen = game_positions(
                game,
                source_games - 1,
                rng,
                maximum_ply,
                positions_per_game,
            )
            rows.extend(chosen[: examples - len(rows)])
    if len(rows) != examples:
        raise ValueError(f"PGN produced {len(rows)} positions; expected {examples}")
    if len({int(row["game_id"]) for row in rows}) < examples // positions_per_game // 2:
        raise ValueError("historical corpus has insufficient game diversity")
    return rows, source_games


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def position_payload(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    source_games: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "historical_master_position_corpus",
        "seed": SEED,
        "source_url": args.source_url,
        "source_archive_sha256": args.archive_sha256,
        "source_pgn_sha256": file_sha256(args.pgn),
        "source_games_read": source_games,
        "examples": len(rows),
        "maximum_ply": args.maximum_ply,
        "positions_per_game": args.positions_per_game,
        "position_sha256": position_digest(rows),
        "protected_opening_list_used": False,
        "rows": rows,
    }


def load_or_generate_positions(args: argparse.Namespace) -> dict[str, Any]:
    if args.positions_cache.exists():
        payload = json.loads(args.positions_cache.read_text())
        required = {
            "schema_version": 1,
            "kind": "historical_master_position_corpus",
            "seed": SEED,
            "source_url": args.source_url,
            "source_archive_sha256": args.archive_sha256,
            "source_pgn_sha256": file_sha256(args.pgn),
            "examples": args.examples,
            "maximum_ply": args.maximum_ply,
            "positions_per_game": args.positions_per_game,
            "protected_opening_list_used": False,
        }
        actual = {key: payload.get(key) for key in required}
        if actual != required:
            raise ValueError(f"historical position cache mismatch: {actual!r}")
        rows = payload.get("rows", [])
        if position_digest(rows) != payload.get("position_sha256"):
            raise ValueError("historical position cache digest mismatch")
        return payload
    rows, source_games = generate_positions(
        args.pgn,
        args.examples,
        args.maximum_ply,
        args.positions_per_game,
    )
    payload = position_payload(rows, args, source_games)
    args.positions_cache.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.positions_cache, payload)
    return payload


def labelled_payload(
    positions: dict[str, Any],
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    engine_path: Path,
    complete: bool,
) -> dict[str, Any]:
    boards = [chess.Board(row["fen"]) for row in rows]
    labels = [float(row["label"]) for row in rows]
    return {
        "schema_version": 1,
        "kind": "historical_master_engine_labelled_dataset",
        "seed": SEED,
        "source_url": positions["source_url"],
        "source_archive_sha256": positions["source_archive_sha256"],
        "source_pgn_sha256": positions["source_pgn_sha256"],
        "source_games_read": positions["source_games_read"],
        "position_sha256": positions["position_sha256"],
        "examples": positions["examples"],
        "maximum_ply": positions["maximum_ply"],
        "positions_per_game": positions["positions_per_game"],
        "label_nodes_per_position": args.label_nodes,
        "teacher_name": "Stockfish 18",
        "teacher_binary_sha256": file_sha256(engine_path),
        "protected_opening_list_used": False,
        "complete": complete,
        "labelled_rows": len(rows),
        "dataset_sha256": dataset_digest(boards, labels) if complete else None,
        "rows": rows,
    }


def validate_label_prefix(
    cached: dict[str, Any],
    positions: dict[str, Any],
    args: argparse.Namespace,
    engine_path: Path,
) -> list[dict[str, Any]]:
    required = {
        "schema_version": 1,
        "kind": "historical_master_engine_labelled_dataset",
        "seed": SEED,
        "source_archive_sha256": positions["source_archive_sha256"],
        "source_pgn_sha256": positions["source_pgn_sha256"],
        "position_sha256": positions["position_sha256"],
        "examples": positions["examples"],
        "label_nodes_per_position": args.label_nodes,
        "teacher_binary_sha256": file_sha256(engine_path),
        "protected_opening_list_used": False,
    }
    actual = {key: cached.get(key) for key in required}
    if actual != required:
        raise ValueError(f"historical label cache mismatch: {actual!r}")
    rows = cached.get("rows", [])
    source_rows = positions["rows"]
    if len(rows) > len(source_rows):
        raise ValueError("historical label cache is longer than its position corpus")
    for labelled, source in zip(rows, source_rows, strict=False):
        if any(labelled.get(key) != source.get(key) for key in source):
            raise ValueError("historical label cache position prefix mismatch")
        if "label" not in labelled:
            raise ValueError("historical label cache row has no label")
    if cached.get("complete"):
        if len(rows) != len(source_rows):
            raise ValueError("complete historical label cache is truncated")
        boards = [chess.Board(row["fen"]) for row in rows]
        labels = [float(row["label"]) for row in rows]
        if dataset_digest(boards, labels) != cached.get("dataset_sha256"):
            raise ValueError("historical label cache digest mismatch")
    return rows


def label_positions(
    positions: dict[str, Any],
    args: argparse.Namespace,
    engine_path: Path,
) -> dict[str, Any]:
    labelled: list[dict[str, Any]] = []
    if args.output.exists():
        cached = json.loads(args.output.read_text())
        labelled = validate_label_prefix(cached, positions, args, engine_path)
        if cached.get("complete"):
            return cached
    source_rows = positions["rows"]
    with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
        engine.configure({"Threads": 1, "Hash": 64})
        for index in range(len(labelled), len(source_rows)):
            source = source_rows[index]
            board = chess.Board(source["fen"])
            information = engine.analyse(board, chess.engine.Limit(nodes=args.label_nodes))
            score = information["score"].pov(board.turn).score(mate_score=10_000)
            if score is None:
                raise RuntimeError(f"teacher produced no score for {board.fen()}")
            labelled.append({**source, "label": float(max(-2_000, min(2_000, score)))})
            completed = index + 1
            if args.progress_every and completed % args.progress_every == 0:
                atomic_json(
                    args.output,
                    labelled_payload(positions, labelled, args, engine_path, False),
                )
                print(f"labelled {completed}/{len(source_rows)} historical positions", flush=True)
    payload = labelled_payload(positions, labelled, args, engine_path, True)
    atomic_json(args.output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", type=Path, required=True)
    parser.add_argument("--positions-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--examples", type=int, default=6000)
    parser.add_argument("--maximum-ply", type=int, default=120)
    parser.add_argument("--positions-per-game", type=int, default=3)
    parser.add_argument("--label-nodes", type=int, default=5000)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()
    if args.examples < 10 or args.maximum_ply < 16 or args.positions_per_game < 1:
        parser.error("examples, maximum ply, or positions per game is invalid")
    if args.label_nodes < 1:
        parser.error("label nodes must be positive")
    discovered = shutil.which("stockfish") if args.engine is None else str(args.engine)
    if discovered is None:
        parser.error("Stockfish is required for offline labels; pass --engine")
    engine_path = Path(discovered).resolve()
    positions = load_or_generate_positions(args)
    payload = label_positions(positions, args, engine_path)
    print(
        json.dumps(
            {
                "examples": payload["examples"],
                "games": len({row["game_id"] for row in payload["rows"]}),
                "dataset_sha256": payload["dataset_sha256"],
                "complete": payload["complete"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
