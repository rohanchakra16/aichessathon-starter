#!/usr/bin/env python3
"""Train a tapered evaluator on deterministic engine-guided self-play.

Unlike the earlier random-position trainer, this generator begins at the normal
initial position and samples only among the teacher's top moves. It does not
read the protected benchmark opening list. The engine remains development-only.
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
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_stockfish_evaluator import (  # noqa: E402
    FEATURES,
    MAX_PHASE,
    SQUARE_FEATURES,
    coefficient_prior,
    file_sha256,
    is_quiet,
    train,
)

SELFPLAY_SEED = 2026083101
SELECTION_WEIGHTS = (0.72, 0.20, 0.08)


def dataset_digest(positions: list[chess.Board], labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    for board, label in zip(positions, labels, strict=True):
        digest.update(f"{board.fen()}\t{label:.1f}\n".encode())
    return digest.hexdigest()


def choose_analysis(count: int, rng: random.Random) -> int:
    return rng.choices(range(count), weights=SELECTION_WEIGHTS[:count], k=1)[0]


def score_for(board: chess.Board, information: dict[str, Any]) -> float:
    score = information["score"].pov(board.turn).score(mate_score=10_000)
    if score is None:
        raise RuntimeError(f"teacher produced no score for {board.fen()}")
    return float(max(-2_000, min(2_000, score)))


def generate_dataset(
    examples: int,
    engine_path: Path,
    nodes: int,
    maximum_plies: int,
    progress_every: int,
) -> tuple[list[chess.Board], np.ndarray, int]:
    rng = random.Random(SELFPLAY_SEED)
    positions: list[chess.Board] = []
    labels: list[float] = []
    seen: set[str] = set()
    games = 0
    with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
        engine.configure({"Threads": 1, "Hash": 64})
        while len(positions) < examples:
            games += 1
            board = chess.Board()
            for ply in range(maximum_plies):
                if board.is_game_over(claim_draw=True):
                    break
                count = min(len(SELECTION_WEIGHTS), board.legal_moves.count())
                analyses = engine.analyse(
                    board,
                    chess.engine.Limit(nodes=nodes),
                    multipv=count,
                )
                if not isinstance(analyses, list):
                    analyses = [analyses]
                usable = [information for information in analyses if information.get("pv")]
                if not usable:
                    break
                if ply >= 8 and is_quiet(board):
                    fen = board.fen()
                    if fen not in seen:
                        seen.add(fen)
                        positions.append(board.copy(stack=False))
                        labels.append(score_for(board, usable[0]))
                        if progress_every and len(positions) % progress_every == 0:
                            print(
                                f"generated {len(positions)}/{examples} positions "
                                f"from {games} games",
                                flush=True,
                            )
                        if len(positions) == examples:
                            break
                choice = choose_analysis(len(usable), rng)
                board.push(usable[choice]["pv"][0])
    return positions, np.asarray(labels, dtype=np.float64), games


def save_dataset(
    path: Path,
    positions: list[chess.Board],
    labels: np.ndarray,
    engine_path: Path,
    nodes: int,
    maximum_plies: int,
    games: int,
) -> str:
    digest = dataset_digest(positions, labels)
    payload = {
        "schema_version": 1,
        "kind": "engine_guided_selfplay_evaluation_dataset",
        "seed": SELFPLAY_SEED,
        "nodes_per_position": nodes,
        "maximum_plies": maximum_plies,
        "games": games,
        "selection_weights": list(SELECTION_WEIGHTS),
        "teacher_binary_sha256": file_sha256(engine_path),
        "dataset_sha256": digest,
        "protected_opening_list_used": False,
        "rows": [
            {"fen": board.fen(), "label": float(label)}
            for board, label in zip(positions, labels, strict=True)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return digest


def load_dataset(
    path: Path,
    engine_path: Path,
    examples: int,
    nodes: int,
    maximum_plies: int,
) -> tuple[list[chess.Board], np.ndarray, str, int]:
    payload = json.loads(path.read_text())
    expected = {
        "schema_version": 1,
        "kind": "engine_guided_selfplay_evaluation_dataset",
        "seed": SELFPLAY_SEED,
        "nodes_per_position": nodes,
        "maximum_plies": maximum_plies,
        "selection_weights": list(SELECTION_WEIGHTS),
        "teacher_binary_sha256": file_sha256(engine_path),
        "protected_opening_list_used": False,
    }
    actual = {key: payload.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"dataset cache metadata mismatch: {actual!r} != {expected!r}")
    rows = payload.get("rows", [])
    if len(rows) != examples:
        raise ValueError(f"dataset cache has {len(rows)} rows; expected {examples}")
    positions = [chess.Board(row["fen"]) for row in rows]
    labels = np.asarray([float(row["label"]) for row in rows], dtype=np.float64)
    digest = dataset_digest(positions, labels)
    if digest != payload.get("dataset_sha256"):
        raise ValueError("dataset cache digest mismatch")
    return positions, labels, digest, int(payload["games"])


def model_payload(
    coefficients: np.ndarray,
    metrics: dict[str, float | int],
    args: argparse.Namespace,
    engine_path: Path,
    digest: str,
    games: int,
) -> dict[str, Any]:
    script = Path(__file__)
    return {
        "schema_version": 4,
        "model_kind": "selfplay_distilled_tapered_piece_square_evaluator",
        "materially_drives": "all non-terminal search leaf evaluations",
        "layout": {
            "midgame_piece_square": [0, SQUARE_FEATURES],
            "endgame_piece_square": [SQUARE_FEATURES, SQUARE_FEATURES * 2],
            "castling": [SQUARE_FEATURES * 2, FEATURES],
            "relative_square_orientation": "white=a1..h8; black vertically mirrored",
            "phase_max": MAX_PHASE,
        },
        "training": {
            "method": "deterministic engine-guided self-play evaluation distillation",
            "seed": SELFPLAY_SEED,
            "examples": args.examples,
            "games": games,
            "nodes_per_position": args.nodes,
            "maximum_plies": args.maximum_plies,
            "selection_weights": list(SELECTION_WEIGHTS),
            "label_clip_centipawns": args.label_clip,
            "ridge_penalty": args.ridge_penalty,
            "coefficient_prior": coefficient_prior().tolist(),
            "script": str(script.relative_to(Path.cwd())),
            "script_sha256": file_sha256(script),
            "external_engine_used": True,
            "teacher_name": "Stockfish 18",
            "teacher_binary_sha256": file_sha256(engine_path),
            "dataset_sha256": digest,
            "protected_opening_list_used": False,
        },
        "bias": float(coefficients[0]),
        "weights": [float(value) for value in coefficients[1:]],
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=8000)
    parser.add_argument("--nodes", type=int, default=1500)
    parser.add_argument("--maximum-plies", type=int, default=120)
    parser.add_argument("--label-clip", type=float, default=1500.0)
    parser.add_argument("--ridge-penalty", type=float, default=500.0)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--dataset-cache", type=Path)
    parser.add_argument("--output", type=Path, default=Path("weights/model.json"))
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()
    if (
        args.examples < 10
        or args.nodes < 1
        or args.maximum_plies < 10
        or args.label_clip <= 0
    ):
        parser.error("examples, nodes, maximum plies, or label clip are invalid")
    discovered = shutil.which("stockfish") if args.engine is None else str(args.engine)
    if discovered is None:
        parser.error("Stockfish is required for offline labels; pass --engine")
    engine_path = Path(discovered).resolve()
    if args.dataset_cache is not None and args.dataset_cache.exists():
        positions, labels, digest, games = load_dataset(
            args.dataset_cache,
            engine_path,
            args.examples,
            args.nodes,
            args.maximum_plies,
        )
    else:
        positions, labels, games = generate_dataset(
            args.examples,
            engine_path,
            args.nodes,
            args.maximum_plies,
            args.progress_every,
        )
        digest = dataset_digest(positions, labels)
        if args.dataset_cache is not None:
            digest = save_dataset(
                args.dataset_cache,
                positions,
                labels,
                engine_path,
                args.nodes,
                args.maximum_plies,
                games,
            )
    training_labels = np.clip(labels, -args.label_clip, args.label_clip)
    coefficients, metrics = train(positions, training_labels, args.ridge_penalty)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            model_payload(coefficients, metrics, args, engine_path, digest, games),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps({**metrics, "games": games}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
