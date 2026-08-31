#!/usr/bin/env python3
"""Relabel a verified self-play dataset with a higher teacher node budget.

The positions and their sampling recipe stay fixed. Only the development-only
teacher labels change, making this a controlled label-fidelity experiment.
Neither the teacher nor this script is included in candidate submissions.
"""

from __future__ import annotations

import argparse
import json
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

from training.train_selfplay_evaluator import dataset_digest  # noqa: E402
from training.train_stockfish_evaluator import (  # noqa: E402
    FEATURES,
    MAX_PHASE,
    SQUARE_FEATURES,
    coefficient_prior,
    file_sha256,
    train,
)


def load_source_dataset(
    path: Path, engine_path: Path
) -> tuple[list[chess.Board], np.ndarray, dict[str, Any]]:
    payload = json.loads(path.read_text())
    required = {
        "schema_version": 1,
        "kind": "engine_guided_selfplay_evaluation_dataset",
        "teacher_binary_sha256": file_sha256(engine_path),
        "protected_opening_list_used": False,
    }
    actual = {key: payload.get(key) for key in required}
    if actual != required:
        raise ValueError(f"source dataset metadata mismatch: {actual!r} != {required!r}")
    rows = payload.get("rows", [])
    if not rows:
        raise ValueError("source dataset has no rows")
    positions = [chess.Board(row["fen"]) for row in rows]
    labels = np.asarray([float(row["label"]) for row in rows], dtype=np.float64)
    digest = dataset_digest(positions, labels)
    if digest != payload.get("dataset_sha256"):
        raise ValueError("source dataset digest mismatch")
    return positions, labels, payload


def teacher_labels(
    positions: list[chess.Board],
    engine_path: Path,
    nodes: int,
    progress_every: int,
) -> np.ndarray:
    labels: list[float] = []
    with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
        engine.configure({"Threads": 1, "Hash": 64})
        for index, board in enumerate(positions, 1):
            information = engine.analyse(board, chess.engine.Limit(nodes=nodes))
            score = information["score"].pov(board.turn).score(mate_score=10_000)
            if score is None:
                raise RuntimeError(f"teacher produced no score for {board.fen()}")
            labels.append(float(max(-2_000, min(2_000, score))))
            if progress_every and index % progress_every == 0:
                print(f"relabelled {index}/{len(positions)}", flush=True)
    return np.asarray(labels, dtype=np.float64)


def relabelled_payload(
    positions: list[chess.Board],
    labels: np.ndarray,
    source: dict[str, Any],
    engine_path: Path,
    label_nodes: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "high_fidelity_relabelled_selfplay_evaluation_dataset",
        "seed": source["seed"],
        "generation_nodes_per_position": source["nodes_per_position"],
        "label_nodes_per_position": label_nodes,
        "maximum_plies": source["maximum_plies"],
        "games": source["games"],
        "selection_weights": source["selection_weights"],
        "teacher_binary_sha256": file_sha256(engine_path),
        "source_dataset_sha256": source["dataset_sha256"],
        "dataset_sha256": dataset_digest(positions, labels),
        "protected_opening_list_used": False,
        "rows": [
            {"fen": board.fen(), "label": float(label)}
            for board, label in zip(positions, labels, strict=True)
        ],
    }


def load_relabelled_dataset(
    path: Path,
    positions: list[chess.Board],
    source: dict[str, Any],
    engine_path: Path,
    label_nodes: int,
) -> tuple[np.ndarray, str]:
    payload = json.loads(path.read_text())
    required = {
        "schema_version": 1,
        "kind": "high_fidelity_relabelled_selfplay_evaluation_dataset",
        "seed": source["seed"],
        "generation_nodes_per_position": source["nodes_per_position"],
        "label_nodes_per_position": label_nodes,
        "maximum_plies": source["maximum_plies"],
        "games": source["games"],
        "selection_weights": source["selection_weights"],
        "teacher_binary_sha256": file_sha256(engine_path),
        "source_dataset_sha256": source["dataset_sha256"],
        "protected_opening_list_used": False,
    }
    actual = {key: payload.get(key) for key in required}
    if actual != required:
        raise ValueError(f"relabel cache metadata mismatch: {actual!r} != {required!r}")
    rows = payload.get("rows", [])
    if len(rows) != len(positions):
        raise ValueError(
            f"relabel cache has {len(rows)} rows; expected {len(positions)}"
        )
    cached_positions = [chess.Board(row["fen"]) for row in rows]
    if [board.fen() for board in cached_positions] != [board.fen() for board in positions]:
        raise ValueError("relabel cache positions differ from the source dataset")
    labels = np.asarray([float(row["label"]) for row in rows], dtype=np.float64)
    digest = dataset_digest(cached_positions, labels)
    if digest != payload.get("dataset_sha256"):
        raise ValueError("relabel cache digest mismatch")
    return labels, digest


def model_payload(
    coefficients: np.ndarray,
    metrics: dict[str, float | int],
    args: argparse.Namespace,
    engine_path: Path,
    source: dict[str, Any],
    digest: str,
) -> dict[str, Any]:
    script = Path(__file__)
    return {
        "schema_version": 5,
        "model_kind": "high_fidelity_selfplay_distilled_tapered_piece_square_evaluator",
        "materially_drives": "all non-terminal search leaf evaluations",
        "layout": {
            "midgame_piece_square": [0, SQUARE_FEATURES],
            "endgame_piece_square": [SQUARE_FEATURES, SQUARE_FEATURES * 2],
            "castling": [SQUARE_FEATURES * 2, FEATURES],
            "relative_square_orientation": "white=a1..h8; black vertically mirrored",
            "phase_max": MAX_PHASE,
        },
        "training": {
            "method": "fixed self-play positions relabelled at a higher teacher node budget",
            "seed": source["seed"],
            "examples": len(source["rows"]),
            "games": source["games"],
            "generation_nodes_per_position": source["nodes_per_position"],
            "label_nodes_per_position": args.label_nodes,
            "maximum_plies": source["maximum_plies"],
            "selection_weights": source["selection_weights"],
            "label_clip_centipawns": args.label_clip,
            "ridge_penalty": args.ridge_penalty,
            "coefficient_prior": coefficient_prior().tolist(),
            "script": str(script.relative_to(Path.cwd())),
            "script_sha256": file_sha256(script),
            "external_engine_used": True,
            "teacher_name": "Stockfish 18",
            "teacher_binary_sha256": file_sha256(engine_path),
            "source_dataset_sha256": source["dataset_sha256"],
            "dataset_sha256": digest,
            "protected_opening_list_used": False,
        },
        "bias": float(coefficients[0]),
        "weights": [float(value) for value in coefficients[1:]],
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--label-nodes", type=int, default=5000)
    parser.add_argument("--label-clip", type=float, default=1500.0)
    parser.add_argument("--ridge-penalty", type=float, default=100.0)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--label-cache", type=Path)
    parser.add_argument("--output", type=Path, default=Path("weights/model.json"))
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()
    if args.label_nodes < 1 or args.label_clip <= 0 or args.ridge_penalty < 0:
        parser.error("label nodes, label clip, or ridge penalty is invalid")
    discovered = shutil.which("stockfish") if args.engine is None else str(args.engine)
    if discovered is None:
        parser.error("Stockfish is required for offline labels; pass --engine")
    engine_path = Path(discovered).resolve()
    positions, _, source = load_source_dataset(args.source_dataset, engine_path)
    if args.label_cache is not None and args.label_cache.exists():
        labels, digest = load_relabelled_dataset(
            args.label_cache, positions, source, engine_path, args.label_nodes
        )
    else:
        labels = teacher_labels(
            positions, engine_path, args.label_nodes, args.progress_every
        )
        payload = relabelled_payload(
            positions, labels, source, engine_path, args.label_nodes
        )
        digest = str(payload["dataset_sha256"])
        if args.label_cache is not None:
            args.label_cache.parent.mkdir(parents=True, exist_ok=True)
            args.label_cache.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    training_labels = np.clip(labels, -args.label_clip, args.label_clip)
    coefficients, metrics = train(positions, training_labels, args.ridge_penalty)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            model_payload(coefficients, metrics, args, engine_path, source, digest),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
