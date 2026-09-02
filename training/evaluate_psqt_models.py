#!/usr/bin/env python3
"""Compare fixed tapered-PSQT models on independent grouped teacher datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import chess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_active_psqt_finetune import ranking_totals  # noqa: E402
from training.train_active_residual_evaluator import (  # noqa: E402
    baseline_prediction,
    load_active_dataset,
)
from training.train_stockfish_evaluator import root_mean_square_error  # noqa: E402


def metrics(model_path: Path, dataset_path: Path) -> dict[str, Any]:
    model = json.loads(model_path.read_text())
    rows, metadata = load_active_dataset(dataset_path)
    positions = [chess.Board(row["fen"]) for row in rows]
    clip = float(model.get("training", {}).get("label_clip_centipawns", 1500.0))
    labels = np.clip(np.asarray([float(row["label"]) for row in rows]), -clip, clip)
    prediction = baseline_prediction(positions, model)
    selected = np.ones(len(rows), dtype=np.bool_)
    groups, top_one, reciprocal_rank = ranking_totals(rows, labels, prediction, selected)
    return {
        "model": str(model_path),
        "dataset": str(dataset_path),
        "dataset_sha256": metadata["dataset_sha256"],
        "examples": len(rows),
        "groups": groups,
        "rmse": root_mean_square_error(labels, prediction),
        "top1": top_one / groups,
        "mrr": reciprocal_rank / groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    args = parser.parse_args()
    results = [
        metrics(model_path, dataset_path)
        for dataset_path in args.dataset
        for model_path in args.model
    ]
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
