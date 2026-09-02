#!/usr/bin/env python3
"""Combine independent game-grouped teacher datasets without ID leakage."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.generate_active_learning_dataset import dataset_digest  # noqa: E402
from training.train_active_residual_evaluator import load_active_dataset  # noqa: E402
from training.train_stockfish_evaluator import file_sha256  # noqa: E402


def shifted(items: list[dict[str, Any]], offset: int) -> list[dict[str, Any]]:
    result = deepcopy(items)
    for item in result:
        item["game_id"] = int(item["game_id"]) + offset
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.dataset) < 2:
        parser.error("at least two datasets are required")
    rows: list[dict[str, Any]] = []
    games: list[dict[str, Any]] = []
    annotated: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    game_offset = 0
    last_metadata: dict[str, Any] | None = None
    teacher_hash: str | None = None
    for path in args.dataset:
        source_rows, metadata = load_active_dataset(path)
        current_hash = str(metadata["teacher_binary_sha256"])
        if teacher_hash is not None and current_hash != teacher_hash:
            raise ValueError("datasets were labelled by different teacher binaries")
        teacher_hash = current_hash
        source_game_ids = sorted({int(row["game_id"]) for row in source_rows})
        rows.extend(shifted(source_rows, game_offset))
        games.extend(shifted(list(metadata.get("games", [])), game_offset))
        annotated.extend(shifted(list(metadata.get("annotated_contexts", [])), game_offset))
        sources.append(
            {
                "path": str(path),
                "file_sha256": file_sha256(path),
                "dataset_sha256": metadata["dataset_sha256"],
                "game_ids": len(source_game_ids),
                "rows": len(source_rows),
                "champion_agent_sha256": metadata["champion_agent_sha256"],
                "champion_model_sha256": metadata["champion_model_sha256"],
            }
        )
        game_offset += max(source_game_ids) + 1
        last_metadata = metadata
    if last_metadata is None:
        raise RuntimeError("no dataset metadata was loaded")
    digest = dataset_digest(rows)
    payload = {
        "schema_version": 1,
        "kind": "champion_disagreement_active_learning_dataset",
        "trajectory_source": "rehearsal_mix_of_independent_teacher_datasets",
        "protected_opening_list_used": False,
        "game_grouped": True,
        "rows_count": len(rows),
        "games_count": game_offset,
        "dataset_sha256": digest,
        "teacher_name": last_metadata["teacher_name"],
        "teacher_binary_sha256": teacher_hash,
        "champion_agent_sha256": last_metadata["champion_agent_sha256"],
        "champion_model_sha256": last_metadata["champion_model_sha256"],
        "sources": sources,
        "games": games,
        "annotated_contexts": annotated,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "sources": len(sources),
                "games": game_offset,
                "rows": len(rows),
                "dataset_sha256": digest,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
