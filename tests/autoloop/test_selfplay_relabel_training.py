import argparse
import json
from pathlib import Path

import chess
import numpy as np
import pytest

from training.relabel_selfplay_evaluator import (
    load_relabelled_dataset,
    load_source_dataset,
    model_payload,
    relabelled_payload,
)
from training.train_selfplay_evaluator import dataset_digest
from training.train_stockfish_evaluator import FEATURES, file_sha256


def source_payload(engine: Path) -> dict[str, object]:
    positions = [chess.Board(), chess.Board("8/8/8/8/8/8/4K3/6k1 w - - 0 1")]
    labels = np.asarray([12.0, 0.0])
    return {
        "schema_version": 1,
        "kind": "engine_guided_selfplay_evaluation_dataset",
        "seed": 7,
        "nodes_per_position": 1500,
        "maximum_plies": 120,
        "games": 2,
        "selection_weights": [0.72, 0.2, 0.08],
        "teacher_binary_sha256": file_sha256(engine),
        "dataset_sha256": dataset_digest(positions, labels),
        "protected_opening_list_used": False,
        "rows": [
            {"fen": board.fen(), "label": float(label)}
            for board, label in zip(positions, labels, strict=True)
        ],
    }


def test_relabel_cache_preserves_positions_and_provenance(tmp_path: Path) -> None:
    engine = tmp_path / "teacher"
    engine.write_bytes(b"fixed teacher")
    source_path = tmp_path / "source.json"
    source = source_payload(engine)
    source_path.write_text(json.dumps(source))
    positions, _, loaded_source = load_source_dataset(source_path, engine)
    labels = np.asarray([25.0, -5.0])
    payload = relabelled_payload(positions, labels, loaded_source, engine, 5000)
    cache = tmp_path / "labels.json"
    cache.write_text(json.dumps(payload))

    loaded_labels, digest = load_relabelled_dataset(
        cache, positions, loaded_source, engine, 5000
    )

    assert loaded_labels.tolist() == labels.tolist()
    assert digest == dataset_digest(positions, labels)
    assert payload["source_dataset_sha256"] == source["dataset_sha256"]


def test_relabel_cache_rejects_tampering(tmp_path: Path) -> None:
    engine = tmp_path / "teacher"
    engine.write_bytes(b"fixed teacher")
    source_path = tmp_path / "source.json"
    source = source_payload(engine)
    source_path.write_text(json.dumps(source))
    positions, _, loaded_source = load_source_dataset(source_path, engine)
    payload = relabelled_payload(
        positions, np.asarray([25.0, -5.0]), loaded_source, engine, 5000
    )
    payload["rows"][0]["label"] = 999.0  # type: ignore[index]
    cache = tmp_path / "labels.json"
    cache.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="digest mismatch"):
        load_relabelled_dataset(cache, positions, loaded_source, engine, 5000)


def test_model_records_both_generation_and_label_budgets(tmp_path: Path) -> None:
    engine = tmp_path / "teacher"
    engine.write_bytes(b"fixed teacher")
    source = source_payload(engine)
    args = argparse.Namespace(label_nodes=5000, label_clip=1500.0, ridge_penalty=100.0)
    coefficients = np.zeros(FEATURES + 1)
    payload = model_payload(
        coefficients,
        {"training_rmse": 1.0, "validation_rmse": 2.0},
        args,
        engine,
        source,
        "new-digest",
    )

    assert payload["training"]["generation_nodes_per_position"] == 1500
    assert payload["training"]["label_nodes_per_position"] == 5000
    assert payload["training"]["source_dataset_sha256"] == source["dataset_sha256"]
