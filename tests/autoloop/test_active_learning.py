import json
import random
from pathlib import Path

import chess
import numpy as np
import pytest

from training.generate_active_learning_dataset import (
    ACTIVE_SEED,
    dataset_digest,
    rows_from_contexts,
    select_contexts,
)
from training.train_active_move_ordering import (
    CAPTURE_OFFSET,
    move_between,
    move_features,
)
from training.train_active_move_ordering import (
    FEATURES as ORDERING_FEATURES,
)
from training.train_active_residual_evaluator import (
    STRATEGIC_BOUNDS,
    STRATEGIC_FEATURE_NAMES,
    fit_residual,
    load_active_dataset,
    pairwise_samples,
    split_game_ids,
    strategic_features,
)


def test_active_selection_mixes_high_regret_and_reproducible_exploration() -> None:
    contexts = [
        {"game_id": index % 5, "ply": index, "regret": float(index)}
        for index in range(20)
    ]
    first = select_contexts(contexts, 12, random.Random(ACTIVE_SEED))
    second = select_contexts(contexts, 12, random.Random(ACTIVE_SEED))
    assert first == second
    selected_regrets = {item["regret"] for item in first}
    assert 19.0 in selected_regrets
    assert any(regret < 16.0 for regret in selected_regrets)
    assert {item["game_id"] for item in first} == set(range(5))


def test_active_rows_keep_parent_and_children_in_one_game_group() -> None:
    context = {
        "game_id": 7,
        "ply": 12,
        "fen": chess.STARTING_FEN,
        "champion_move": "a2a3",
        "best_score": 30.0,
        "champion_score": -20.0,
        "regret": 50.0,
        "teacher_lines": [
            {"move": "e2e4", "score": 30.0},
            {"move": "d2d4", "score": 20.0},
        ],
    }
    rows = rows_from_contexts([context])
    assert len(rows) == 4
    assert {row["game_id"] for row in rows} == {7}
    assert {row["source"] for row in rows} == {
        "parent",
        "teacher_child_1",
        "teacher_child_2",
        "champion_child",
    }


def test_strategic_features_reverse_with_turn_and_have_declared_shape() -> None:
    white = chess.Board("4k3/pp6/8/3P4/8/2B5/PP6/4K3 w - - 0 1")
    black = chess.Board("4k3/pp6/8/3P4/8/2B5/PP6/4K3 b - - 0 1")
    assert strategic_features(white).shape == (len(STRATEGIC_FEATURE_NAMES),)
    assert np.array_equal(strategic_features(white), -strategic_features(black))


def test_game_split_is_disjoint_and_deterministic() -> None:
    rows = [{"game_id": game_id} for game_id in range(10) for _ in range(3)]
    first_training, first_validation = split_game_ids(rows)
    second_training, second_validation = split_game_ids(rows)
    assert (first_training, first_validation) == (second_training, second_validation)
    assert first_training.isdisjoint(first_validation)
    assert first_training | first_validation == set(range(10))
    assert len(first_validation) == 2


def test_active_cache_rejects_digest_tampering(tmp_path: Path) -> None:
    rows = [
        {
            "game_id": game_id,
            "parent_ply": 8,
            "source": "parent",
            "teacher_rank": None,
            "regret": 10.0,
            "fen": chess.STARTING_FEN,
            "label": float(game_id),
        }
        for game_id in range(5)
    ]
    payload = {
        "schema_version": 1,
        "kind": "champion_disagreement_active_learning_dataset",
        "protected_opening_list_used": False,
        "game_grouped": True,
        "rows_count": len(rows),
        "dataset_sha256": dataset_digest(rows),
        "rows": rows,
    }
    path = tmp_path / "active.json"
    path.write_text(json.dumps(payload))
    loaded, _ = load_active_dataset(path)
    assert loaded == rows
    payload["rows"][0]["label"] = 999.0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest mismatch"):
        load_active_dataset(path)


def test_residual_fit_respects_chess_valid_bounds() -> None:
    design = np.ones((20, len(STRATEGIC_FEATURE_NAMES)))
    labels = np.full(20, 50_000.0)
    coefficients = fit_residual(design, labels, 1.0)
    for coefficient, (lower, upper) in zip(coefficients, STRATEGIC_BOUNDS, strict=True):
        assert lower <= coefficient <= upper


def test_pairwise_samples_target_alternative_minus_best_margin() -> None:
    rows = [
        {"game_id": 3, "parent_ply": 12, "source": "teacher_child_1"},
        {"game_id": 3, "parent_ply": 12, "source": "teacher_child_2"},
        {"game_id": 3, "parent_ply": 12, "source": "champion_child"},
    ]
    design = np.asarray([[1.0, 0.0], [3.0, 1.0], [5.0, 2.0]])
    labels = np.asarray([-40.0, -10.0, 20.0])
    baseline = np.asarray([-20.0, -15.0, -5.0])
    pair_design, residual, games, targets, baseline_margins = pairwise_samples(
        rows, design, labels, baseline
    )
    assert pair_design.tolist() == [[2.0, 1.0], [4.0, 2.0]]
    assert targets.tolist() == [30.0, 60.0]
    assert baseline_margins.tolist() == [5.0, 15.0]
    assert residual.tolist() == [25.0, 45.0]
    assert games.tolist() == [3, 3]


def test_move_ordering_recovers_child_move_and_fixed_features() -> None:
    parent = chess.Board()
    child = parent.copy(stack=False)
    child.push_uci("e2e4")
    move = move_between(parent.fen(), child.fen())
    vector = move_features(parent, move)
    assert move.uci() == "e2e4"
    assert vector.shape == (ORDERING_FEATURES,)
    assert vector.sum() == 2.0


def test_move_ordering_marks_captures() -> None:
    board = chess.Board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")
    vector = move_features(board, chess.Move.from_uci("e4d5"))
    assert vector[CAPTURE_OFFSET] == 1.0
