import random
from pathlib import Path

import chess
import numpy as np

from training.train_selfplay_evaluator import (
    SELFPLAY_SEED,
    choose_analysis,
    dataset_digest,
    load_dataset,
    save_dataset,
)


def test_selfplay_choice_is_reproducible_and_explores_alternatives() -> None:
    first = random.Random(SELFPLAY_SEED)
    second = random.Random(SELFPLAY_SEED)
    choices = [choose_analysis(3, first) for _ in range(100)]
    assert choices == [choose_analysis(3, second) for _ in range(100)]
    assert set(choices) == {0, 1, 2}


def test_selfplay_cache_round_trip_records_independent_source(tmp_path: Path) -> None:
    engine = tmp_path / "teacher"
    engine.write_bytes(b"fixed teacher")
    positions = [chess.Board(), chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")]
    labels = np.asarray([12.0, 0.0])
    cache = tmp_path / "selfplay.json"
    digest = save_dataset(cache, positions, labels, engine, 1500, 120, 2)
    loaded_positions, loaded_labels, loaded_digest, games = load_dataset(
        cache, engine, 2, 1500, 120
    )
    assert digest == dataset_digest(positions, labels) == loaded_digest
    assert [board.fen() for board in loaded_positions] == [board.fen() for board in positions]
    assert np.array_equal(loaded_labels, labels)
    assert games == 2
