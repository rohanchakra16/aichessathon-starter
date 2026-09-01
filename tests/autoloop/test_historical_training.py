import io
import random

import chess.pgn

from training.generate_historical_dataset import (
    SEED,
    game_positions,
    position_digest,
)

PGN = """[Event "Historical test"]
[Site "?"]
[Date "2026.08.31"]
[Round "1"]
[White "White"]
[Black "Black"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. d3 Bc5 5. O-O d6 6. c3 O-O
7. Re1 a6 8. Bb3 Ba7 9. Nbd2 Re8 10. Nf1 h6 11. Ng3 Be6 12. Bc2 d5
13. exd5 Bxd5 14. Be3 Bxe3 15. Rxe3 Qd6 16. Qe2 Rad8 17. Re1 1-0
"""


def test_historical_selection_is_game_grouped_and_reproducible() -> None:
    first_game = chess.pgn.read_game(io.StringIO(PGN))
    second_game = chess.pgn.read_game(io.StringIO(PGN))
    assert first_game is not None and second_game is not None
    first = game_positions(first_game, 17, random.Random(SEED), 120, 3)
    second = game_positions(second_game, 17, random.Random(SEED), 120, 3)
    assert first == second
    assert len(first) == 3
    assert {row["game_id"] for row in first} == {17}
    assert {row["source"] for row in first} <= {
        "historical_quiet",
        "historical_tactical",
    }
    assert position_digest(first) == position_digest(second)
