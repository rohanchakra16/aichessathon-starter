#!/usr/bin/env python3
"""One-time generator for the positional/endgame retrain datasets.

Wraps ``generate_active_learning_dataset.py`` (Stockfish-18 offline teacher, no
protected opening list) with endgame-favouring settings, then splits the rows by
whole game into two disjoint datasets:

    training/data/positional_teacher_train.json
    training/data/positional_teacher_validation.json
    training/data/MANIFEST.json

Run it once from a normal Terminal with the pinned Stockfish 18 on PATH, commit
the three files, then pin the printed sha256 values into
``.autoloop/protected/policy.json`` -> ``retrain.datasets``.

    ./.venv/bin/python training/generate_positional_teacher_dataset.py

Nothing here ships in the agent zip (``DEFAULT_INCLUDES = ("agent.py", "weights")``).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.generate_active_learning_dataset import dataset_digest  # noqa: E402
from training.train_stockfish_evaluator import file_sha256  # noqa: E402

DATA_DIR = ROOT / "training/data"
GENERATOR = ROOT / "training/generate_active_learning_dataset.py"
SEED = 20260904
# Endgame-favouring: long champion trajectories so conversion/technical
# positions are sampled, a broad context budget, disagreement-ranked selection.
GENERATE_ARGS = [
    "--games", "96",
    "--opening-plies", "8",
    "--maximum-plies", "150",
    "--champion-time-left-ms", "1000",
    "--sample-stride", "3",
    "--opening-nodes", "1000",
    "--label-nodes", "3000",
    "--multipv", "3",
    "--selected-contexts", "900",
    "--seed", str(SEED),
]
VALIDATION_FRACTION = 0.25


def write_dataset(path: Path, base: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    payload = {
        **{key: value for key, value in base.items() if key != "rows"},
        "rows": rows,
        "rows_count": len(rows),
        "dataset_sha256": dataset_digest(rows),
        "split_seed": SEED,
        "split_fraction_validation": VALIDATION_FRACTION,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return str(payload["dataset_sha256"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, help="Stockfish binary (default: PATH)")
    parser.add_argument("--keep-combined", action="store_true")
    args = parser.parse_args()

    engine = args.engine or (
        Path(shutil.which("stockfish")) if shutil.which("stockfish") else None
    )
    if engine is None:
        parser.error("Stockfish 18 is required on PATH or via --engine")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        combined = Path(temporary) / "combined.json"
        command = [
            sys.executable, str(GENERATOR),
            "--engine", str(engine),
            "--output", str(combined),
            *GENERATE_ARGS,
        ]
        print("running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        base = json.loads(combined.read_text())
        if args.keep_combined:
            shutil.copy(combined, DATA_DIR / "positional_teacher_combined.json")

    rows: list[dict[str, Any]] = base["rows"]
    game_ids = sorted({int(row["game_id"]) for row in rows})
    if len(game_ids) < 8:
        raise SystemExit(f"only {len(game_ids)} games; need >= 8 for a grouped split")
    validation_count = max(2, round(len(game_ids) * VALIDATION_FRACTION))
    # Deterministic interleaved split so both halves span the run.
    validation_games = {game_ids[index] for index in range(0, len(game_ids), 4)}
    while len(validation_games) < validation_count:
        for candidate in game_ids:
            if candidate not in validation_games:
                validation_games.add(candidate)
                break
    training_rows = [row for row in rows if int(row["game_id"]) not in validation_games]
    validation_rows = [row for row in rows if int(row["game_id"]) in validation_games]

    train_path = DATA_DIR / "positional_teacher_train.json"
    validation_path = DATA_DIR / "positional_teacher_validation.json"
    train_sha = write_dataset(train_path, base, training_rows)
    validation_sha = write_dataset(validation_path, base, validation_rows)

    manifest = {
        "schema_version": 1,
        "purpose": "learned-evaluator-retrain family (training/train_positional_evaluator.py)",
        "generator_script": str(GENERATOR.relative_to(ROOT)),
        "generator_script_sha256": file_sha256(GENERATOR),
        "generator_args": GENERATE_ARGS,
        "teacher_name": "Stockfish 18",
        "teacher_binary_sha256": file_sha256(Path(engine)),
        "seed": SEED,
        "combined_rows": len(rows),
        "combined_games": len(game_ids),
        "combined_dataset_sha256": base["dataset_sha256"],
        "validation_fraction": VALIDATION_FRACTION,
        "validation_games": sorted(validation_games),
        "training": {
            "path": str(train_path.relative_to(ROOT)),
            "rows": len(training_rows),
            "games": len(game_ids) - len(validation_games),
            "sha256": train_sha,
        },
        "validation": {
            "path": str(validation_path.relative_to(ROOT)),
            "rows": len(validation_rows),
            "games": len(validation_games),
            "sha256": validation_sha,
        },
    }
    (DATA_DIR / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    print("\nwrote:")
    print(f"  {train_path.relative_to(ROOT)}  sha256={train_sha}")
    print(f"  {validation_path.relative_to(ROOT)}  sha256={validation_sha}")
    print(f"  {(DATA_DIR / 'MANIFEST.json').relative_to(ROOT)}")
    print(
        "\npin these into .autoloop/protected/policy.json -> retrain.datasets:\n"
        f'  training.sha256   = "{train_sha}"\n'
        f'  validation.sha256 = "{validation_sha}"'
    )


if __name__ == "__main__":
    main()
