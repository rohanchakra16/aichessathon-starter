#!/usr/bin/env python3
"""Protected correctness/compliance evaluation for an exact submission artifact."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import chess  # noqa: E402
from artifact import build_deterministic  # noqa: E402

from harness.package import DEFAULT_INCLUDES  # noqa: E402
from harness.referee import FAILED_TERMINATIONS, play_match  # noqa: E402
from harness.sandbox import AgentFailure, local  # noqa: E402

FENS = (
    chess.STARTING_FEN,
    "7k/P7/8/8/8/8/7p/7K w - - 0 1",
    "7k/8/8/8/8/8/p7/7K b - - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "7k/5Q2/7K/8/8/8/8/8 b - - 0 1",
    "8/8/8/8/8/2k5/1q6/K7 w - - 0 1",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 2",
    "8/8/8/8/2k5/8/4K3/7R w - - 99 80",
)
REQUIREMENT = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:\[[A-Za-z0-9_,.-]+\])?"
    r"(?:\s*(?:===|==|~=|!=|<=|>=|<|>)\s*[A-Za-z0-9.*+!_-]+)?(?:\s*;.*)?$"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(command: list[str], timeout: int = 300) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, timeout=timeout
    )
    return {
        "command": command,
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "stdout": completed.stdout[-12000:],
        "stderr": completed.stderr[-12000:],
    }


def static_checks(policy: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    submission = policy["submission"]
    agent = ROOT / "agent.py"
    try:
        tree = ast.parse(agent.read_text(), filename="agent.py")
    except (OSError, SyntaxError) as exc:
        return [f"agent.py cannot be parsed: {exc}"]
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    api = [node for node in functions if node.name == "get_move"]
    if len(api) != 1 or len(api[0].args.args) != 2:
        problems.append("agent.py must define one two-argument get_move function")
    source_lower = agent.read_text().lower()
    for term in submission["forbidden_source_terms"]:
        if term in source_lower:
            problems.append(f"forbidden source term: {term}")
    for name in submission["forbidden_shadow_names"]:
        if (ROOT / name).exists() and name != "agent.py":
            problems.append(f"import-shadowing filename: {name}")
    requirements = ROOT / "requirements.txt"
    if requirements.exists():
        for number, raw in enumerate(requirements.read_text().splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-") or "://" in line or not REQUIREMENT.fullmatch(line):
                problems.append(f"invalid requirements.txt line {number}: {raw}")
    return problems


def package_checks(archive: Path, policy: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    problems: list[str] = []
    submission = policy["submission"]
    with zipfile.ZipFile(archive) as zipped:
        entries = zipped.infolist()
        names = [entry.filename for entry in entries]
        expanded = sum(entry.file_size for entry in entries)
    for required in submission["required_root_files"]:
        if required not in names:
            problems.append(f"missing required root file: {required}")
    if expanded > submission["expanded_size_limit_bytes"]:
        problems.append(f"expanded ZIP is {expanded} bytes")
    suffixes = tuple(submission["native_binary_suffixes"])
    binaries = [name for name in names if name.lower().endswith(suffixes)]
    if binaries:
        problems.append(f"native binary files present: {binaries}")
    return problems, {
        "sha256": sha256(archive),
        "compressed_bytes": archive.stat().st_size,
        "expanded_bytes": expanded,
        "members": names,
    }


def agent_stress(extracted: Path, policy: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    timings: list[float] = []
    agent = local(extracted)
    init_started = time.monotonic()
    try:
        agent.start(policy["reliability"]["max_init_seconds"])
        init_seconds = time.monotonic() - init_started
        for index, fen in enumerate(FENS):
            board = chess.Board(fen)
            if board.is_game_over(claim_draw=True):
                continue
            clock = policy["reliability"]["stress_time_left_ms"][
                index % len(policy["reliability"]["stress_time_left_ms"])
            ]
            started = time.monotonic()
            uci = agent.move(fen, clock)
            elapsed = time.monotonic() - started
            timings.append(elapsed)
            try:
                move = chess.Move.from_uci(uci)
            except chess.InvalidMoveError:
                failures.append(f"invalid UCI for corpus position {index}: {uci!r}")
                continue
            if move not in board.legal_moves:
                failures.append(f"illegal move for corpus position {index}: {uci}")
            if elapsed > policy["reliability"]["max_single_move_seconds"]:
                failures.append(f"slow corpus position {index}: {elapsed:.3f}s")
    except AgentFailure as exc:
        init_seconds = time.monotonic() - init_started
        failures.append(f"agent failure: {exc.reason}")
    finally:
        agent.stop()
    return {
        "passed": not failures,
        "failures": failures,
        "init_seconds": round(init_seconds, 6),
        "max_move_seconds": round(max(timings, default=0.0), 6),
        "positions_checked": len(timings),
    }


def smoke(extracted: Path) -> dict[str, Any]:
    opponent = ROOT / "baselines/random"
    outcomes: list[dict[str, str]] = []
    failures: list[str] = []
    for candidate_white in (True, False):
        white, black = (extracted, opponent) if candidate_white else (opponent, extracted)
        outcome = play_match(local(white), local(black), 5000, 100, ply_cap=160)
        outcomes.append({"result": outcome.result, "termination": outcome.termination})
        if outcome.termination in FAILED_TERMINATIONS:
            failures.append(outcome.termination)
    return {"passed": not failures, "failures": failures, "outcomes": outcomes}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("evaluation.json"))
    args = parser.parse_args()
    started = time.monotonic()
    policy = load_json(ROOT / ".autoloop/protected/policy.json")
    result: dict[str, Any] = {"schema_version": 1, "started_at_epoch": time.time()}
    try:
        result["static_problems"] = static_checks(policy)
        result["ruff"] = command(["uv", "run", "ruff", "check", "."])
        result["mypy"] = command(["uv", "run", "mypy"])
        with tempfile.TemporaryDirectory(prefix="chessathon-eval-") as temporary:
            temp = Path(temporary)
            archive = temp / "submission.zip"
            written = build_deterministic(ROOT, archive, DEFAULT_INCLUDES)
            package_problems, package = package_checks(archive, policy)
            result["packaged_files"] = written
            result["package"] = package
            result["package_problems"] = package_problems
            extracted = temp / "extracted"
            extracted.mkdir()
            with zipfile.ZipFile(archive) as zipped:
                zipped.extractall(extracted)
            result["stress"] = agent_stress(extracted, policy)
            result["smoke"] = smoke(extracted)
        result["passed"] = all(
            (
                not result["static_problems"],
                not result["package_problems"],
                result["ruff"]["passed"],
                result["mypy"]["passed"],
                result["stress"]["passed"],
                result["smoke"]["passed"],
            )
        )
    except Exception as exc:  # persist evaluator failures as data
        result["passed"] = False
        result["evaluator_error"] = f"{type(exc).__name__}: {exc}"
    result["duration_seconds"] = round(time.monotonic() - started, 3)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": result["passed"], "duration_seconds": result["duration_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
