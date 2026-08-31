#!/usr/bin/env python3
"""Protected exact-artifact evaluation inside the competition resource envelope."""

from __future__ import annotations

import argparse
import ast
import contextlib
import errno
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROTECTED = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROTECTED))

import chess  # noqa: E402
from arena import opening_fen, play_from_fen  # noqa: E402
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


def command(arguments: list[str], timeout: int = 300) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        arguments, cwd=ROOT, text=True, capture_output=True, timeout=timeout
    )
    return {
        "command": arguments,
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "stdout": completed.stdout[-12000:],
        "stderr": completed.stderr[-12000:],
    }


def read_cgroup(name: str) -> str | None:
    path = Path("/sys/fs/cgroup") / name
    try:
        return path.read_text().strip()
    except OSError:
        return None


def environment_probe(policy: dict[str, Any]) -> dict[str, Any]:
    expected = policy["environment"]
    problems: list[str] = []
    cpu_raw = read_cgroup("cpu.max")
    memory_raw = read_cgroup("memory.max")
    pids_raw = read_cgroup("pids.max")
    tmp_stats = os.statvfs("/tmp")
    tmp_bytes = tmp_stats.f_frsize * tmp_stats.f_blocks

    cpu_cores: float | None = None
    if cpu_raw is not None:
        quota, period = cpu_raw.split()
        if quota != "max":
            cpu_cores = int(quota) / int(period)
    memory_bytes = int(memory_raw) if memory_raw and memory_raw != "max" else None
    pids_limit = int(pids_raw) if pids_raw and pids_raw != "max" else None

    if os.environ.get("AUTOLOOP_CONSTRAINED") != "1":
        problems.append("constrained-run marker is absent")
    if cpu_cores is None or cpu_cores > float(expected["cpu_quota_cores"]) + 0.01:
        problems.append(f"CPU quota is not <= {expected['cpu_quota_cores']}: {cpu_raw}")
    if memory_bytes != int(expected["memory_limit_bytes"]):
        problems.append(f"memory.max mismatch: {memory_raw}")
    if pids_limit is None or pids_limit > int(expected["pids_limit"]):
        problems.append(f"pids.max is not <= {expected['pids_limit']}: {pids_raw}")
    if tmp_bytes > int(expected["tmp_limit_bytes"]):
        problems.append(f"/tmp exceeds protected limit: {tmp_bytes}")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.25)
        network_errno = probe.connect_ex(("1.1.1.1", 53))
    network_disabled = network_errno in {
        errno.ENETDOWN,
        errno.ENETUNREACH,
        errno.EHOSTUNREACH,
        errno.EPERM,
        errno.EACCES,
    }
    if expected["require_network_disabled"] and not network_disabled:
        problems.append(f"outbound network was not conclusively disabled: errno={network_errno}")

    workspace_probe = ROOT / ".autoloop-write-probe"
    workspace_read_only = False
    try:
        workspace_probe.write_text("probe")
    except OSError as exc:
        workspace_read_only = exc.errno in {errno.EROFS, errno.EACCES, errno.EPERM}
    finally:
        with contextlib.suppress(OSError):
            workspace_probe.unlink()
    if expected["require_read_only_workspace"] and not workspace_read_only:
        problems.append("workspace bind mount is writable")

    tmp_probe = Path("/tmp/autoloop-write-probe")
    tmp_writable = False
    try:
        tmp_probe.write_text("probe")
        tmp_writable = True
    finally:
        tmp_probe.unlink(missing_ok=True)
    if not tmp_writable:
        problems.append("/tmp scratch is not writable")

    return {
        "passed": not problems,
        "problems": problems,
        "image_id": os.environ.get("AUTOLOOP_IMAGE_ID"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_max": cpu_raw,
        "cpu_quota_cores": cpu_cores,
        "memory_max": memory_raw,
        "memory_limit_bytes": memory_bytes,
        "pids_max": pids_raw,
        "pids_limit": pids_limit,
        "tmp_limit_bytes": tmp_bytes,
        "network_probe_errno": network_errno,
        "network_disabled": network_disabled,
        "workspace_read_only": workspace_read_only,
        "tmp_writable": tmp_writable,
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


def freeze_tree(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def agent_stress(extracted: Path, policy: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    timings: list[float] = []
    agent = local(extracted)
    init_started = time.monotonic()
    init_seconds = 0.0
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


def create_ablated(candidate: Path, destination: Path, policy: dict[str, Any]) -> dict[str, Any]:
    shutil.copytree(candidate, destination)
    relative = Path(policy["model_ablation"]["model_path"])
    model_path = destination / relative
    model = load_json(model_path)
    raw_weights = model.get("weights")
    if not isinstance(raw_weights, list) or not raw_weights:
        raise ValueError(f"ablation model has no weight vector: {relative}")
    weights = [float(value) for value in raw_weights]
    nonzero = sum(value != 0.0 for value in weights)
    if nonzero == 0:
        raise ValueError("learned evaluator already has no non-zero weights")
    model["weights"] = [0.0] * len(weights)
    model["bias"] = 0.0
    model_path.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n")
    return {"model_path": str(relative), "weights_zeroed": nonzero}


def model_move_ablation(
    candidate: Path, ablated: Path, policy: dict[str, Any]
) -> dict[str, Any]:
    settings = policy["model_ablation"]
    openings = load_json(ROOT / policy["arena"]["openings_file"])["openings"]
    candidate_agent = local(candidate)
    ablated_agent = local(ablated)
    comparisons: list[dict[str, str]] = []
    failures: list[str] = []
    try:
        candidate_agent.start(policy["reliability"]["max_init_seconds"])
        ablated_agent.start(policy["reliability"]["max_init_seconds"])
        for opening in openings:
            fen = opening_fen(opening["moves"])
            board = chess.Board(fen)
            candidate_move = candidate_agent.move(fen, settings["move_time_left_ms"])
            ablated_move = ablated_agent.move(fen, settings["move_time_left_ms"])
            for label, uci in (("candidate", candidate_move), ("ablated", ablated_move)):
                try:
                    move = chess.Move.from_uci(uci)
                except chess.InvalidMoveError:
                    failures.append(f"{opening['id']}:{label}:invalid:{uci}")
                    continue
                if move not in board.legal_moves:
                    failures.append(f"{opening['id']}:{label}:illegal:{uci}")
            comparisons.append(
                {
                    "opening": opening["id"],
                    "candidate": candidate_move,
                    "ablated": ablated_move,
                }
            )
    except AgentFailure as exc:
        failures.append(f"agent failure: {exc.reason}")
    finally:
        candidate_agent.stop()
        ablated_agent.stop()
    differences = sum(row["candidate"] != row["ablated"] for row in comparisons)
    minimum = int(settings["minimum_move_differences"])
    return {
        "passed": not failures and differences >= minimum,
        "failures": failures,
        "differences": differences,
        "minimum_differences": minimum,
        "positions": len(comparisons),
        "comparisons": comparisons,
    }


def model_strength_ablation(
    candidate: Path, ablated: Path, policy: dict[str, Any]
) -> dict[str, Any]:
    settings = policy["model_ablation"]
    openings = load_json(ROOT / policy["arena"]["openings_file"])["openings"]
    required_games = int(settings["strength_games"])
    selected = openings[: required_games // 2]
    games: list[dict[str, Any]] = []
    wins = draws = losses = 0
    failures: list[str] = []
    for opening in selected:
        fen = opening_fen(opening["moves"])
        for candidate_white in (True, False):
            white, black = (candidate, ablated) if candidate_white else (ablated, candidate)
            game = play_from_fen(
                local(white),
                local(black),
                fen,
                opening["id"],
                int(settings["base_ms"]),
                int(settings["increment_ms"]),
                int(settings["ply_cap"]),
            )
            game["candidate_colour"] = "white" if candidate_white else "black"
            games.append(game)
            if game["termination"] in FAILED_TERMINATIONS:
                failures.append(
                    f"{opening['id']}:{game['candidate_colour']}:{game['termination']}"
                )
            if game["result"] in ("draw", "void"):
                draws += 1
            elif (game["result"] == "white") == candidate_white:
                wins += 1
            else:
                losses += 1
    score = (wins + draws / 2.0) / len(games)
    minimum = float(settings["minimum_strength_score"])
    return {
        "passed": not failures and len(games) == required_games and score >= minimum,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "minimum_score": minimum,
        "failures": failures,
        "games": games,
    }


def real_clock_games(extracted: Path, policy: dict[str, Any]) -> dict[str, Any]:
    settings = policy["release"]
    opponent = ROOT / "baselines/greedy"
    outcomes: list[dict[str, Any]] = []
    failures: list[str] = []
    started = time.monotonic()
    for candidate_white in (True, False):
        white, black = (extracted, opponent) if candidate_white else (opponent, extracted)
        game_started = time.monotonic()
        outcome = play_match(
            local(white),
            local(black),
            int(settings["base_ms"]),
            int(settings["increment_ms"]),
            ply_cap=int(settings["ply_cap"]),
        )
        outcomes.append(
            {
                "candidate_colour": "white" if candidate_white else "black",
                "result": outcome.result,
                "termination": outcome.termination,
                "duration_seconds": round(time.monotonic() - game_started, 3),
                "pgn": outcome.pgn,
            }
        )
        if outcome.termination in FAILED_TERMINATIONS:
            failures.append(f"{outcomes[-1]['candidate_colour']}:{outcome.termination}")
    return {
        "passed": not failures and len(outcomes) == int(settings["real_clock_games"]),
        "failures": failures,
        "outcomes": outcomes,
        "duration_seconds": round(time.monotonic() - started, 3),
    }


def resource_telemetry(policy: dict[str, Any]) -> dict[str, Any]:
    peak_raw = read_cgroup("memory.peak")
    current_raw = read_cgroup("memory.current")
    peak = int(peak_raw) if peak_raw and peak_raw != "max" else None
    limit = int(policy["environment"]["memory_limit_bytes"])
    return {
        "passed": peak is not None and peak <= limit,
        "memory_peak_bytes": peak,
        "memory_current_bytes": int(current_raw) if current_raw else None,
        "memory_limit_bytes": limit,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("evaluation.json"))
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    policy = load_json(ROOT / ".autoloop/protected/policy.json")
    result: dict[str, Any] = {
        "schema_version": 2,
        "mode": "release" if args.release else "candidate",
        "started_at_epoch": time.time(),
    }
    try:
        result["environment"] = environment_probe(policy)
        result["static_problems"] = static_checks(policy)
        result["ruff"] = command(["ruff", "check", "."])
        result["mypy"] = command(["mypy"])
        result["policy_tests"] = command(
            ["pytest", "-q", "-p", "no:cacheprovider", "tests/autoloop"]
        )
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
            ablated = temp / "ablated"
            result["ablation_setup"] = create_ablated(extracted, ablated, policy)
            freeze_tree(extracted)
            freeze_tree(ablated)
            result["stress"] = agent_stress(extracted, policy)
            result["smoke"] = smoke(extracted)
            result["model_move_ablation"] = model_move_ablation(extracted, ablated, policy)
            if args.release:
                result["model_strength_ablation"] = model_strength_ablation(
                    extracted, ablated, policy
                )
                result["real_clock"] = real_clock_games(extracted, policy)
        result["resources"] = resource_telemetry(policy)
        checks = [
            result["environment"]["passed"],
            not result["static_problems"],
            not result["package_problems"],
            result["ruff"]["passed"],
            result["mypy"]["passed"],
            result["policy_tests"]["passed"],
            result["stress"]["passed"],
            result["smoke"]["passed"],
            result["model_move_ablation"]["passed"],
            result["resources"]["passed"],
        ]
        if args.release:
            checks.extend(
                (
                    result["model_strength_ablation"]["passed"],
                    result["real_clock"]["passed"],
                )
            )
        result["passed"] = all(checks)
    except Exception as exc:  # persist evaluator failures as data
        result["passed"] = False
        result["evaluator_error"] = f"{type(exc).__name__}: {exc}"
    result["duration_seconds"] = round(time.monotonic() - started, 3)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": result["passed"], "duration_seconds": result["duration_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
