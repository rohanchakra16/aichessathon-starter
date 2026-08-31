import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import chess

import controller

sys.path.insert(0, str(controller.ROOT / ".autoloop/protected"))
from arena import statistical_decision, wilson_score_interval


def policy() -> dict[str, Any]:
    return controller.load(controller.POLICY_PATH)


def test_submission_paths_are_candidate_editable() -> None:
    current = policy()
    assert controller.path_allowed("agent.py", current)
    assert controller.path_allowed("weights/model.json", current)
    assert not controller.path_allowed("requirements.txt", current)


def test_live_submission_size_limit_is_fifty_megabytes() -> None:
    assert policy()["submission"]["expanded_size_limit_bytes"] == 50_000_000


def test_protected_paths_are_not_candidate_editable() -> None:
    current = policy()
    protected = (
        ".autoloop/protected/policy.json",
        ".github/workflows/candidate-evaluate.yml",
        "Makefile",
        "controller.py",
        "harness/referee.py",
        "tests/autoloop/test_policy.py",
    )
    assert all(not controller.path_allowed(path, current) for path in protected)


def test_decision_requires_compliance_before_strength() -> None:
    status, _ = controller.decide(
        {"passed": False}, {"passed": True, "score": 1.0}, policy()
    )
    assert status == "rejected"


def test_decision_accepts_only_promotion_boundary() -> None:
    status, _ = controller.decide(
        {"passed": True},
        {
            "passed": True,
            "score": 0.75,
            "statistical_decision": "accept",
            "confidence_interval": {"lower": 0.6, "upper": 0.85},
        },
        policy(),
    )
    assert status == "accepted"


def test_decision_preserves_inconclusive_result() -> None:
    status, _ = controller.decide(
        {"passed": True},
        {
            "passed": True,
            "score": 0.5,
            "statistical_decision": "inconclusive",
            "confidence_interval": {"lower": 0.4, "upper": 0.6},
        },
        policy(),
    )
    assert status == "inconclusive"


def test_decision_honours_statistical_rejection() -> None:
    status, _ = controller.decide(
        {"passed": True},
        {
            "passed": True,
            "score": 0.25,
            "statistical_decision": "reject",
            "confidence_interval": {"lower": 0.15, "upper": 0.4},
        },
        policy(),
    )
    assert status == "rejected"


def test_upload_boundary_is_disabled() -> None:
    assert policy()["competition_upload_enabled"] is False


def test_workflow_queries_are_pinned_to_user_fork() -> None:
    assert policy()["github_repository"] == "rohanchakra16/aichessathon-starter"


def test_status_parser_preserves_first_filename_character() -> None:
    completed = Mock(stdout=" M agent.py\n?? weights/new.json\n")
    with patch("controller.run", return_value=completed):
        assert controller.status_paths(controller.ROOT) == ["agent.py", "weights/new.json"]


def test_frozen_openings_are_legal_and_paired() -> None:
    current = policy()
    openings_path = controller.ROOT / current["arena"]["openings_file"]
    openings = controller.load(openings_path)["openings"]
    assert current["arena"]["games"] == 2 * len(openings)
    for opening in openings:
        board = chess.Board()
        for uci in opening["moves"]:
            move = chess.Move.from_uci(uci)
            assert move in board.legal_moves
            board.push(move)


def test_sequential_boundaries_are_declared_and_directional() -> None:
    settings = policy()["arena"]
    assert settings["minimum_games"] % settings["batch_games"] == 0
    assert settings["games"] % settings["batch_games"] == 0
    accept, lower, _ = statistical_decision(28, 4, 0, settings)
    reject, _, upper = statistical_decision(0, 4, 28, settings)
    assert accept == "accept" and lower > settings["null_score"]
    assert reject == "reject" and upper < settings["null_score"]


def test_wilson_interval_contains_even_score() -> None:
    lower, upper = wilson_score_interval(8, 16, 8, 1.6448536269514722)
    assert lower < 0.5 < upper


def test_release_gate_cannot_upload_to_competition() -> None:
    workflow = (controller.ROOT / ".github/workflows/release-evaluate.yml").read_text()
    assert "workflow_dispatch" in workflow
    assert "aichessathon.com" not in workflow


def test_release_zip_is_byte_reproducible() -> None:
    script = controller.ROOT / ".autoloop/protected/artifact.py"
    with tempfile.TemporaryDirectory() as temporary:
        first = Path(temporary) / "first.zip"
        second = Path(temporary) / "second.zip"
        for output in (first, second):
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--root",
                    str(controller.ROOT),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        assert first.read_bytes() == second.read_bytes()
