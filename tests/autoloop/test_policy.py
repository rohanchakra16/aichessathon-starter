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
from evaluate import stress_limit


def policy() -> dict[str, Any]:
    return controller.load(controller.POLICY_PATH)


def test_submission_paths_are_candidate_editable() -> None:
    current = policy()
    assert controller.path_allowed("agent.py", current)
    assert controller.path_allowed("weights/model.json", current)
    assert not controller.path_allowed("requirements.txt", current)


def test_live_submission_size_limit_is_fifty_megabytes() -> None:
    assert policy()["submission"]["expanded_size_limit_bytes"] == 50_000_000


def test_stress_limits_keep_short_clock_strict_and_allow_real_clock_thinking() -> None:
    current = policy()
    assert stress_limit(current, 3000) == 0.75
    assert stress_limit(current, 120000) == 2.25


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
    status, _ = controller.decide({"passed": False}, {"passed": True, "score": 1.0}, policy())
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


def test_clock_promotion_requires_fast_safety_and_prospective_acceptance() -> None:
    experiment = {
        "ci": {"passed": True},
        "arena": {"statistical_decision": "inconclusive"},
    }
    match = {
        "passed": True,
        "score": 0.75,
        "statistical_decision": "accept",
        "confidence_interval": {"lower": 0.6, "upper": 0.85},
    }
    status, _ = controller.clock_sensitive_decide(experiment, match, policy())
    assert status == "accepted"

    experiment["arena"]["statistical_decision"] = "reject"
    status, _ = controller.clock_sensitive_decide(experiment, match, policy())
    assert status == "rejected"

    experiment["arena"]["statistical_decision"] = "inconclusive"
    match["statistical_decision"] = "inconclusive"
    status, _ = controller.clock_sensitive_decide(experiment, match, policy())
    assert status == "inconclusive"


def test_upload_boundary_is_disabled() -> None:
    assert policy()["competition_upload_enabled"] is False


def test_accepted_candidate_invalidates_previous_release_artifact() -> None:
    state = {
        "next_experiment": 40,
        "submission_candidate": {"champion_commit": "old"},
    }
    with (
        patch("controller.git"),
        patch("controller.atomic_json"),
    ):
        controller.persist("exp-0040", {}, state, "accepted", "new")
    assert state["champion_commit"] == "new"
    assert state["submission_candidate"] is None
    assert state["last_completed_experiment"] == "exp-0040"
    assert state["next_experiment"] == 41


def test_workflow_queries_are_pinned_to_user_fork() -> None:
    assert policy()["github_repository"] == "rohanchakra16/aichessathon-starter"


def test_generator_schedule_uses_claude_for_most_candidate_turns() -> None:
    current = policy()
    expected = {
        0: "claude-code",
        1: "claude-code",
        2: "claude-code",
        3: "codex-exec",
        4: "claude-code",
        5: "claude-code",
        6: "codex-exec",
    }
    assert {
        count: controller.generator_for_stall_count(count, current)
        for count in expected
    } == expected


def test_stall_counter_uses_only_scientific_non_improvements() -> None:
    records = [
        {"status": "rejected"},
        {"status": "infrastructure_error"},
        {"status": "inconclusive"},
        {"status": "accepted"},
        {"status": "rejected"},
    ]
    assert controller.consecutive_non_improvements(records) == 2
    assert controller.consecutive_non_improvements([{"status": "failed"}, *records]) == 0


def test_claude_command_has_no_shell_web_or_permission_bypass() -> None:
    command = controller.claude_command("safe prompt", policy())
    assert command[:2] == ["claude", "-p"]
    assert "--dangerously-skip-permissions" not in command
    assert "--allow-dangerously-skip-permissions" not in command
    assert command[command.index("--permission-mode") + 1] == "acceptEdits"
    disallowed = command[command.index("--disallowed-tools") + 1 :]
    assert "Bash" in disallowed
    assert "WebFetch" in disallowed
    assert "WebSearch" in disallowed
    assert "--no-session-persistence" in command
    assert "--strict-mcp-config" in command


def test_experiment_digest_is_bounded_and_omits_raw_payloads() -> None:
    records = [
        {
            "id": "exp-0046",
            "status": "rejected",
            "generator": "diagnosis",
            "generator_summary": "HYPOTHESIS: prefer safe captures " * 100,
            "decision_reason": "neutral result " * 100,
            "arena": {"pgn": "must not be copied"},
        }
    ]
    digest = controller.experiment_digest(records)
    assert "exp-0046" in digest
    assert "hypothesis:" in digest
    assert "prefer safe captures" in digest
    assert "must not be copied" not in digest
    assert len(digest) < 500


def test_generator_summary_is_single_line_and_bounded() -> None:
    summary = controller.bounded_generator_summary("  first\nsecond  " * 100, 40)
    assert "\n" not in summary
    assert len(summary) == 40


def test_generator_summary_prefers_hypothesis_marker() -> None:
    summary = controller.bounded_generator_summary(
        "long preamble " * 100 + "HYPOTHESIS: focused mechanism"
    )
    assert summary == "HYPOTHESIS: focused mechanism"


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


def test_confirmation_openings_are_independent_legal_and_paired() -> None:
    current = policy()
    promotion = controller.load(controller.ROOT / current["arena"]["openings_file"])
    confirmation = controller.load(controller.ROOT / current["confirmation_arena"]["openings_file"])
    promotion_sequences = {tuple(item["moves"]) for item in promotion["openings"]}
    confirmation_sequences = {tuple(item["moves"]) for item in confirmation["openings"]}
    assert len(confirmation_sequences) == 32
    assert promotion_sequences.isdisjoint(confirmation_sequences)
    assert current["confirmation_arena"]["games"] == 2 * len(confirmation_sequences)
    assert current["real_clock_confirmation"]["games"] == (
        2 * current["real_clock_confirmation"]["maximum_openings"]
    )
    prospective = current["prospective_real_clock_arena"]
    assert prospective["opening_offset"] == current["real_clock_confirmation"]["maximum_openings"]
    assert prospective["games"] == 2 * prospective["maximum_openings"]
    exploratory_sequences = {
        tuple(item["moves"])
        for item in confirmation["openings"][
            : current["real_clock_confirmation"]["maximum_openings"]
        ]
    }
    prospective_sequences = {
        tuple(item["moves"])
        for item in confirmation["openings"][
            prospective["opening_offset"] : prospective["opening_offset"]
            + prospective["maximum_openings"]
        ]
    }
    assert len(prospective_sequences) == 8
    assert exploratory_sequences.isdisjoint(prospective_sequences)
    for sequence in confirmation_sequences:
        board = chess.Board()
        for uci in sequence:
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
