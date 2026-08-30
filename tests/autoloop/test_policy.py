from typing import Any
from unittest.mock import Mock, patch

import controller


def policy() -> dict[str, Any]:
    return controller.load(controller.POLICY_PATH)


def test_submission_paths_are_candidate_editable() -> None:
    current = policy()
    assert controller.path_allowed("agent.py", current)
    assert controller.path_allowed("weights/model.json", current)


def test_protected_paths_are_not_candidate_editable() -> None:
    current = policy()
    protected = (
        ".autoloop/protected/policy.json",
        ".github/workflows/candidate-evaluate.yml",
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
        {"passed": True}, {"passed": True, "score": 0.625}, policy()
    )
    assert status == "accepted"


def test_decision_preserves_inconclusive_result() -> None:
    status, _ = controller.decide(
        {"passed": True}, {"passed": True, "score": 0.5}, policy()
    )
    assert status == "inconclusive"


def test_upload_boundary_is_disabled() -> None:
    assert policy()["competition_upload_enabled"] is False


def test_status_parser_preserves_first_filename_character() -> None:
    completed = Mock(stdout=" M agent.py\n?? weights/new.json\n")
    with patch("controller.run", return_value=completed):
        assert controller.status_paths(controller.ROOT) == ["agent.py", "weights/new.json"]
