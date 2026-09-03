from __future__ import annotations

import json

import pytest

import claude_supervisor as supervisor


def decision(decision_name: str = "CONTINUE") -> dict[str, object]:
    continuing = decision_name == "CONTINUE"
    return {
        "decision": decision_name,
        "stop_condition": None if continuing else "scientific_saturation",
        "streak_assessment": "The evidence has been reviewed.",
        "recurring_failure_modes": ["threefold repetition"],
        "next_direction": (
            {
                "title": "Mobility",
                "family": "leaf-evaluation",
                "hypothesis": "Add a cheap mobility signal.",
                "rationale": "The evaluator currently lacks activity features.",
                "guardrails": "Preserve NPS and the model-ablation gate.",
            }
            if continuing
            else None
        ),
        "audit_summary": "A concise evidence summary.",
    }


def test_schema_enforced_output_wins_over_malformed_prose() -> None:
    expected = decision()
    payload = {
        "structured_output": expected,
        "result": '<AUDIT_DECISION>{"decision": CONTINUE}</AUDIT_DECISION>',
    }
    assert supervisor.decision_from_payload(payload) == expected


def test_structured_output_may_be_serialized_json() -> None:
    expected = decision("STOP")
    assert supervisor.decision_from_payload(
        {"structured_output": json.dumps(expected)}
    ) == expected


def test_legacy_tagged_output_remains_readable() -> None:
    expected = decision()
    result = (
        "analysis\n"
        f"{supervisor.DECISION_OPEN}\n{json.dumps(expected)}\n"
        f"{supervisor.DECISION_CLOSE}"
    )
    assert supervisor.parse_decision(result) == expected


def test_continue_requires_a_direction() -> None:
    payload = decision()
    payload["next_direction"] = None
    with pytest.raises(supervisor.ClaudeUnavailable, match="no next direction"):
        supervisor.validate_decision(payload)


def test_stop_requires_a_stop_condition() -> None:
    payload = decision("STOP")
    payload["stop_condition"] = None
    with pytest.raises(supervisor.ClaudeUnavailable, match="no valid stop condition"):
        supervisor.validate_decision(payload)
