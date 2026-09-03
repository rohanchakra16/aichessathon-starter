#!/usr/bin/env python3
"""Unattended research supervisor for the AI Chessathon autoloop.

Run this from a normal macOS Terminal where ``gh`` and GitHub Actions work. It
turns the takeover handoff into a genuinely unattended loop:

    evidence audit (Claude)  ->  research direction  ->  bounded controller batch
        ^                                                        |
        +--------------------  repeat  ---------------------------+

Each cycle:

1. Build a compact evidence packet (champion, the batch's new experiment
   records with PGNs stripped, streak, saturated families, the previous audit).
2. Ask Claude Code (``claude -p``) for the takeover's full between-batch audit
   and a structured ``CONTINUE`` / ``STOP`` decision plus the next research
   direction.
3. Persist the audit verbatim under ``research/audits/`` (permanent trail).
4. On ``CONTINUE``: write ``research/next-direction.md`` (the controller feeds it
   to the next candidate), commit the trail, push, then run one bounded
   ``controller.py`` batch. On ``STOP``: stop and print the reason.
5. A promotion or a finished batch is never itself a stop condition; the next
   cycle simply audits against the (possibly new) champion and continues.

Boundaries this script never crosses:

* It never edits ``agent.py``, ``weights/`` or any protected path, and never
  changes evaluation, reliability, promotion or the upload boundary. The
  controller and its protected framework remain the sole evaluation/promotion
  authority. This script only schedules batches and routes research direction.
* It never uploads anything to the competition.
* It only ever commits ``research/`` (audit trail + next-direction file).
* Codex is never used; the loop is Claude-only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import controller  # helper reuse only; controller stays the evaluation authority

ROOT = Path(controller.ROOT)
STATE_PATH = Path(controller.STATE_PATH)
POLICY_PATH = Path(controller.POLICY_PATH)
AUDIT_DIR = ROOT / "research/audits"
PACKET_DIR = ROOT / "research/audit-input"
DIRECTION_PATH = ROOT / "research/next-direction.md"
STOP_FILE = ROOT / ".autoloop/supervisor.stop"
LOG_PATH = ROOT / "research/audits/supervisor-log.jsonl"

DECISION_OPEN = "<AUDIT_DECISION>"
DECISION_CLOSE = "</AUDIT_DECISION>"
AUDIT_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["CONTINUE", "STOP"]},
        "stop_condition": {
            "type": ["string", "null"],
            "enum": [None, "infrastructure_blocker", "scientific_saturation"],
        },
        "streak_assessment": {"type": "string"},
        "recurring_failure_modes": {
            "type": "array",
            "items": {"type": "string"},
        },
        "next_direction": {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "family": {"type": "string"},
                        "hypothesis": {"type": "string"},
                        "rationale": {"type": "string"},
                        "guardrails": {"type": "string"},
                    },
                    "required": [
                        "title",
                        "family",
                        "hypothesis",
                        "rationale",
                        "guardrails",
                    ],
                    "additionalProperties": False,
                },
                {"type": "null"},
            ]
        },
        "audit_summary": {"type": "string"},
    },
    "required": [
        "decision",
        "stop_condition",
        "streak_assessment",
        "recurring_failure_modes",
        "next_direction",
        "audit_summary",
    ],
    "additionalProperties": False,
}

# Mechanism families the handoff and the retained registry already record as
# exhausted in their tested form. Kept here so every audit sees them without
# re-deriving; the audit may still refine this list in its reasoning.
SATURATED_FAMILIES = [
    "history / killer / TT-best-move move ordering",
    "aspiration windows",
    "guarded null-move pruning",
    "bounded check extensions / evasions",
    "general forcing-check priority (decisively worse)",
    "mate-distance scoring",
    "quiet promotions in quiescence",
    "quiescence delta pruning",
    "draw contempt",
    "halfmove-clock TT bucketing / early repetition-call guarding",
    "timeout exception unwinding",
    "exact quiescence stalemate repair",
    "move-ordering key micro-optimisation (packed int vs UCI string)",
    "time-budget / clock allocation changes without a structural search change",
    "simple six-feature linear king-safety residual (offline: no signal)",
]


class ClaudeUnavailable(RuntimeError):
    """Claude Code could not complete an audit (auth, usage, or transport)."""


class SupervisorBlocked(RuntimeError):
    """A precondition for safe unattended operation is not met."""


def now() -> str:
    return datetime.now(UTC).isoformat()


def run(
    command: list[str],
    *,
    cwd: Path = ROOT,
    timeout: int | None = None,
    check: bool = True,
    stdin: str | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
        input=stdin,
        env=env,
    )
    if check and completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}"
        )
    return completed


def git(*arguments: str, check: bool = True) -> str:
    return run(["git", *arguments], check=check).stdout.strip()


def load_json(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def subscription_env() -> dict[str, str]:
    """Environment for ``claude -p`` that forces the interactive subscription."""
    environment = os.environ.copy()
    for variable in ("ANTHROPIC_API_KEY", "CODEX_API_KEY", "OPENAI_API_KEY"):
        environment.pop(variable, None)
    return environment


# --------------------------------------------------------------------------- #
# Evidence packet
# --------------------------------------------------------------------------- #


def experiment_summary(path: Path) -> dict[str, Any]:
    """Compact, PGN-free summary of one retained experiment record."""
    record = load_json(path)
    arena = record.get("arena", {}) or {}
    games = arena.get("games", []) or []
    losses: list[dict[str, str]] = []
    draw_terminations: dict[str, int] = {}
    for game in games:
        colour = game.get("candidate_colour")
        result = game.get("result")
        termination = str(game.get("termination", "unknown"))
        if result not in (None, "draw", colour):
            losses.append({"colour": str(colour), "termination": termination})
        if result == "draw":
            draw_terminations[termination] = draw_terminations.get(termination, 0) + 1
    ci = record.get("ci", {}) or {}
    return {
        "id": record.get("id"),
        "status": record.get("status"),
        "generator": record.get("generator"),
        "hypothesis": record.get("hypothesis") or record.get("generator_summary") or "",
        "decision_reason": record.get("decision_reason") or record.get("failure") or "",
        "consecutive_non_improvements": record.get("consecutive_non_improvements"),
        "candidate_commit": record.get("candidate_commit"),
        "arena": {
            "wins": arena.get("wins"),
            "draws": arena.get("draws"),
            "losses": arena.get("losses"),
            "score": arena.get("score"),
            "statistical_decision": arena.get("statistical_decision"),
            "confidence_interval": arena.get("confidence_interval"),
            "agent_failures": len(arena.get("failures", []) or []),
        },
        "ci_passed": ci.get("passed"),
        "reliability_stress_passed": (ci.get("stress", {}) or {}).get("passed"),
        "draw_terminations": draw_terminations,
        "loss_breakdown": losses,
        "record_path": str(path.relative_to(ROOT)),
    }


def previous_audit() -> dict[str, Any] | None:
    if not AUDIT_DIR.exists():
        return None
    files = sorted(AUDIT_DIR.glob("audit-*.json"))
    if not files:
        return None
    return load_json(files[-1])


def build_packet(
    state: dict[str, Any],
    batch_ids: list[str],
    champion_before: str | None,
    promotion: bool,
    batch_stdout: str,
    batch_infra_error: str | None,
) -> str:
    policy = load_json(POLICY_PATH)
    limit = int(policy["candidate_generators"]["recent_experiment_limit"])
    records = controller.recent_experiment_records(state, limit)
    streak = controller.consecutive_non_improvements(records)
    champion = state["champion_commit"]

    history_lines = []
    for record in records[:20]:
        hypothesis = record.get("hypothesis") or record.get("generator_summary") or ""
        hypothesis = " ".join(str(hypothesis).split())[:150]
        history_lines.append(f"- {record.get('id')}: {record.get('status')}; {hypothesis}")

    batch_blocks = []
    for experiment_id in batch_ids:
        path = ROOT / f"experiments/{experiment_id}.json"
        if path.exists():
            batch_blocks.append(json.dumps(experiment_summary(path), indent=2))
        else:
            batch_blocks.append(f'{{"id": "{experiment_id}", "note": "record not found"}}')
    batch_text = (
        "\n".join(batch_blocks)
        if batch_blocks
        else "None yet - this audit sets the direction for the first batch."
    )

    prior = previous_audit()
    if prior:
        prior_block = json.dumps(
            {
                "completed_at": prior.get("completed_at"),
                "after_experiment": prior.get("after_experiment"),
                "decision": prior.get("decision"),
                "next_direction": prior.get("next_direction"),
                "audit_summary": prior.get("audit_summary"),
            },
            indent=2,
        )
    else:
        prior_block = "None. This is the first audit of the run."

    infra_note = (
        f"\nThe last controller batch FAILED with an infrastructure error:\n{batch_infra_error}\n"
        "An infrastructure/generator failure is NOT a scientific non-improvement.\n"
        if batch_infra_error
        else ""
    )
    promotion_note = (
        f"\nA PROMOTION occurred in the last batch. Champion moved "
        f"{champion_before} -> {champion}. Audit against the NEW champion and, if "
        "continuing, reset the scientific non-improvement streak in your reasoning.\n"
        if promotion
        else ""
    )

    return f"""You are the supervising chess-engine researcher for the AI Chessathon bot
Phineas, performing the mandatory between-batch evidence audit from
docs/autoloop/claude-takeover-prompt.md and
docs/autoloop/claude-handoff-2026-09-03.md.

You have read-only tools (Read, Grep, Glob). The working directory is the repo
root. Use them to inspect anything you need: agent.py (the current champion),
experiments/exp-*.json (full records, including game PGNs, for the batch just
run and any earlier experiment), docs/autoloop/*, research/*, .autoloop/state.json,
.autoloop/protected/policy.json. Do not attempt to edit anything.

## Fixed context

Champion commit: {champion}
Live competition upload: older exp-0052-era champion (do not upload; out of scope).
next_experiment in state.json: {state.get("next_experiment")}
Consecutive scientific non-improvements before this audit: {streak}
{promotion_note}{infra_note}
## Recent experiment registry (newest first, compact)

{chr(10).join(history_lines)}

## New experiment record(s) from the batch just run

{batch_text}

Controller stdout (tail):
{batch_stdout[-1500:] if batch_stdout else "(none)"}

## Mechanism families already recorded as exhausted in their tested form

{chr(10).join(f"- {family}" for family in SATURATED_FAMILIES)}

Changing only a cap, threshold, depth, margin, table size, ordering weight or
time constant of any of the above is NOT a materially new hypothesis.

## Previous audit

{prior_block}

## Your task

1. Inspect the champion, the new record(s), and whatever retained
   losing/drawn-game evidence or profiling you need. Note recurring failure
   modes (e.g. threefold-repetition draws in balanced or better positions,
   endgame conversion, king-attack defence).
2. Judge whether the current search/move-ordering/quiescence/time-management
   tuning direction is saturated per the handoff's stop rule (>= 5 consecutive
   scientific non-improvements since the newest promotion AND no materially
   different, well-motivated experiment left in that direction).
3. Decide CONTINUE or STOP.
   - CONTINUE if a materially different, well-motivated experiment remains,
     whether in that direction or another (richer leaf evaluation such as
     mobility / pawn structure / passed pawns / bishop pair / rook-on-open-file
     / non-linear king safety; a small opening book designed to not defeat the
     model-move-ablation gate; persisted cross-move search state; something you
     identify from the evidence). Prefer genuinely new directions over more
     tuning.
   - STOP only for: a genuine infrastructure/auth/rules blocker
     (stop_condition "infrastructure_blocker"); or true scientific saturation
     (stop_condition "scientific_saturation") after a full evidence audit with
     no materially new well-motivated experiment anywhere.
4. If CONTINUE, specify the single next hypothesis. Constraints on it:
   - The candidate generator is one `claude -p` run restricted to editing
     agent.py and/or weights/. It has no Bash and cannot run training/ pipelines
     or produce a newly trained weights/model.json. So the hypothesis must be
     self-contained: a code change in agent.py, optionally with small
     hand-chosen or closed-form coefficients, or a change to how existing
     weights are used. If a direction truly needs an offline-trained artifact,
     say so explicitly and scope a self-contained first step the candidate can
     ship now (e.g. a single cheaply-computed feature with a conservative fixed
     coefficient) and flag the training follow-up for the human.
   - It must keep the trained model materially determining leaf evaluation and
     move selection, hold one CPU / 2 GB / no network, keep get_move returning a
     legal UCI move under the real clock, and not repeat a saturated family.
   - Give concrete guardrails (NPS budget, model-ablation-gate safety,
     reliability/flag risk).

## Required structured output

Return exactly one object matching the JSON Schema enforced by the Claude CLI.
Do not wrap it in Markdown, prose, code fences or XML tags. Put the important
analysis in `streak_assessment`, `recurring_failure_modes`, `rationale` and
`audit_summary`. For CONTINUE, `stop_condition` must be null and
`next_direction` must be a complete object. For STOP, `stop_condition` must be
`infrastructure_blocker` or `scientific_saturation`, and `next_direction` may
be null."""


# --------------------------------------------------------------------------- #
# Claude audit
# --------------------------------------------------------------------------- #


@dataclass
class AuditResult:
    decision: str
    stop_condition: str | None
    next_direction: dict[str, Any] | None
    audit_summary: str
    raw_result: str
    payload: dict[str, Any]
    parsed: dict[str, Any] = field(default_factory=dict)


def validate_decision(parsed: dict[str, Any]) -> dict[str, Any]:
    """Apply semantic checks in addition to Claude's JSON Schema validation."""
    decision = parsed.get("decision")
    if decision not in ("CONTINUE", "STOP"):
        raise ClaudeUnavailable(f"audit decision was not CONTINUE/STOP: {decision!r}")
    stop_condition = parsed.get("stop_condition")
    direction = parsed.get("next_direction")
    if decision == "CONTINUE":
        if stop_condition is not None:
            raise ClaudeUnavailable("CONTINUE audit supplied a stop condition")
        if not isinstance(direction, dict):
            raise ClaudeUnavailable("CONTINUE audit supplied no next direction")
    elif stop_condition not in ("infrastructure_blocker", "scientific_saturation"):
        raise ClaudeUnavailable("STOP audit supplied no valid stop condition")
    return parsed


def parse_decision(result_text: str) -> dict[str, Any]:
    """Parse legacy tagged audit output retained for backward compatibility."""
    start = result_text.rfind(DECISION_OPEN)
    end = result_text.rfind(DECISION_CLOSE)
    if start < 0 or end < 0 or end <= start:
        blob = result_text.strip()
    else:
        blob = result_text[start + len(DECISION_OPEN) : end].strip()
    try:
        parsed: dict[str, Any] = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ClaudeUnavailable(f"audit decision block was not valid JSON: {exc}") from exc
    return validate_decision(parsed)


def decision_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Prefer schema-enforced output; fall back to legacy result text."""
    structured = payload.get("structured_output")
    if structured is None:
        return parse_decision(str(payload.get("result", "")))
    if isinstance(structured, str):
        try:
            structured = json.loads(structured)
        except json.JSONDecodeError as exc:
            raise ClaudeUnavailable(f"structured audit output was not valid JSON: {exc}") from exc
    if not isinstance(structured, dict):
        raise ClaudeUnavailable("structured audit output was not an object")
    return validate_decision(structured)


def run_claude_audit(packet: str, args: argparse.Namespace) -> AuditResult:
    command = [
        "claude",
        "-p",
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(AUDIT_DECISION_SCHEMA, separators=(",", ":")),
        "--model",
        args.audit_model,
        "--effort",
        args.audit_effort,
        "--permission-mode",
        "acceptEdits",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--disable-slash-commands",
        "--allowed-tools",
        "Read",
        "Grep",
        "Glob",
        "--disallowed-tools",
        "Bash",
        "Edit",
        "Write",
        "WebFetch",
        "WebSearch",
        "--max-budget-usd",
        str(args.audit_budget_usd),
    ]
    try:
        completed = run(
            command,
            stdin=packet,
            timeout=args.audit_timeout_seconds,
            check=False,
            env=subscription_env(),
        )
    except subprocess.TimeoutExpired as exc:
        raise ClaudeUnavailable(f"audit timed out after {args.audit_timeout_seconds}s") from exc
    if completed.returncode:
        raise ClaudeUnavailable(
            f"claude exited {completed.returncode}: "
            f"{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ClaudeUnavailable("claude result was not valid JSON") from exc
    if payload.get("is_error") or payload.get("subtype") != "success":
        raise ClaudeUnavailable(f"claude returned an error: {str(payload.get('result'))[:500]}")
    if payload.get("permission_denials"):
        raise ClaudeUnavailable(f"claude hit permission denials: {payload['permission_denials']}")
    parsed = decision_from_payload(payload)
    result_text = str(payload.get("result", "")) or json.dumps(parsed, indent=2)
    return AuditResult(
        decision=parsed["decision"],
        stop_condition=parsed.get("stop_condition"),
        next_direction=parsed.get("next_direction"),
        audit_summary=str(parsed.get("audit_summary", "")),
        raw_result=result_text,
        payload=payload,
        parsed=parsed,
    )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def persist_audit(
    audit: AuditResult,
    *,
    after_experiment: str | None,
    champion: str,
    batch_ids: list[str],
    promotion: bool,
    packet: str,
) -> Path:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    PACKET_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    tag = after_experiment or "pre-first-batch"
    record = {
        "schema_version": 1,
        "completed_at": now(),
        "after_experiment": after_experiment,
        "batch_experiments": batch_ids,
        "champion_commit": champion,
        "promotion_in_batch": promotion,
        "decision": audit.decision,
        "stop_condition": audit.stop_condition,
        "streak_assessment": audit.parsed.get("streak_assessment"),
        "recurring_failure_modes": audit.parsed.get("recurring_failure_modes"),
        "next_direction": audit.next_direction,
        "audit_summary": audit.audit_summary,
        "claude": {
            "session_id": audit.payload.get("session_id"),
            "model": audit.payload.get("modelUsage", {}),
            "total_cost_usd": audit.payload.get("total_cost_usd"),
            "num_turns": audit.payload.get("num_turns"),
        },
        "full_result_text": audit.raw_result,
    }
    audit_path = AUDIT_DIR / f"audit-{stamp}-{tag}.json"
    audit_path.write_text(json.dumps(record, indent=2) + "\n")
    (PACKET_DIR / f"packet-{stamp}-{tag}.md").write_text(packet)
    return audit_path


def write_direction(audit: AuditResult, after_experiment: str | None) -> None:
    direction = audit.next_direction or {}
    lines = [
        "# Next research direction",
        "",
        f"Set by the Claude evidence audit on {now()}"
        + (f" after {after_experiment}." if after_experiment else " (pre-first-batch)."),
        "It steers the next candidate's hypothesis only; the controller and its",
        "protected framework remain the sole evaluation and promotion authority.",
        "",
        f"- Title: {direction.get('title', 'unspecified')}",
        f"- Family: {direction.get('family', 'unspecified')}",
        f"- Hypothesis: {direction.get('hypothesis', 'unspecified')}",
        f"- Rationale: {direction.get('rationale', 'unspecified')}",
        f"- Guardrails: {direction.get('guardrails', 'unspecified')}",
        "",
        "Constraints that always hold: edit only agent.py and/or weights/; the",
        "trained model must keep materially determining leaf evaluation and move",
        "selection; one CPU / 2 GB / no network; do not repeat a saturated family",
        "by only changing a parameter.",
        "",
        f"Audit summary: {audit.audit_summary}",
    ]
    DIRECTION_PATH.write_text("\n".join(lines) + "\n")


def append_log(entry: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as handle:
        handle.write(json.dumps({"at": now(), **entry}) + "\n")


def commit_trail(message: str, *, push: bool) -> None:
    """Commit the research trail. A push failure is logged, never fatal."""
    paths = ["research/audits", "research/audit-input", "research/next-direction.md"]
    existing = [p for p in paths if (ROOT / p).exists()]
    if not existing:
        return
    run(["git", "add", "--", *existing])
    staged = git("diff", "--cached", "--name-only")
    if not staged:
        return
    run(["git", "commit", "-m", message])
    if push:
        pushed = run(["git", "push", "origin", "main"], check=False)
        if pushed.returncode:
            print(f"[{now()}] warning: git push failed (will retry next commit): "
                  f"{pushed.stderr[-500:]}", file=sys.stderr, flush=True)
            append_log({"event": "push_failed", "detail": pushed.stderr[-500:]})
            run(["git", "add", "--", str(LOG_PATH.relative_to(ROOT))])
            run(["git", "commit", "-m", "supervisor: record deferred push failure"])


def record_stop(reason: str, args: argparse.Namespace, *, detail: str | None = None) -> None:
    """Persist the final event so a safe stop never blocks the next preflight."""
    entry: dict[str, Any] = {"event": "stop", "reason": reason}
    if detail:
        entry["detail"] = detail
    append_log(entry)
    commit_trail(
        f"supervisor: record stop ({reason})",
        push=not args.no_push,
    )


# --------------------------------------------------------------------------- #
# Controller batch
# --------------------------------------------------------------------------- #


@dataclass
class BatchOutcome:
    ran_experiments: list[str]
    champion_before: str
    champion_after: str
    promotion: bool
    infra_error: str | None
    stdout: str


def run_controller_batch(iterations: int, args: argparse.Namespace) -> BatchOutcome:
    state_before = load_json(STATE_PATH)
    first = int(state_before["next_experiment"])
    champion_before = state_before["champion_commit"]
    command = [args.python, "controller.py", "--iterations", str(iterations)]
    try:
        completed = run(command, timeout=args.batch_timeout_seconds, check=False)
    except subprocess.TimeoutExpired:
        state_timeout = load_json(STATE_PATH)
        ran_ids = [
            f"exp-{n:04d}" for n in range(first, int(state_timeout["next_experiment"]))
        ]
        return BatchOutcome(
            ran_experiments=ran_ids,
            champion_before=champion_before,
            champion_after=state_timeout["champion_commit"],
            promotion=state_timeout["champion_commit"] != champion_before,
            infra_error=f"controller batch exceeded {args.batch_timeout_seconds}s wall time",
            stdout="",
        )
    state_after = load_json(STATE_PATH)
    last = int(state_after["next_experiment"])
    ran = [f"exp-{number:04d}" for number in range(first, last)]
    champion_after = state_after["champion_commit"]
    infra_error: str | None = None
    if completed.returncode:
        tail = (completed.stderr or completed.stdout)[-1500:]
        infra_error = f"controller exited {completed.returncode}: {tail}"
    return BatchOutcome(
        ran_experiments=ran,
        champion_before=champion_before,
        champion_after=champion_after,
        promotion=champion_after != champion_before,
        infra_error=infra_error,
        stdout=completed.stdout,
    )


# --------------------------------------------------------------------------- #
# Preconditions and loop
# --------------------------------------------------------------------------- #


def preflight(args: argparse.Namespace) -> None:
    if git("branch", "--show-current") != "main":
        raise SupervisorBlocked("supervisor must run from main")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SupervisorBlocked("checkout has tracked modifications; commit or stash first")
    policy = load_json(POLICY_PATH)
    generators = {
        str(policy["candidate_generators"]["primary"]),
        str(policy["candidate_generators"]["secondary"]),
    }
    if generators != {"claude-code"}:
        raise SupervisorBlocked(f"policy is not Claude-only: {sorted(generators)}")
    if policy.get("competition_upload_enabled") is not False:
        raise SupervisorBlocked("competition upload boundary is not disabled")
    auth = run(["gh", "auth", "status", "-h", "github.com"], check=False)
    if auth.returncode:
        raise SupervisorBlocked(f"gh is not authenticated:\n{auth.stdout}\n{auth.stderr}")
    if not Path(args.python).exists() and args.python not in ("python", "python3"):
        raise SupervisorBlocked(f"python interpreter not found: {args.python}")
    # controller.preflight covers the rest (executables, remote, upload boundary).
    controller.preflight()


@dataclass
class CycleResult:
    status: str  # "continue" | "stop" | "dry-run"
    detail: str
    batch: BatchOutcome | None = None


def cycle(args: argparse.Namespace, last_batch: BatchOutcome | None) -> CycleResult:
    """One audit, then (on CONTINUE, unless --dry-run) one bounded batch."""
    state = load_json(STATE_PATH)
    batch_ids = last_batch.ran_experiments if last_batch else []
    champion_before = last_batch.champion_before if last_batch else None
    promotion = last_batch.promotion if last_batch else False
    infra_error = last_batch.infra_error if last_batch else None
    stdout = last_batch.stdout if last_batch else ""
    after_experiment = state["last_completed_experiment"] if last_batch else None

    packet = build_packet(state, batch_ids, champion_before, promotion, stdout, infra_error)
    print(
        f"[{now()}] running evidence audit "
        f"(after {after_experiment or 'no batch yet'}, "
        f"champion {state['champion_commit'][:12]})",
        flush=True,
    )
    audit = run_claude_audit(packet, args)
    audit_path = persist_audit(
        audit,
        after_experiment=after_experiment,
        champion=state["champion_commit"],
        batch_ids=batch_ids,
        promotion=promotion,
        packet=packet,
    )
    stop_tag = f" ({audit.stop_condition})" if audit.stop_condition else ""
    print(
        f"[{now()}] audit -> {audit.decision}{stop_tag}; {audit_path.name}; "
        f"${audit.payload.get('total_cost_usd')}",
        flush=True,
    )
    append_log(
        {
            "event": "audit",
            "after_experiment": after_experiment,
            "decision": audit.decision,
            "stop_condition": audit.stop_condition,
            "audit_path": audit_path.name,
            "next_direction_title": (audit.next_direction or {}).get("title"),
        }
    )

    if audit.decision == "STOP":
        if audit.next_direction:
            write_direction(audit, after_experiment)
        commit_trail(
            f"supervisor: STOP audit after {after_experiment or 'pre-first-batch'}",
            push=not args.no_push,
        )
        reason = audit.stop_condition or "unspecified"
        return CycleResult("stop", f"claude audit returned STOP ({reason})")

    write_direction(audit, after_experiment)
    commit_trail(
        f"supervisor: audit after {after_experiment or 'pre-first-batch'} -> "
        f"{(audit.next_direction or {}).get('title', 'continue')}",
        push=not args.no_push,
    )

    if args.dry_run:
        return CycleResult("dry-run", "audit complete; controller batch skipped")

    print(f"[{now()}] running controller batch (--iterations {args.iterations})", flush=True)
    outcome = run_controller_batch(args.iterations, args)
    print(
        f"[{now()}] batch ran {outcome.ran_experiments or '[]'}; "
        f"promotion={outcome.promotion}; "
        f"infra_error={'yes' if outcome.infra_error else 'no'}",
        flush=True,
    )
    append_log(
        {
            "event": "batch",
            "experiments": outcome.ran_experiments,
            "promotion": outcome.promotion,
            "infra_error": outcome.infra_error,
        }
    )
    return CycleResult("continue", "batch complete", batch=outcome)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations", type=int, default=1,
        help="controller experiments per batch (default 1: audit after every experiment)",
    )
    parser.add_argument(
        "--max-batches", type=int, default=40,
        help="hard backstop on total batches this run (default 40)",
    )
    parser.add_argument(
        "--max-infra-failures", type=int, default=2,
        help="consecutive controller infrastructure failures before stopping",
    )
    parser.add_argument("--audit-model", default="sonnet")
    parser.add_argument("--audit-effort", default="high")
    parser.add_argument("--audit-budget-usd", type=float, default=2.0)
    parser.add_argument("--audit-timeout-seconds", type=int, default=1200)
    parser.add_argument("--batch-timeout-seconds", type=int, default=5400)
    parser.add_argument("--python", default=str(ROOT / ".venv/bin/python"))
    parser.add_argument("--no-push", action="store_true", help="commit the trail but do not push")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="run one audit, persist it, write direction, do not run the controller",
    )
    args = parser.parse_args()
    if args.iterations < 1 or args.max_batches < 1:
        parser.error("--iterations and --max-batches must be positive")

    try:
        preflight(args)
    except (SupervisorBlocked, RuntimeError) as exc:
        print(f"supervisor blocked: {exc}", file=sys.stderr)
        return 2

    append_log({"event": "start", "args": vars(args), "run_id": str(uuid.uuid4())})
    last_batch: BatchOutcome | None = None
    consecutive_infra = 0

    for _ in range(args.max_batches):
        if STOP_FILE.exists():
            print(f"[{now()}] {STOP_FILE.name} present; stopping.", flush=True)
            record_stop("stop_file", args)
            return 0
        try:
            result = cycle(args, last_batch)
        except ClaudeUnavailable as exc:
            print(
                f"[{now()}] Claude unavailable: {exc}. Stopping safely; state preserved.",
                file=sys.stderr,
                flush=True,
            )
            record_stop("claude_unavailable", args, detail=str(exc))
            return 20
        except KeyboardInterrupt:
            print(f"\n[{now()}] interrupted; state preserved.", file=sys.stderr, flush=True)
            record_stop("keyboard_interrupt", args)
            return 130

        if result.status in ("stop", "dry-run"):
            print(f"[{now()}] stopping: {result.detail}", flush=True)
            record_stop(result.detail, args)
            return 0

        last_batch = result.batch
        if last_batch and last_batch.infra_error:
            consecutive_infra += 1
            if consecutive_infra >= args.max_infra_failures:
                print(
                    f"[{now()}] {consecutive_infra} consecutive controller infrastructure "
                    f"failures; stopping. State preserved.",
                    file=sys.stderr,
                    flush=True,
                )
                record_stop("infrastructure", args, detail=last_batch.infra_error)
                return 30
        else:
            consecutive_infra = 0

    print(
        f"[{now()}] reached --max-batches ({args.max_batches}); stopping. Re-run to continue.",
        flush=True,
    )
    record_stop("max_batches", args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
