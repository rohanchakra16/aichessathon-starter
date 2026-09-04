#!/usr/bin/env python3
"""Persistent deterministic controller for internal Chessathon experiments.

AI generators propose candidate submission changes. This controller alone owns
generator scheduling, path protection, evaluation, promotion, journaling, and
the no-upload boundary.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
POLICY_PATH = ROOT / ".autoloop/protected/policy.json"
STATE_PATH = ROOT / ".autoloop/state.json"
WORKTREES = ROOT / ".autoloop/worktrees"
NON_IMPROVEMENT_STATUSES = frozenset({"rejected", "inconclusive"})

# Each candidate generator needs exactly one external executable. Preflight and
# the per-iteration guard use this map so an unused generator is never required.
# ``claude-retrain`` is deterministic (no model call); it only needs ``uv`` to
# run the frozen offline trainer, and ``uv`` is already a common executable.
GENERATOR_EXECUTABLES = {
    "claude-code": "claude",
    "codex-exec": "codex",
    "claude-retrain": "uv",
}
COMMON_EXECUTABLES = ("gh", "git", "uv")

# Deterministic splice markers for the learned-evaluator-retrain candidate path.
RESIDUAL_BEGIN = "# === BEGIN learned positional/endgame residual ==="
RESIDUAL_END = "# === END learned positional/endgame residual ==="
RESIDUAL_CALL = "    score += _positional_residual(board)  # learned residual"
RESIDUAL_MODULE_ANCHOR = "\n\n_deadline = math.inf\n"
RESIDUAL_CALL_ANCHOR = "    return BIAS + score\n"


class InfrastructureError(RuntimeError):
    """External service or host failure; not evidence about candidate quality."""


class CandidateError(RuntimeError):
    """Candidate generation or protected-scope violation."""


def now() -> str:
    return datetime.now(UTC).isoformat()


def run(
    command: list[str],
    *,
    cwd: Path = ROOT,
    timeout: int | None = None,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command, cwd=cwd, text=True, capture_output=True, timeout=timeout, env=env
    )
    if check and completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout[-8000:]}\n{completed.stderr[-8000:]}"
        )
    return completed


def git(*arguments: str, cwd: Path = ROOT, check: bool = True) -> str:
    return run(["git", *arguments], cwd=cwd, check=check).stdout.strip()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def path_allowed(path: str, policy: dict[str, Any]) -> bool:
    for allowed in policy["candidate_allowed_paths"]:
        if allowed.endswith("/") and path.startswith(allowed):
            return True
        if path == allowed:
            return True
    return False


def changed_paths(base: str, head: str) -> list[str]:
    output = git("diff", "--name-only", f"{base}..{head}")
    return [line for line in output.splitlines() if line]


def protected_hash() -> str:
    digest = hashlib.sha256()
    paths = [
        ROOT / ".dockerignore",
        ROOT / "controller.py",
        ROOT / "Makefile",
        *sorted(path for path in (ROOT / ".autoloop/protected").rglob("*") if path.is_file()),
        *sorted(path for path in (ROOT / ".github/workflows").glob("*.yml") if path.is_file()),
        *sorted((ROOT / "harness").glob("*.py")),
        *sorted((ROOT / "training").glob("*.py")),
        *sorted((ROOT / "tests/autoloop").glob("*.py")),
    ]
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def status_paths(worktree: Path) -> list[str]:
    lines = run(["git", "status", "--porcelain"], cwd=worktree).stdout.splitlines()
    paths: list[str] = []
    for line in lines:
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(path)
    return paths


def recent_experiment_records(
    state: dict[str, Any], limit: int
) -> list[dict[str, Any]]:
    """Load a bounded newest-first evidence window for generator coordination."""
    next_number = int(state["next_experiment"])
    records: list[dict[str, Any]] = []
    for number in range(next_number - 1, max(0, next_number - limit - 1), -1):
        path = ROOT / f"experiments/exp-{number:04d}.json"
        if path.exists():
            records.append(load(path))
    return records


def consecutive_non_improvements(records: list[dict[str, Any]]) -> int:
    """Count consecutive scientific non-improvements, ignoring no evidence."""
    count = 0
    for record in records:
        status = record.get("status")
        if status in NON_IMPROVEMENT_STATUSES:
            count += 1
            continue
        if status == "infrastructure_error":
            continue
        break
    return count


def generator_for_stall_count(count: int, policy: dict[str, Any]) -> str:
    """Choose the secondary generator at a frozen, deterministic cadence."""
    settings = policy["candidate_generators"]
    threshold = int(settings["secondary_after_non_improvements"])
    cadence = int(settings["secondary_cadence"])
    if count >= threshold and (count - threshold) % cadence == 0:
        return str(settings["secondary"])
    return str(settings["primary"])


def select_candidate_generator(
    state: dict[str, Any], policy: dict[str, Any]
) -> tuple[str, list[dict[str, Any]]]:
    """Return the scheduled generator and its bounded evidence window."""
    limit = int(policy["candidate_generators"]["recent_experiment_limit"])
    records = recent_experiment_records(state, limit)
    stalled = consecutive_non_improvements(records)
    return generator_for_stall_count(stalled, policy), records


def experiment_digest(records: list[dict[str, Any]]) -> str:
    """Render concise evidence rather than repeatedly feeding full raw journals."""
    if not records:
        return "- No retained experiment summaries are available."
    lines: list[str] = []
    for record in records:
        identifier = str(record.get("id", "unknown"))
        status = str(record.get("status", "unknown"))
        generator = str(record.get("generator", "unknown"))
        hypothesis = str(
            record.get("hypothesis", record.get("generator_summary", ""))
        )
        hypothesis = " ".join(hypothesis.split())[:180]
        reason = str(record.get("decision_reason", record.get("failure", "no reason")))
        reason = " ".join(reason.split())[:180]
        detail = f"; hypothesis: {hypothesis}" if hypothesis else ""
        lines.append(f"- {identifier}: {status}; {generator}{detail}; {reason}")
    return "\n".join(lines)


def bounded_generator_summary(value: Any, limit: int = 600) -> str:
    """Keep a concise, journal-safe generator explanation for later coordination."""
    summary = " ".join(str(value or "").split())
    marker = summary.rfind("HYPOTHESIS:")
    if marker >= 0:
        summary = summary[marker:]
    return summary[:limit]


def supervisor_directive(worktree: Path) -> str:
    """Optional research direction from the external Claude evidence-audit loop.

    ``claude_supervisor.py`` writes ``research/next-direction.md`` after each
    between-batch audit. It only steers which hypothesis the candidate explores;
    evaluation, reliability gating and promotion stay entirely with this
    controller and the protected framework. Absent or empty file: no directive.
    """
    path = worktree / "research/next-direction.md"
    if not path.exists():
        return ""
    text = path.read_text().strip()
    if not text:
        return ""
    return (
        "\n\nBetween-batch supervising-researcher directive (from the Claude "
        "evidence audit). Pursue this research direction unless the evidence "
        "digest already shows it exhausted. It does not change evaluation, "
        "reliability or promotion, which remain the controller's:\n"
        f"{text[:6000]}\n"
    )


def candidate_prompt(
    worktree: Path,
    experiment_id: str,
    generator: str,
    records: list[dict[str, Any]],
) -> str:
    champion = git("rev-parse", "HEAD", cwd=worktree)
    isolated_root = worktree.resolve()
    return f"""You are the {generator} candidate engineer for {experiment_id} in the
AI Chessathon internal optimizer. The frozen champion is {champion}.

Your only permitted filesystem root is {isolated_root}. This is the isolated
candidate worktree. Use paths relative to the current working directory. Never
read or write the parent/main checkout, another worktree, or any absolute path
outside this root. Read AGENTS.md, agent.py, weights/model.json, and
.autoloop/protected/policy.json from this worktree only. Do not fetch the URLs
mentioned in AGENTS.md; the frozen protected policy is authoritative here.

Start from the current champion and make exactly one focused, reversible
strength or reliability improvement. You may edit only agent.py or files under
weights/.
Do not edit the harness, tests, workflows, controller, acceptance criteria,
experiment state, training code, documentation, or Git metadata. Do not commit.

Recent evidence digest (newest first):
{experiment_digest(records)}

Use the digest to avoid duplicating rejected or inconclusive ideas. Treat an
unsuccessful mechanism family as exhausted: changing only its cap, threshold,
depth, or other parameter is not a new hypothesis. Before editing, compare your
mechanism with every digest line and choose a materially different direction.
Inspect only the specific retained experiment records or losing-game evidence
needed for that hypothesis. Do not perform broad parameter sweeps manually.
{supervisor_directive(worktree)}
Hard requirements: get_move must always return a legal UCI move under the real
clock; one CPU, 2 GB RAM, no network/GPU; readable source; no existing engine or
wrapper. The repository-trained model must continue to materially determine
leaf evaluation and move selection. The deterministic controller will run all
tests and benchmarks after you finish.

In your final response, state the implemented mechanism and intended benefit in
one concise sentence beginning with `HYPOTHESIS:`. Do not paste logs."""


def codex_generate(worktree: Path, prompt: str, policy: dict[str, Any]) -> dict[str, Any]:
    completed = run(
        [
            "codex",
            "exec",
            "--ephemeral",
            "--sandbox",
            "workspace-write",
            "--ignore-rules",
            prompt,
        ],
        cwd=worktree,
        timeout=policy["candidate_timeout_seconds"],
        check=False,
    )
    if completed.returncode:
        raise CandidateError(f"Codex failed: {completed.stderr[-5000:]}")
    return {
        "generator": "codex-exec",
        "generator_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "generator_summary": bounded_generator_summary(completed.stdout),
    }


def claude_command(prompt: str, policy: dict[str, Any]) -> list[str]:
    """Build the no-shell, no-web, non-persistent Claude invocation."""
    settings = policy["candidate_generators"]
    return [
        "claude",
        "-p",
        "--safe-mode",
        "--no-chrome",
        "--no-session-persistence",
        "--disable-slash-commands",
        "--strict-mcp-config",
        "--permission-mode",
        "acceptEdits",
        "--tools",
        "Read",
        "Edit",
        "Write",
        "Glob",
        "Grep",
        "--disallowed-tools",
        "Bash",
        "WebFetch",
        "WebSearch",
        "--model",
        str(settings["claude_model"]),
        "--effort",
        str(settings["claude_effort"]),
        "--max-budget-usd",
        str(settings["claude_max_budget_usd"]),
        "--output-format",
        "json",
        prompt,
    ]


def claude_generate(worktree: Path, prompt: str, policy: dict[str, Any]) -> dict[str, Any]:
    environment = os.environ.copy()
    for variable in (
        "ANTHROPIC_API_KEY",
        "CODEX_API_KEY",
        "OPENAI_API_KEY",
        "GH_TOKEN",
        "GITHUB_TOKEN",
    ):
        environment.pop(variable, None)
    authentication = run(
        ["claude", "auth", "status"], check=False, env=environment
    )
    try:
        auth_payload = json.loads(authentication.stdout)
    except json.JSONDecodeError as exc:
        raise CandidateError("Claude authentication status was not valid JSON") from exc
    if authentication.returncode or not auth_payload.get("loggedIn"):
        raise CandidateError("Claude Code is not authenticated")
    version = run(["claude", "--version"], check=False, env=environment)
    if version.returncode:
        raise CandidateError(f"Claude version check failed: {version.stderr[-2000:]}")
    completed = run(
        claude_command(prompt, policy),
        cwd=worktree,
        timeout=policy["candidate_timeout_seconds"],
        check=False,
        env=environment,
    )
    if completed.returncode:
        raise CandidateError(
            f"Claude failed: {completed.stdout[-3000:]}\n{completed.stderr[-3000:]}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise CandidateError("Claude result was not valid JSON") from exc
    if payload.get("is_error") or payload.get("subtype") != "success":
        raise CandidateError(f"Claude returned an error: {payload.get('result', '')}")
    denials = payload.get("permission_denials", [])
    if denials:
        raise CandidateError(f"Claude attempted disallowed operations: {denials}")
    model_usage = payload.get("modelUsage", {})
    models = sorted(model_usage) if isinstance(model_usage, dict) else []
    return {
        "generator": "claude-code",
        "generator_version": version.stdout.strip(),
        "generator_models": models,
        "generator_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "generator_summary": bounded_generator_summary(payload.get("result", "")),
        "generator_usage": {
            "duration_ms": payload.get("duration_ms"),
            "num_turns": payload.get("num_turns"),
            "total_cost_usd": payload.get("total_cost_usd"),
        },
    }


def generator_executable(generator: str) -> str:
    """Return the single external executable a candidate generator requires."""
    try:
        return GENERATOR_EXECUTABLES[generator]
    except KeyError:
        raise CandidateError(f"unknown candidate generator: {generator}") from None


def configured_generators(policy: dict[str, Any]) -> set[str]:
    """Generators the frozen policy schedule can actually select."""
    settings = policy["candidate_generators"]
    return {str(settings["primary"]), str(settings["secondary"])}


def generate_candidate(
    worktree: Path,
    experiment_id: str,
    policy: dict[str, Any],
    generator: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    executable = generator_executable(generator)
    if shutil.which(executable) is None:
        raise InfrastructureError(
            f"selected candidate generator {generator!r} is unavailable: "
            f"required executable {executable!r} is not installed"
        )
    prompt = candidate_prompt(worktree, experiment_id, generator, records)
    if generator == "codex-exec":
        metadata = codex_generate(worktree, prompt, policy)
    elif generator == "claude-code":
        metadata = claude_generate(worktree, prompt, policy)
    else:
        raise CandidateError(f"unknown candidate generator: {generator}")
    paths = status_paths(worktree)
    if not paths:
        raise CandidateError(f"{generator} produced no change")
    illegal = [path for path in paths if not path_allowed(path, policy)]
    if illegal:
        raise CandidateError(f"candidate changed protected/disallowed paths: {illegal}")
    git("add", "--", *paths, cwd=worktree)
    staged = git("diff", "--cached", "--name-only", cwd=worktree).splitlines()
    if not staged:
        raise CandidateError("candidate produced no stageable submission change")
    git("commit", "-m", f"experiment {experiment_id}: AI candidate", cwd=worktree)
    metadata["candidate_commit"] = git("rev-parse", "HEAD", cwd=worktree)
    return metadata


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def splice_residual_block(source: str, feature_source: str, glue_source: str) -> str:
    """Insert (or replace in place) the learned-residual block in agent.py.

    Pure text transform, no model call: the feature maths comes verbatim from
    ``training/train_positional_evaluator.AGENT_FEATURE_SOURCE`` and the same
    string builds the trainer's design matrix, so the shipped features and the
    fitted coefficients cannot drift.
    """
    if RESIDUAL_BEGIN in source:
        start = source.index(RESIDUAL_BEGIN)
        end = source.index("\n", source.index(RESIDUAL_END) + len(RESIDUAL_END)) + 1
        source = source[:start].rstrip("\n") + "\n\n" + source[end:].lstrip("\n")
    source = source.replace(RESIDUAL_CALL + "\n", "")

    if source.count(RESIDUAL_MODULE_ANCHOR) != 1:
        raise CandidateError("agent.py module anchor for the residual block is not unique")
    if source.count(RESIDUAL_CALL_ANCHOR) != 1:
        raise CandidateError("agent.py evaluate-return anchor is not unique")

    block = (
        f"{RESIDUAL_BEGIN}\n"
        f"{feature_source.strip()}\n\n\n"
        f"{glue_source.strip()}\n"
        f"{RESIDUAL_END}\n"
    )
    source = source.replace(
        RESIDUAL_MODULE_ANCHOR, f"\n\n{block}\n_deadline = math.inf\n", 1
    )
    source = source.replace(
        RESIDUAL_CALL_ANCHOR, f"{RESIDUAL_CALL}\n{RESIDUAL_CALL_ANCHOR}", 1
    )
    return source


def retrain_generate(
    worktree: Path,
    experiment_id: str,
    policy: dict[str, Any],
    entrypoint: str,
) -> dict[str, Any]:
    """Deterministic learned-evaluator-retrain candidate.

    Splices the frozen feature block into agent.py, runs the whitelisted offline
    trainer to refit only the bounded residual coefficients in
    weights/model.json, and commits {agent.py, weights/model.json}. Evaluation,
    ablation, reliability, arena and promotion downstream are unchanged.
    """
    retrain = policy.get("retrain")
    if not isinstance(retrain, dict) or not retrain.get("enabled"):
        raise CandidateError("policy.retrain is absent or disabled")
    if entrypoint not in retrain.get("allowed_entrypoints", []):
        raise CandidateError(f"retrain entrypoint {entrypoint!r} is not whitelisted")

    entrypoint_path = worktree / entrypoint
    if not entrypoint_path.is_file():
        raise CandidateError(f"retrain entrypoint {entrypoint!r} is missing in the worktree")

    datasets: dict[str, Path] = {}
    for name, spec in retrain["datasets"].items():
        if not spec.get("sha256"):
            raise InfrastructureError(
                f"retrain {name} dataset sha256 is not pinned in policy.retrain.datasets; "
                "generate the datasets and pin their sha256 before running a retrain"
            )
        dataset_path = worktree / spec["path"]
        if not dataset_path.is_file():
            raise InfrastructureError(f"retrain {name} dataset missing: {spec['path']}")
        actual = file_sha256(dataset_path)
        if actual != spec["sha256"]:
            raise InfrastructureError(
                f"retrain {name} dataset sha256 mismatch for {spec['path']}: "
                f"{actual} != pinned {spec['sha256']}"
            )
        datasets[name] = dataset_path

    trainer_module = import_trainer(entrypoint_path)
    agent_path = worktree / "agent.py"
    agent_path.write_text(
        splice_residual_block(
            agent_path.read_text(),
            trainer_module.AGENT_FEATURE_SOURCE,
            trainer_module.AGENT_RESIDUAL_GLUE,
        )
    )

    trainer = run(
        [
            "uv",
            "run",
            "python",
            entrypoint,
            "--base-model",
            "weights/model.json",
            "--output",
            "weights/model.json",
            "--training-dataset",
            str(datasets["training"].relative_to(worktree)),
            "--validation-dataset",
            str(datasets["validation"].relative_to(worktree)),
            "--seed",
            str(int(retrain["seed"])),
        ],
        cwd=worktree,
        timeout=int(retrain["timeout_seconds"]),
        check=False,
    )
    if trainer.returncode:
        raise CandidateError(
            f"offline retrain failed: {trainer.stdout[-3000:]}\n{trainer.stderr[-3000:]}"
        )

    paths = status_paths(worktree)
    permitted = {"agent.py", "weights/model.json"}
    unexpected = [path for path in paths if path not in permitted]
    if unexpected:
        raise CandidateError(f"retrain touched unexpected paths: {unexpected}")
    if not paths:
        raise CandidateError("retrain produced no change")

    model = json.loads((worktree / "weights/model.json").read_text())
    if model.get("layout", {}).get("positional_offset") != 770:
        raise CandidateError("retrained model.json is missing the positional residual layout")

    git("add", "--", *paths, cwd=worktree)
    git("commit", "-m", f"experiment {experiment_id}: retrained evaluator", cwd=worktree)
    return {
        "generator": "claude-retrain",
        "retrain_entrypoint": entrypoint,
        "retrain_entrypoint_sha256": file_sha256(entrypoint_path),
        "retrain_seed": int(retrain["seed"]),
        "retrain_datasets": {
            name: {"path": spec["path"], "sha256": spec["sha256"]}
            for name, spec in retrain["datasets"].items()
        },
        "residual_coefficients": model.get("residual_coefficients"),
        "selected_cross_validation": model.get("selected_cross_validation"),
        "independent_validation": model.get("independent_validation"),
        "model_kind": model.get("model_kind"),
        "candidate_commit": git("rev-parse", "HEAD", cwd=worktree),
    }


def import_trainer(entrypoint_path: Path) -> Any:
    """Import a whitelisted trainer module by path to read its feature source."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_retrain_trainer", entrypoint_path)
    if spec is None or spec.loader is None:
        raise CandidateError(f"cannot import trainer at {entrypoint_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for attribute in ("AGENT_FEATURE_SOURCE", "AGENT_RESIDUAL_GLUE"):
        if not isinstance(getattr(module, attribute, None), str):
            raise CandidateError(f"trainer is missing string attribute {attribute!r}")
    return module


def github_evaluate(
    worktree: Path,
    branch: str,
    candidate_sha: str,
    timeout: int,
    repository: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pushed_at = time.monotonic()
    push = run(
        ["git", "push", "--set-upstream", "origin", branch],
        cwd=worktree,
        check=False,
    )
    if push.returncode:
        raise InfrastructureError(f"candidate push failed: {push.stderr[-4000:]}")
    deadline = time.monotonic() + timeout
    run_id: int | None = None
    while time.monotonic() < deadline:
        lookup = run(
            [
                "gh",
                "run",
                "list",
                "--workflow",
                "candidate-evaluate.yml",
                "--branch",
                branch,
                "--commit",
                candidate_sha,
                "--repo",
                repository,
                "--limit",
                "1",
                "--json",
                "databaseId,status,conclusion,createdAt,updatedAt,url",
            ],
            check=False,
        )
        if lookup.returncode == 0:
            rows = json.loads(lookup.stdout or "[]")
            if rows:
                row = rows[0]
                run_id = int(row["databaseId"])
                if row["status"] == "completed":
                    artifact_dir = worktree / ".autoloop-artifact"
                    artifact_dir.mkdir(exist_ok=True)
                    download = run(
                        [
                            "gh",
                            "run",
                            "download",
                            str(run_id),
                            "-n",
                            "evaluation",
                            "-D",
                            str(artifact_dir),
                            "--repo",
                            repository,
                        ],
                        check=False,
                    )
                    if download.returncode:
                        raise InfrastructureError(
                            f"evaluation artifact download failed: {download.stderr[-4000:]}"
                        )
                    evaluation = load(artifact_dir / "evaluation.json")
                    arena_path = artifact_dir / "arena.json"
                    if arena_path.exists():
                        evaluation["paired_arena"] = load(arena_path)
                    return evaluation, {
                        "run_id": run_id,
                        "run_url": row["url"],
                        "workflow_conclusion": row["conclusion"],
                        "wait_seconds": round(time.monotonic() - pushed_at, 3),
                    }
        time.sleep(5)
    raise InfrastructureError(f"GitHub evaluation timed out; last run id: {run_id}")


def github_release_evaluate(
    commit: str, policy: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    repository = policy["github_repository"]
    triggered_at = time.monotonic()
    dispatch = run(
        [
            "gh",
            "workflow",
            "run",
            "release-evaluate.yml",
            "--ref",
            "main",
            "--repo",
            repository,
        ],
        check=False,
    )
    if dispatch.returncode:
        raise InfrastructureError(f"release workflow dispatch failed: {dispatch.stderr[-4000:]}")
    deadline = time.monotonic() + int(policy["release_timeout_seconds"])
    run_id: int | None = None
    while time.monotonic() < deadline:
        lookup = run(
            [
                "gh",
                "run",
                "list",
                "--workflow",
                "release-evaluate.yml",
                "--branch",
                "main",
                "--commit",
                commit,
                "--event",
                "workflow_dispatch",
                "--repo",
                repository,
                "--limit",
                "1",
                "--json",
                "databaseId,status,conclusion,createdAt,updatedAt,url",
            ],
            check=False,
        )
        if lookup.returncode == 0:
            rows = json.loads(lookup.stdout or "[]")
            if rows:
                row = rows[0]
                run_id = int(row["databaseId"])
                if row["status"] == "completed":
                    artifact_dir = ROOT / f".autoloop/artifacts/release-{run_id}"
                    artifact_dir.mkdir(parents=True, exist_ok=False)
                    download = run(
                        [
                            "gh",
                            "run",
                            "download",
                            str(run_id),
                            "-n",
                            "release-evaluation",
                            "-D",
                            str(artifact_dir),
                            "--repo",
                            repository,
                        ],
                        check=False,
                    )
                    if download.returncode:
                        raise InfrastructureError(
                            f"release artifact download failed: {download.stderr[-4000:]}"
                        )
                    return load(artifact_dir / "release-evaluation.json"), {
                        "run_id": run_id,
                        "run_url": row["url"],
                        "workflow_conclusion": row["conclusion"],
                        "wait_seconds": round(time.monotonic() - triggered_at, 3),
                    }
        time.sleep(5)
    raise InfrastructureError(f"release evaluation timed out; last run id: {run_id}")


def release_check() -> bool:
    policy = load(POLICY_PATH)
    state = load(STATE_PATH)
    commit = git("rev-parse", "HEAD")
    evaluation, workflow = github_release_evaluate(commit, policy)
    record = {
        "schema_version": 1,
        "evaluated_main_commit": commit,
        "champion_commit": state["champion_commit"],
        "completed_at": now(),
        "protected_hash": protected_hash(),
        "workflow": workflow,
        "evaluation": evaluation,
        "status": "passed" if evaluation.get("passed") else "failed",
    }
    release_path = ROOT / f"releases/{commit[:12]}-{workflow['run_id']}.json"
    atomic_json(release_path, record)
    if evaluation.get("passed"):
        state["submission_candidate"] = {
            "champion_commit": state["champion_commit"],
            "evaluated_main_commit": commit,
            "release_record": str(release_path.relative_to(ROOT)),
            "artifact_sha256": evaluation["package"]["sha256"],
            "validated_at": now(),
        }
    atomic_json(STATE_PATH, state)
    git("add", str(release_path.relative_to(ROOT)), ".autoloop/state.json")
    git("commit", "-m", f"record release evaluation: {record['status']}")
    git("push", "origin", "main")
    print(
        f"release evaluation {record['status']}: {workflow['run_url']}",
        flush=True,
    )
    return bool(evaluation.get("passed"))


def arena(worktree: Path, policy: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    settings = policy["arena"]
    environment = os.environ.copy()
    environment.pop("CODEX_API_KEY", None)
    environment.pop("OPENAI_API_KEY", None)
    output = ROOT / f".autoloop/artifacts/{experiment_id}-arena.json"
    completed = run(
        [
            "uv",
            "run",
            "python",
            ".autoloop/protected/arena.py",
            "--candidate",
            str(worktree),
            "--champion",
            str(ROOT),
            "--policy",
            str(POLICY_PATH),
            "--output",
            str(output),
        ],
        timeout=max(300, settings["games"] * 30),
        check=False,
        env=environment,
    )
    if completed.returncode != 0 or not output.exists():
        return {
            "passed": False,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-16000:],
            "stderr": completed.stderr[-8000:],
            "settings": settings,
        }
    return load(output)


def decide(ci: dict[str, Any], match: dict[str, Any], policy: dict[str, Any]) -> tuple[str, str]:
    if not ci.get("passed"):
        return "rejected", "protected compliance/correctness evaluation failed"
    if not match.get("passed"):
        return "rejected", "candidate failed to complete the paired arena"
    score = match.get("score")
    decision = match.get("statistical_decision")
    interval = match.get("confidence_interval", {})
    if not isinstance(score, (float, int)):
        return "rejected", "arena produced no numeric score"
    evidence = (
        f"score {float(score):.3f}, interval "
        f"[{float(interval.get('lower', 0.0)):.3f}, "
        f"{float(interval.get('upper', 1.0)):.3f}]"
    )
    if decision == "accept":
        return "accepted", f"sequential paired arena accepted candidate ({evidence})"
    if decision == "reject":
        return "rejected", f"sequential paired arena rejected candidate ({evidence})"
    return "inconclusive", f"sequential paired arena remained inconclusive ({evidence})"


def clock_sensitive_decide(
    experiment: dict[str, Any], match: dict[str, Any], policy: dict[str, Any]
) -> tuple[str, str]:
    """Apply the frozen secondary gate for changes that activate only at real clocks."""
    settings = policy["clock_sensitive_promotion"]
    if settings["require_ci_passed"] and not experiment.get("ci", {}).get("passed"):
        return "rejected", "protected compliance/correctness evaluation failed"
    fast_decision = experiment.get("arena", {}).get("statistical_decision")
    if fast_decision not in settings["allowed_fast_arena_decisions"]:
        return "rejected", f"fast arena decision {fast_decision!r} is not eligible"
    if not match.get("passed"):
        return "rejected", "prospective real-clock arena had an agent failure"
    decision = match.get("statistical_decision")
    score = float(match.get("score", 0.0))
    interval = match.get("confidence_interval", {})
    evidence = (
        f"score {score:.3f}, interval "
        f"[{float(interval.get('lower', 0.0)):.3f}, "
        f"{float(interval.get('upper', 1.0)):.3f}]"
    )
    if decision == settings["required_real_clock_decision"]:
        return "accepted", f"prospective real-clock arena accepted candidate ({evidence})"
    if decision == "reject":
        return "rejected", f"prospective real-clock arena rejected candidate ({evidence})"
    return "inconclusive", f"prospective real-clock arena remained inconclusive ({evidence})"


def clock_promotion(experiment_id: str) -> None:
    """Evaluate and deterministically promote/reject one retained clock-sensitive candidate."""
    if not experiment_id.startswith("exp-") or not experiment_id[4:].isdigit():
        raise ValueError("clock-promotion requires an experiment id such as exp-0032")
    policy = load(POLICY_PATH)
    state = load(STATE_PATH)
    experiment_path = ROOT / f"experiments/{experiment_id}.json"
    experiment = load(experiment_path)
    if experiment.get("id") != experiment_id:
        raise ValueError(f"experiment record id mismatch: {experiment_id}")
    candidate_sha = str(experiment["candidate_commit"])
    champion_sha = str(state["champion_commit"])
    illegal = [
        path
        for path in changed_paths(f"{candidate_sha}^", candidate_sha)
        if not path_allowed(path, policy)
    ]
    if illegal:
        raise CandidateError(f"clock candidate changed disallowed paths: {illegal}")

    candidate_tree = WORKTREES / f"clock-{experiment_id}-candidate"
    champion_tree = WORKTREES / f"clock-{experiment_id}-champion"
    if candidate_tree.exists() or champion_tree.exists():
        raise InfrastructureError("clock-promotion worktree already exists")
    output = ROOT / f"confirmations/{experiment_id}-prospective-real-clock.json"
    if output.exists():
        raise InfrastructureError(f"prospective evidence already exists: {output}")
    settings_key = str(policy["clock_sensitive_promotion"]["real_clock_settings_key"])
    settings = policy[settings_key]
    try:
        git("worktree", "add", "--detach", str(candidate_tree), candidate_sha)
        git("worktree", "add", "--detach", str(champion_tree), champion_sha)
        environment = os.environ.copy()
        environment.pop("CODEX_API_KEY", None)
        environment.pop("OPENAI_API_KEY", None)
        completed = run(
            [
                sys.executable,
                ".autoloop/protected/arena.py",
                "--candidate",
                str(candidate_tree),
                "--champion",
                str(champion_tree),
                "--policy",
                str(POLICY_PATH),
                "--settings-key",
                settings_key,
                "--output",
                str(output),
            ],
            timeout=max(1200, int(settings["games"]) * 300),
            check=False,
            env=environment,
        )
        if completed.returncode or not output.exists():
            raise InfrastructureError(
                "prospective real-clock arena failed: "
                f"{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}"
            )
        match = load(output)
        status, reason = clock_sensitive_decide(experiment, match, policy)
        promotion_record: dict[str, Any] = {
            "status": status,
            "decision_reason": reason,
            "candidate_commit": candidate_sha,
            "champion_commit": champion_sha,
            "protected_hash": protected_hash(),
            "settings_key": settings_key,
            "evidence": str(output.relative_to(ROOT)),
            "completed_at": now(),
        }
        if status == "accepted":
            git("cherry-pick", candidate_sha)
            promoted_sha = git("rev-parse", "HEAD")
            state["champion_commit"] = promoted_sha
            state["submission_candidate"] = None
            promotion_record["promoted_commit"] = promoted_sha
        experiment["clock_sensitive_promotion"] = promotion_record
        experiment["status"] = status
        experiment["decision_reason"] = reason
        atomic_json(experiment_path, experiment)
        atomic_json(STATE_PATH, state)
        git(
            "add",
            str(experiment_path.relative_to(ROOT)),
            str(output.relative_to(ROOT)),
            ".autoloop/state.json",
        )
        git("commit", "-m", f"record clock promotion {experiment_id}: {status}")
        git("push", "origin", "main")
        print(f"{experiment_id} clock promotion: {status}: {reason}", flush=True)
    finally:
        if candidate_tree.exists():
            git("worktree", "remove", "--force", str(candidate_tree), check=False)
        if champion_tree.exists():
            git("worktree", "remove", "--force", str(champion_tree), check=False)


def persist(
    experiment_id: str,
    record: dict[str, Any],
    state: dict[str, Any],
    status: str,
    candidate_sha: str | None,
) -> None:
    if status == "accepted" and candidate_sha is not None:
        git("merge", "--ff-only", candidate_sha)
        state["champion_commit"] = candidate_sha
        state["submission_candidate"] = None
    state["last_completed_experiment"] = experiment_id
    state["next_experiment"] += 1
    record["status"] = status
    record["completed_at"] = now()
    atomic_json(ROOT / f"experiments/{experiment_id}.json", record)
    atomic_json(STATE_PATH, state)
    git("add", ".autoloop/state.json", f"experiments/{experiment_id}.json")
    git("commit", "-m", f"record experiment {experiment_id}: {status}")


def one_iteration(args: argparse.Namespace) -> bool:
    policy = load(POLICY_PATH)
    state = load(STATE_PATH)
    number = int(state["next_experiment"])
    experiment_id = f"exp-{number:04d}"
    branch = f"autoloop/candidate-{number:04d}"
    worktree = WORKTREES / experiment_id
    record: dict[str, Any] = {
        "schema_version": 1,
        "id": experiment_id,
        "branch": branch,
        "started_at": now(),
        "status": "running",
        "protected_hash": protected_hash(),
    }
    candidate_sha: str | None = None
    try:
        champion = git("rev-parse", "HEAD")
        state["champion_commit"] = state["champion_commit"] or champion
        show_ref = run(
            ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
            check=False,
        )
        if show_ref.returncode == 0:
            git("worktree", "add", str(worktree), branch)
            resumed_sha = git("rev-parse", "HEAD", cwd=worktree)
            if resumed_sha == champion:
                raise InfrastructureError(
                    f"interrupted branch contains no candidate commit: {branch}"
                )
            proposal = {
                "generator": "resumed-candidate",
                "candidate_commit": resumed_sha,
            }
            record["resumed_after_interruption"] = True
        elif getattr(args, "retrain_entrypoint", None):
            git("worktree", "add", "-b", branch, str(worktree), champion)
            _, recent_records = select_candidate_generator(state, policy)
            record["generator"] = "claude-retrain"
            record["consecutive_non_improvements"] = consecutive_non_improvements(
                recent_records
            )
            proposal = retrain_generate(
                worktree, experiment_id, policy, args.retrain_entrypoint
            )
        else:
            git("worktree", "add", "-b", branch, str(worktree), champion)
            generator, recent_records = select_candidate_generator(state, policy)
            record["generator"] = generator
            record["consecutive_non_improvements"] = consecutive_non_improvements(
                recent_records
            )
            proposal = generate_candidate(
                worktree,
                experiment_id,
                policy,
                generator,
                recent_records,
            )
        record.update(proposal)
        candidate_sha = proposal["candidate_commit"]
        illegal = [
            path
            for path in changed_paths(champion, candidate_sha)
            if not path_allowed(path, policy)
        ]
        if illegal:
            raise CandidateError(f"committed candidate changed disallowed paths: {illegal}")
        ci, workflow = github_evaluate(
            worktree,
            branch,
            candidate_sha,
            policy["ci_timeout_seconds"],
            policy["github_repository"],
        )
        record["ci"] = ci
        record["workflow"] = workflow
        match = ci.get("paired_arena", {"passed": False, "skipped": True})
        record["arena"] = match
        status, reason = decide(ci, match, policy)
        record["decision_reason"] = reason
        persist(experiment_id, record, state, status, candidate_sha)
        git("push", "origin", "main")
        print(f"{experiment_id}: {status}: {reason}", flush=True)
        return True
    except InfrastructureError as exc:
        record["failure"] = f"{type(exc).__name__}: {exc}"
        persist(experiment_id, record, state, "infrastructure_error", None)
        git("push", "origin", "main", check=False)
        print(f"{experiment_id}: infrastructure_error: {exc}", file=sys.stderr, flush=True)
        return False
    except Exception as exc:
        record["failure"] = f"{type(exc).__name__}: {exc}"
        persist(experiment_id, record, state, "failed", None)
        git("push", "origin", "main", check=False)
        print(f"{experiment_id}: failed: {exc}", file=sys.stderr, flush=True)
        return True
    finally:
        if worktree.exists() and not args.keep_worktrees:
            git("worktree", "remove", "--force", str(worktree), check=False)


def preflight() -> None:
    policy = load(POLICY_PATH)
    required = list(COMMON_EXECUTABLES)
    generators = configured_generators(policy)
    # Only require a generator's executable when the frozen schedule can select
    # nothing else. A mixed schedule verifies the generator each experiment
    # actually selects in generate_candidate, so a missing alternate generator is
    # reported as that generator being unavailable rather than blocking preflight.
    if len(generators) == 1:
        required.append(generator_executable(next(iter(generators))))
    for executable in required:
        if shutil.which(executable) is None:
            raise RuntimeError(f"required executable not found: {executable}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("controller checkout must have no tracked modifications")
    if git("branch", "--show-current") != "main":
        raise RuntimeError("controller must run from main")
    run(["gh", "auth", "status", "-h", "github.com"])
    if not git("remote", "get-url", "origin"):
        raise RuntimeError("origin remote is required")
    if policy.get("competition_upload_enabled") is not False:
        raise RuntimeError("competition upload boundary is not disabled")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--keep-worktrees", action="store_true")
    parser.add_argument("--release-check", action="store_true")
    parser.add_argument("--clock-promotion")
    parser.add_argument(
        "--retrain-entrypoint",
        help="run learned-evaluator-retrain candidates with this whitelisted trainer",
    )
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.retrain_entrypoint:
        retrain = load(POLICY_PATH).get("retrain")
        if not isinstance(retrain, dict) or not retrain.get("enabled"):
            parser.error("policy.retrain is absent or disabled")
        if args.retrain_entrypoint not in retrain.get("allowed_entrypoints", []):
            parser.error(f"retrain entrypoint {args.retrain_entrypoint!r} is not whitelisted")
    lock_path = ROOT / ".autoloop/controller.lock"
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another controller instance is already running") from exc
        preflight()
        if args.release_check:
            return 0 if release_check() else 1
        if args.clock_promotion:
            clock_promotion(args.clock_promotion)
            return 0
        if args.continuous:
            stop_path = ROOT / ".autoloop/controller.stop"
            while not stop_path.exists():
                if not one_iteration(args):
                    return 1
        else:
            for _ in range(args.iterations):
                if not one_iteration(args):
                    return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"controller stopped: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error
