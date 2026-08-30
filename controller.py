#!/usr/bin/env python3
"""Persistent deterministic controller for internal Chessathon experiments.

Codex proposes candidate submission changes. This controller alone owns path
protection, evaluation, promotion, journaling, and the no-upload boundary.
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
        POLICY_PATH,
        ROOT / ".autoloop/protected/evaluate.py",
        ROOT / ".autoloop/protected/arena.py",
        ROOT / ".autoloop/protected/openings.json",
        ROOT / ".github/workflows/candidate-evaluate.yml",
        ROOT / "controller.py",
        *sorted((ROOT / "harness").glob("*.py")),
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


def generate_candidate(
    worktree: Path, experiment_id: str, policy: dict[str, Any]
) -> dict[str, str]:
    prompt = f"""You are proposing {experiment_id} for the AI Chessathon internal optimizer.
Read AGENTS.md and the protected policy. Make one focused strength or reliability
improvement to the submission. You may edit only agent.py, requirements.txt, or
files under weights/. Do not edit the harness, tests, workflow, controller,
acceptance criteria, experiment state, training code, or documentation.

Hard requirements: get_move must always return a legal UCI move under the real
clock; one CPU, 2 GB RAM, no network/GPU; readable source; no existing engine or
wrapper. The repository-trained model must continue to materially determine
leaf evaluation and move selection. Prefer a small reversible experiment. Run
the focused local tests you need, but do not commit and do not create notes."""
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
    paths = status_paths(worktree)
    if not paths:
        raise CandidateError("Codex produced no change")
    illegal = [path for path in paths if not path_allowed(path, policy)]
    if illegal:
        raise CandidateError(f"candidate changed protected/disallowed paths: {illegal}")
    git("add", "--", *paths, cwd=worktree)
    staged = git("diff", "--cached", "--name-only", cwd=worktree).splitlines()
    if not staged:
        raise CandidateError("candidate produced no stageable submission change")
    git("commit", "-m", f"experiment {experiment_id}: AI candidate", cwd=worktree)
    return {
        "generator": "codex-exec",
        "candidate_commit": git("rev-parse", "HEAD", cwd=worktree),
    }


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
                    return load(artifact_dir / "evaluation.json"), {
                        "run_id": run_id,
                        "run_url": row["url"],
                        "workflow_conclusion": row["conclusion"],
                        "wait_seconds": round(time.monotonic() - pushed_at, 3),
                    }
        time.sleep(5)
    raise InfrastructureError(f"GitHub evaluation timed out; last run id: {run_id}")


def arena(
    worktree: Path, policy: dict[str, Any], experiment_id: str
) -> dict[str, Any]:
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


def decide(
    ci: dict[str, Any], match: dict[str, Any], policy: dict[str, Any]
) -> tuple[str, str]:
    if not ci.get("passed"):
        return "rejected", "protected compliance/correctness evaluation failed"
    if not match.get("passed"):
        return "rejected", "candidate failed to complete the paired arena"
    score = match.get("score")
    threshold = policy["arena"]["minimum_score"]
    if not isinstance(score, float):
        return "rejected", "arena produced no numeric score"
    if score >= threshold:
        return "accepted", f"paired arena score {score:.3f} met {threshold:.3f}"
    if score > 1.0 - threshold:
        return "inconclusive", f"paired arena score {score:.3f} did not reach a boundary"
    return "rejected", f"paired arena score {score:.3f} was below rejection boundary"


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
    state["last_completed_experiment"] = experiment_id
    state["next_experiment"] += 1
    record["status"] = status
    record["completed_at"] = now()
    atomic_json(ROOT / f"experiments/{experiment_id}.json", record)
    atomic_json(STATE_PATH, state)
    git("add", ".autoloop/state.json", f"experiments/{experiment_id}.json")
    git("commit", "-m", f"record experiment {experiment_id}: {status}")


def one_iteration(args: argparse.Namespace) -> None:
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
        else:
            git("worktree", "add", "-b", branch, str(worktree), champion)
            proposal = generate_candidate(worktree, experiment_id, policy)
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
        match = (
            arena(worktree, policy, experiment_id)
            if ci.get("passed")
            else {"passed": False, "skipped": True}
        )
        record["arena"] = match
        status, reason = decide(ci, match, policy)
        record["decision_reason"] = reason
        persist(experiment_id, record, state, status, candidate_sha)
        git("push", "origin", "main")
        print(f"{experiment_id}: {status}: {reason}", flush=True)
    except InfrastructureError as exc:
        record["failure"] = f"{type(exc).__name__}: {exc}"
        persist(experiment_id, record, state, "infrastructure_error", None)
        git("push", "origin", "main", check=False)
        print(f"{experiment_id}: infrastructure_error: {exc}", file=sys.stderr, flush=True)
    except Exception as exc:
        record["failure"] = f"{type(exc).__name__}: {exc}"
        persist(experiment_id, record, state, "failed", None)
        git("push", "origin", "main", check=False)
        print(f"{experiment_id}: failed: {exc}", file=sys.stderr, flush=True)
    finally:
        if worktree.exists() and not args.keep_worktrees:
            git("worktree", "remove", "--force", str(worktree), check=False)


def preflight() -> None:
    for executable in ("codex", "gh", "git", "uv"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"required executable not found: {executable}")
    if git("status", "--porcelain"):
        raise RuntimeError("controller checkout must be clean")
    if git("branch", "--show-current") != "main":
        raise RuntimeError("controller must run from main")
    run(["gh", "auth", "status", "-h", "github.com"])
    if not git("remote", "get-url", "origin"):
        raise RuntimeError("origin remote is required")
    if load(POLICY_PATH).get("competition_upload_enabled") is not False:
        raise RuntimeError("competition upload boundary is not disabled")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--keep-worktrees", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    lock_path = ROOT / ".autoloop/controller.lock"
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another controller instance is already running") from exc
        preflight()
        for _ in range(args.iterations):
            one_iteration(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"controller stopped: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error
