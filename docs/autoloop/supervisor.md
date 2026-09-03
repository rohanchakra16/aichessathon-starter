# Unattended research supervisor

`claude_supervisor.py` (repo root) makes the takeover handoff run unattended.
Run it from a **normal macOS Terminal** where `gh` and GitHub Actions work. It
does not run correctly inside the Claude desktop app's Bash sandbox, where `gh`
cannot verify the intercepting proxy's certificate.

```
evidence audit (Claude)  ->  research/next-direction.md  ->  controller.py batch
      ^                                                              |
      +-----------------------  repeat  ------------------------------+
```

## What one cycle does

1. **Evidence packet.** Compact, PGN-free: champion commit, the new experiment
   record(s) from the last batch (status, hypothesis, arena W/D/L + CI,
   reliability, draw/loss breakdown), the 20 newest registry lines, the streak
   count, the exhausted-family list, and the previous audit. Saved to
   `research/audit-input/`.
2. **Audit.** `claude -p --output-format json` with read-only tools
   (`Read`, `Grep`, `Glob`; `Bash`/`Edit`/`Write`/web disabled). Claude performs
   the handoff's between-batch audit against the live repo and ends its reply
   with an `<AUDIT_DECISION>` JSON block: `decision` (`CONTINUE`/`STOP`),
   `stop_condition`, `streak_assessment`, `recurring_failure_modes`,
   `next_direction`, `audit_summary`.
3. **Persist.** The full audit (including Claude's prose) is written to
   `research/audits/audit-<timestamp>-<after-exp>.json`. A one-line event is
   appended to `research/audits/supervisor-log.jsonl`. This is the permanent
   research trail.
4. **Route direction.** On `CONTINUE`, `research/next-direction.md` is rewritten
   from `next_direction`. The `research/` trail is committed and pushed.
5. **Batch.** `./.venv/bin/python controller.py --iterations N` runs in the
   normal environment, so GitHub Actions evaluation and promotion happen exactly
   as before. The controller is unchanged in authority.
6. **Loop.** Reload `.autoloop/state.json`; detect promotion (champion commit
   moved) and infrastructure failure (controller non-zero exit). Then go to 1.
   A finished batch and a promotion are **never** stop conditions — after a
   promotion the next audit simply runs against the new champion.

## How the direction reaches the candidate

`controller.py` gained one small, additive hook: `supervisor_directive()` reads
`research/next-direction.md` from the candidate worktree and appends it to the
candidate-generation prompt, labelled as *research direction only*. It has **no
effect** on evaluation, reliability gating, the arena, or promotion — those stay
entirely with the controller and the protected framework. If the file is absent
or empty the prompt is exactly as before.

## Stop conditions

| Condition | Exit | State |
| --- | --- | --- |
| Claude audit returns `STOP` (`scientific_saturation` / `infrastructure_blocker`) | 0 | preserved, direction/trail committed |
| Claude Code unavailable — auth, usage limit, transport, malformed decision | 20 | preserved, **no batch run** |
| `--max-infra-failures` consecutive controller infrastructure failures (default 2) | 30 | preserved |
| `.autoloop/supervisor.stop` file present (checked each cycle) | 0 | preserved |
| `--max-batches` backstop reached (default 40) | 0 | preserved; re-run to continue |
| Ctrl-C | 130 | preserved |

A generator/infrastructure failure is never counted as a scientific
non-improvement, matching the handoff.

## Usage

```sh
# unattended, audit after every experiment
./.venv/bin/python claude_supervisor.py

# larger batches, more headroom
./.venv/bin/python claude_supervisor.py --iterations 3 --max-batches 20

# see one audit and the direction it produces without running the controller
./.venv/bin/python claude_supervisor.py --dry-run

# stop cleanly from another shell
touch .autoloop/supervisor.stop
```

Key flags: `--iterations` (experiments/batch, default 1), `--max-batches`,
`--max-infra-failures`, `--audit-model` (default `sonnet`), `--audit-effort`
(default `high`), `--audit-budget-usd` (default 2.0), `--no-push`, `--dry-run`.

## Boundaries

The supervisor **never**: edits `agent.py`, `weights/`, or any protected path;
changes evaluation, reliability, promotion, openings, thresholds, or the upload
boundary; uploads to the competition; invokes Codex. It only ever commits
`research/`. The controller remains the sole authority for candidate evaluation
and promotion; Claude controls research direction only.
