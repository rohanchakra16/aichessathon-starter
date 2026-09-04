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
2. **Audit.** `claude -p --output-format json --json-schema ...` with read-only tools
   (`Read`, `Grep`, `Glob`; `Bash`/`Edit`/`Write`/web disabled). Claude performs
   the handoff's between-batch audit against the live repo and returns a
   schema-enforced object containing `decision` (`CONTINUE`/`STOP`),
   `stop_condition`, `streak_assessment`, `recurring_failure_modes`,
   `next_direction`, and `audit_summary`. The supervisor prefers Claude CLI's
   validated `structured_output` field; the older tagged-text parser remains
   only for backward compatibility with retained audit responses.
3. **Persist.** The full structured audit and Claude result text are written to
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

## The `learned-evaluator-retrain` family

When an audit returns `next_direction.family == "learned-evaluator-retrain"`, the
supervisor runs the batch as
`controller.py --retrain-entrypoint training/train_positional_evaluator.py`. That
path is **deterministic — no `claude -p` in the candidate**: the controller
splices the trainer's canonical feature source into `agent.py` between markers,
runs the whitelisted offline trainer to refit only a bounded positional/endgame
residual on the frozen 770-weight PSQT (`weights/model.json` schema 7), and
commits `{agent.py, weights/model.json}`. Evaluation, ablation, reliability,
arena and promotion downstream are unchanged. `retrain_blocker()` refuses the
family (safe stop) until `policy.retrain.datasets` sha256 are pinned and match
the committed `training/data/` files. See
`docs/autoloop/retrain-family-proposal.md`.

## Stop conditions

| Condition | Exit | State |
| --- | --- | --- |
| Claude audit returns `STOP` (`scientific_saturation` / `infrastructure_blocker`) | 0 | preserved, direction/trail committed |
| Audit picked `learned-evaluator-retrain` but `policy.retrain.datasets` are not pinned | 0 | preserved; pin the sha256 and re-run |
| Claude Code unavailable — auth, usage limit, transport, or missing schema output | 20 | preserved, **no batch run** |
| `--max-infra-failures` consecutive controller infrastructure failures (default 2) | 30 | preserved |
| `.autoloop/supervisor.stop` file present (checked each cycle) | 0 | preserved |
| `--max-batches` backstop reached (default 40) | 0 | preserved; re-run to continue |
| Ctrl-C | 130 | preserved |

A generator/infrastructure failure is never counted as a scientific
non-improvement, matching the handoff.

Every exit path commits its final stop event, including dry runs, Claude
unavailability, Ctrl-C, stop-file exits and the batch backstop. A failed push is
also recorded in a clean local commit, so neither condition leaves a tracked
log modification that would block the next preflight; the next successful push
publishes the deferred commits.

## Usage

```sh
# fully autonomous: no arbitrary batch limit; stop only on the rules above
./.venv/bin/python claude_supervisor.py --continuous

# unattended, audit after every experiment
./.venv/bin/python claude_supervisor.py

# larger batches, more headroom
./.venv/bin/python claude_supervisor.py --iterations 3 --max-batches 20

# see one audit and the direction it produces without running the controller
./.venv/bin/python claude_supervisor.py --dry-run

# stop cleanly from another shell
touch .autoloop/supervisor.stop
```

Key flags: `--iterations` (experiments/batch, default 1), `--continuous`, `--max-batches`,
`--max-infra-failures`, `--audit-model` (default `sonnet`), `--audit-effort`
(default `high`), `--audit-budget-usd` (default 2.0), `--no-push`, `--dry-run`.

## Boundaries

The supervisor **never**: edits `agent.py`, `weights/`, or any protected path;
changes evaluation, reliability, promotion, openings, thresholds, or the upload
boundary; uploads to the competition; invokes Codex. It only ever commits
`research/`. The controller remains the sole authority for candidate evaluation
and promotion; Claude controls research direction only.
