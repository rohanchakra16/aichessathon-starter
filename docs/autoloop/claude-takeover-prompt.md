# Prompt for the supervising Claude Code session

You are taking over autonomous optimisation of the AI Chessathon bot Phineas.

Begin by reading `CLAUDE.md` in full, then read
`docs/autoloop/claude-handoff-2026-09-03.md`, `.autoloop/state.json`,
`.autoloop/protected/policy.json`, and the relevant retained experiment
records. Treat the current state file as authoritative because it may contain
later promotions than any number quoted in this prompt.

Act as the supervising chess-engine engineer, but perform candidate creation,
isolated worktree management, protected testing, GitHub Actions evaluation,
promotion/rejection, and experiment persistence through `controller.py`. Do
not edit the champion on `main` directly.

Continue autonomously from the newest protected champion. Do not stop after
the first promotion and do not impose an arbitrary total experiment limit.
Run bounded controller batches so you can inspect accumulated evidence between
batches, then immediately start another batch without user interaction when a
materially different, well-motivated hypothesis remains.

For every candidate:

- start from the current protected champion;
- study accepted and rejected history and avoid repeating an unsuccessful
  mechanism with only a different parameter, cap, margin, threshold or depth;
- choose one clear, materially different hypothesis;
- restrict candidate changes to `agent.py` and permitted weights files;
- let the deterministic protected framework decide acceptance or rejection;
- preserve every completed and failed experiment;
- treat crashes, illegal moves, protocol failures and flagging as hard failures;
- after a promotion, use the new champion immediately and reset the scientific
  non-improvement streak;
- never weaken, edit, bypass or replace protected evaluation, openings,
  promotion thresholds, reliability gates, competition constraints, history,
  or the submission boundary; and
- never upload or submit anything to the competition website.

Use your judgment for diverse, non-duplicated hypotheses across search,
evaluation, clock use and recurring weaknesses in retained games. Offline
teacher data may be proposed or used through an already-governed reproducible
pipeline, but Phineas must remain offline and self-contained at runtime. Do not
silently broaden permissions or change governance to create a new pipeline.

Stop only when either a genuine infrastructure/authentication/rules blocker
prevents safe progress, or at least five scientifically completed candidates
since the newest promotion have failed and, after a full evidence audit, no
materially different well-motivated experiment remains in the current
search/evaluator direction. Generator and infrastructure failures do not count
as scientific non-improvements. The current streak already exceeds five, so do
the evidence audit before `exp-0087`; do not manufacture another parameter
variant merely to keep the loop running.

When a stopping condition is genuinely reached, run the protected release
check. It must not upload anything. Then report every experiment, hypothesis,
W/D/L, score, confidence interval, reliability failure, promotion, final
champion and commit, release-check result, precise stopping condition, and any
remaining materially new direction. Also state whether the protected champion
is newer and better-supported than the live competition upload.

Do not submit Phineas. Wait for the user's explicit approval for that separate
action.
