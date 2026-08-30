# Chessathon autonomous optimiser bootstrap report

Date: 2026-08-30 (Europe/London)

Repository: <https://github.com/rohanchakra16/aichessathon-starter>

Official upstream audited at commit `c9aa95339bd61dc4dea7099328e55a2ae8a86cb1`.
The official `harness/` was not edited.

## Current internal champion

- Source commit: `ab5286b54b1e35988c681ca26cfec34b1122cdb8`
- Journal commit: `a5fecd53c5ece912472b79e16f64d8f1be0cbd59`
- Deterministic ZIP SHA-256:
  `f1d076bf6e502185dd90aa9f4fae40192c32b8ba1dc4b76d71e3dfe820162d69`
- ZIP contents: root-level `agent.py` and `weights/model.json`
- Expanded size: 7,207 bytes
- Learned evaluator: repository-trained 12-feature linear leaf model; all
  non-terminal search leaf scores come from the model.
- Search: iterative deepening negamax/alpha-beta, ordered moves, bounded
  transposition table, conservative clock budget, and promoted depth-2
  quiescence search.

This is an **internal champion**, not a submission candidate or live upload.

## Compliance and reliability evidence

- Exact byte-reproducible ZIP is built and extracted for evaluation.
- Lint and strict typing pass.
- Ten autonomous-policy/reproducibility tests pass.
- Adversarial corpus covers promotions, castling, en passant, mate/stalemate,
  low clocks, and late-game counters.
- Latest measured initialization: approximately 0.052 seconds.
- Latest maximum adversarial move response: approximately 0.232 seconds.
- Two-colour exact-package smoke games pass.
- No illegal moves, crashes, flags, or initialization failures were observed in
  the promoted candidate's protected suite and 16-game arena.
- Original GitHub `ci` passed after promotion.

## Strength evidence

The learned seed scored 93.8% against `baselines/greedy` over eight games and
25% against `baselines/minimax` over four fast games during bootstrap.

Promoted experiment 5 was evaluated from eight frozen legal openings with
paired colours and exact packaged agents:

- 6 wins, 8 draws, 2 losses
- score 62.5%
- zero failed terminations
- duration 42.904 seconds
- every PGN retained in `experiments/exp-0005.json`
- fixed promotion threshold: 62.5%

The sample is intentionally sufficient only for early internal promotion. It
is not strong enough for submission-candidate nomination.

## Autonomous behavior actually observed

- AI candidate generation in isolated worktrees.
- Candidate-only branches pushed automatically.
- GitHub event-triggered exact-package evaluation.
- Evaluation artifact discovery and download pinned explicitly to the user fork.
- Fixed acceptance, rejection, and inconclusive decision logic.
- Promotion and journal push to `main` without a manual merge.
- Failed and inconclusive candidate branches retained.
- Interrupted experiment 2 resumed from its retained commit after a controller
  fix rather than being regenerated.
- Experiments 3 and 4 ran consecutively in one invocation. Experiment 4 started
  1.446 seconds after experiment 3 completed, with no click or approval.
- Experiment 5 crossed the fixed boundary and was promoted automatically.

Measured end-to-end times:

| Experiment | Status | Duration | GitHub wait | Arena score |
|---|---:|---:|---:|---:|
| 3 | inconclusive | 208.217 s | 31.928 s | 50.0% |
| 4 | inconclusive | 145.640 s | 25.890 s | 50.0% |
| 5 | accepted | 254.136 s | 20.040 s | 62.5% |

AI generation and its own local pretests dominate latency. The controller can
now run persistently with `--continuous` and stops cleanly on a stop-file or
infrastructure/authentication failure.

## Observed failure and recovery modes

- A Git porcelain parser defect safely rejected experiment 1; nothing was
  promoted and the failure was journaled.
- A fork/upstream ambiguity caused experiment 2's controller poll to query the
  official repository. CI still completed; the controller was stopped, fixed,
  and resumed the retained candidate. All workflow queries are now pinned to
  `rohanchakra16/aichessathon-starter`.
- Identical-start deterministic matches produced repeated draws. Those records
  remained inconclusive. The protected arena now uses frozen paired openings.
- Infrastructure failures are distinct from candidate rejection and halt
  continuous mode instead of causing an immediate retry storm.

## Remaining work before submission-candidate nomination

1. Provision a fixed Linux benchmark host/container that enforces one CPU,
   2 GB RAM, no network, 256 MB scratch, and process/thread limits. Docker is
   not currently installed on the Mac, so close performance measurements are
   not yet competition-envelope evidence.
2. Expand the frozen opening suite and use a declared sequential statistical
   test with substantially more games.
3. Add protected learned-model ablation demonstrating material move-selection
   and strength impact.
4. Run resource telemetry and two-colour real-clock release games at 120 s +
   0.5 s before setting `submission_candidate`.
5. Rehearse the deterministic ZIP on Linux using the competition dependency
   versions and record the image digest.
6. Confirm the timezone for the Sep 11 11:00 upload close and 12:00 lock; the
   live rules omit it.

Competition upload remains disabled in policy. The controller has no upload
code or competition credentials, and `live_submission` remains `null`.

