# Autonomous optimiser checkpoint — 2026-08-31

Repository: <https://github.com/rohanchakra16/aichessathon-starter>

Competition upload remains disabled. No agent has been uploaded to the AI
Chessathon dashboard, and `.autoloop/state.json` still has
`live_submission: null`.

## Internally nominated submission candidate

The current internal champion passed the protected release workflow and is now
an internal `submission_candidate`. This is a repository state, not a live
competition submission.

- Champion source commit: `ab5286b54b1e35988c681ca26cfec34b1122cdb8`
- Evaluated main commit: `bfaf6a0d90bce0bb611940f9cea9532b4a6563e5`
- Exact ZIP SHA-256:
  `f1d076bf6e502185dd90aa9f4fae40192c32b8ba1dc4b76d71e3dfe820162d69`
- ZIP: 2,818 compressed bytes; 7,207 expanded bytes
- Release run: <https://github.com/rohanchakra16/aichessathon-starter/actions/runs/33344363611>
- Release workflow wait: 95.611 seconds
- Protected evaluation duration: 66.188 seconds

Recorded Linux envelope:

- Python 3.12.11 on x86-64 Linux
- one-core cgroup quota (`cpu.max = 100000 100000`)
- 2,147,483,648-byte memory limit
- 128-process limit
- network disabled (probe returned `ENETUNREACH`)
- read-only repository mount
- writable 268,435,456-byte `/tmp`
- peak container memory: 120,139,776 bytes
- resolved evaluation image ID:
  `sha256:ba6cd6ffe8e29d12dd30ead1ac6deaaebe75990d10470e660071ce8feb1edbce`

Release evidence:

- zero static, package, initialization, legality, crash, flag, or smoke failures
- maximum adversarial move response: 0.354 seconds
- learned-model ablation changed 30 of 32 opening moves
- intact learned model versus zero-weight ablation: 13 wins, 3 draws,
  0 losses; 90.625% score over 16 paired games
- full 120 s + 0.5 s games: two wins by checkmate, one with each colour,
  zero failures

Passing this gate establishes packaging, compliance, reliability evidence, and
material model influence. It does not establish that the agent is strong enough
for the live ladder.

## Autonomous experiment verified

The AI-generated change retained as experiment 9 added more frequent deadline
checks only under extremely short move budgets. The controller automatically:

1. recovered the retained candidate into an isolated branch;
2. pushed the candidate branch;
3. triggered GitHub evaluation from the branch push;
4. built and evaluated the exact ZIP inside the constrained Linux container;
5. ran the protected 64-game candidate-versus-champion arena;
6. downloaded the evidence;
7. applied the fixed sequential boundary;
8. journaled and pushed the result without a merge or approval.

Experiment 9 evidence:

- candidate branch: `autoloop/candidate-0009`
- candidate commit: `d3c88c3d2a748a2782fc26abbe0e012ccaaa1ecb`
- workflow: <https://github.com/rohanchakra16/aichessathon-starter/actions/runs/33344796274>
- exact candidate ZIP SHA-256:
  `c5f75b5538ecee35db24dede7b8d07a73132c119116b706a0f5e14c7ce3de134`
- constrained checks: passed; peak memory 120,127,488 bytes
- arena: 12 wins, 40 draws, 12 losses; 50.0% score
- one-sided 95% interval: 42.806%–57.194%
- decision: inconclusive; candidate retained and champion unchanged
- protected arena duration: 230.593 seconds
- workflow wait: 278.854 seconds

The original AI generation took approximately 164 seconds. A comparable
steady-state iteration therefore takes roughly 7.4 minutes: about 2.7 minutes
for generation and 4.7 minutes for GitHub build, evaluation, and the maximum
64-game arena. Strong candidates can stop at an earlier declared boundary.

## Actual autonomy and remaining approvals

After one controller start, candidate generation, worktree isolation, branch
push, event-triggered evaluation, fixed accept/reject/inconclusive decision,
journaling, promotion when justified, and the next iteration are autonomous.
The Mac must remain awake and online. The controller's nested Codex process
requires the previously granted host permission to run its own sandbox.

The release nomination is a separate `controller.py --release-check` command;
it is not run after every internal promotion. Competition upload is deliberately
manual and unimplemented. There are no dashboard credentials or upload calls in
the controller or workflows.

GitHub Actions supplies a shared physical host. Docker enforces the declared
resource envelope and both arena agents run on the same host, but this is not a
dedicated fixed benchmark machine. Close nodes-per-second comparisons still
need dedicated hardware; match outcomes and failure evidence are retained.

## Failure modes observed and contained

- A pre-created result file was not writable by the unprivileged container.
  The failed release was retained; file permissions were fixed; the same gate
  then passed.
- A nested Codex process could not create a second macOS sandbox while launched
  under the outer app sandbox. No candidate was produced or promoted. The
  controller is now run with the narrowly scoped host permission needed for its
  own isolated worktree sandbox.
- The first constrained champion arena could not traverse a private temporary
  directory, and its hidden artifact directory was ignored by upload. The
  workflow now makes only that temporary checkout traversable and names result
  files explicitly. The same AI candidate was replayed rather than regenerated.
- Infrastructure failures are journaled separately from chess rejection and
  never promote a candidate.

## Minimal next phase

Run small bounded autonomous batches and prioritize improvements to learned
evaluation, move ordering, search efficiency, and real-clock time allocation.
Re-run the protected release check after a statistically accepted internal
champion. Do not upload until additional strength evidence supports spending a
competition slot.
