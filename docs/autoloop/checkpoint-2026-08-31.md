# Autonomous optimiser checkpoint — 2026-08-31

Repository: <https://github.com/rohanchakra16/aichessathon-starter>

Competition upload remains disabled. No agent has been uploaded to the AI
Chessathon dashboard, and `.autoloop/state.json` still has
`live_submission: null`.

## Internally nominated submission candidate

The current internal champion passed the protected release workflow and is now
an internal `submission_candidate`. This is a repository state, not a live
competition submission.

- Champion source commit: `064f56d246a6378d0a83a9386a31d498559ed27d`
- Evaluated main commit: `d1f3f359c30cad092c00f04faa118834c6781641`
- Exact ZIP SHA-256:
  `91887beea1e9d89cbb4355ceb5cd3827c35643097166ed05a7770251b9d60448`
- ZIP: 10,804 compressed bytes; 35,275 expanded bytes
- Release run: <https://github.com/rohanchakra16/aichessathon-starter/actions/runs/33415322062>
- Release workflow wait: 166.098 seconds
- Protected evaluation duration: 137.042 seconds

Recorded Linux envelope:

- Python 3.12.11 on x86-64 Linux
- one-core cgroup quota (`cpu.max = 100000 100000`)
- 2,147,483,648-byte memory limit
- 128-process limit
- network disabled (probe returned `ENETUNREACH`)
- read-only repository mount
- writable 268,435,456-byte `/tmp`
- peak container memory: 119,984,128 bytes
- resolved evaluation image ID:
  `sha256:d52f3e9d23bb3ce9cf7adfe48c85c7e4d477a95d9af75e8f98f298e75b48bb47`

Release evidence:

- zero static, package, initialization, legality, crash, flag, or smoke failures
- maximum adversarial move response: 0.689 seconds
- learned-model ablation changed 31 of 32 opening moves
- intact learned model versus zero-weight ablation: 15 wins, 1 draw,
  0 losses; 96.875% score over 16 paired games
- full 120 s + 0.5 s games: two wins by checkmate, one with each colour,
  zero failures

Passing this gate establishes packaging, compliance, reliability evidence, and
material model influence. It does not establish that the agent is strong enough
for the live ladder.

## Autonomous promotion verified

Experiment 23 combined the self-play-trained tapered evaluator with depth-six
iterative search and a more assertive clock budget. The controller
automatically:

1. recovered the retained candidate into an isolated branch;
2. pushed the candidate branch;
3. triggered GitHub evaluation from the branch push;
4. built and evaluated the exact ZIP inside the constrained Linux container;
5. ran the protected sequential candidate-versus-champion arena;
6. downloaded the evidence;
7. applied the fixed sequential boundary;
8. promoted, journaled, and pushed the accepted result without a merge approval.

Experiment 23 evidence:

- candidate branch: `autoloop/candidate-0023`
- candidate commit: `064f56d246a6378d0a83a9386a31d498559ed27d`
- workflow: <https://github.com/rohanchakra16/aichessathon-starter/actions/runs/33414971398>
- exact candidate ZIP SHA-256:
  `91887beea1e9d89cbb4355ceb5cd3827c35643097166ed05a7770251b9d60448`
- constrained checks: passed; zero candidate or incumbent failures
- arena stopped at the first permitted boundary: 11 wins, 17 draws,
  4 losses; 60.9375% score over 32 games
- one-sided 95% interval: 50.658%–70.330%
- decision: accepted and promoted because the lower bound exceeded 50%
- protected arena duration: 156.395 seconds
- workflow wait: 211.322 seconds

The accepted workflow took about 3.5 minutes after controller dispatch. A
maximum 64-game inconclusive iteration has recently taken about 5–6 minutes,
plus local candidate generation and screening. Strong candidates can stop at
the fixed 32-game boundary, as experiment 23 did.

## Bounded strength batch through experiment 23

Experiments 10–23 were run without changing the frozen arena threshold. Every
candidate and PGN remains in `experiments/`. Experiment 23 is the first
statistically accepted promotion; the champion is now
`064f56d246a6378d0a83a9386a31d498559ed27d`.

| Experiment | Change | W-D-L | Score | 90% interval | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| 10 | checking-move ordering | 14-37-13 | 50.8% | 43.6%–58.0% | inconclusive |
| 11 | mate-distance terminal scoring | 13-37-14 | 49.2% | 42.0%–56.4% | inconclusive |
| 12 | first offline-teacher evaluator | — | — | — | protected image dependency failure |
| 13 | exact experiment 12 replay after infrastructure repair | 1-15-16 | 26.6% | 18.6%–36.5% | rejected |
| 14 | quiet-position, material-anchored tapered evaluator | 15-32-17 | 48.4% | 41.3%–55.7% | inconclusive |
| 15 | bounded principal-variation/null-move search | 12-39-13 | 49.2% | 42.0%–56.4% | inconclusive |
| 16 | constrained compact positional evaluator | 12-32-20 | 43.8% | 36.7%–51.0% | inconclusive |
| 17 | independently generated 3,163-position opening book | 11-42-11 | 50.0% | 42.8%–57.2% | inconclusive |
| 18 | search among three ranked opening candidates | 10-46-8 | 51.6% | 44.3%–58.7% | inconclusive |
| 19 | conservative clock budget (`clock/80`, 0.35 s cap) | 12-44-8 | 53.1% | 45.9%–60.2% | inconclusive |
| 20 | larger clock budget (`clock/35`, 0.70 s cap) | 14-41-9 | 53.9% | 46.7%–61.0% | inconclusive |
| 21 | realistic self-play-trained tapered evaluator | 17-38-9 | 56.2% | 49.0%–63.3% | inconclusive |
| 22 | experiment 21 plus PVS/null-move/TT search | 15-37-12 | 52.3% | 45.1%–59.5% | inconclusive |
| 23 | experiment 21 model, depth 6, `clock/25` budget | 11-17-4 | 60.9% | 50.7%–70.3% | **accepted** |

The experiment 21–23 evaluator was trained from 8,000 quiet positions sampled
from 587 independently generated engine-guided games. Training starts from the
standard initial board, records the dataset and teacher hashes, uses a fixed
node budget and sampling recipe, and clips labels at 1,500 centipawns. Its
learned material scale is sane (approximately pawn 96, knight 317, bishop 329,
rook 496, queen 897 centipawns). The candidate ZIP contains only the compact
770-weight tapered model and Python agent; it contains and invokes no engine.

The first teacher model exposed a data-quality failure: noisy tactical random
positions let linear regression learn absurd material scales. The repaired
trainer now uses quiet legal positions, a conventional material prior, a
recorded teacher-binary hash, deterministic node limits, and a dataset digest.
The corrected tapered model learned approximately pawn 96, knight 317, bishop
327, rook 495, and queen 896 centipawns, then tied rather than lost badly.

A second protected trainer covers material, pawn structure, mobility, king
safety, and phase in 39 runtime features. It supports a digest-verified local
label cache, fixes material at conventional values, and constrains learned
positional coefficients to recorded chess-valid ranges. Its lower static
validation error did not translate into match strength, so this evaluator
family should not receive more coefficient tuning without a better position
distribution or model class.

The opening-book generator starts from the standard initial position and never
reads the protected benchmark opening list. Its fixed 4,4,4,4,2,2,2,2 branch
schedule produced 3,163 entries in a 241 KB JSON source artifact. Candidate 17
still passed learned-model ablation on 17 of 32 positions, but the book produced
no net match advantage and was not promoted.

Experiment 18 regenerated the same independent tree with up to three ranked
candidate moves per position and let the learned search choose among them. It
passed model ablation on 21 of 32 positions and scored 51.6% overall. The 30
games whose starts were covered by the tree scored 51.7%; the 34 uncovered
games scored 51.5%. Root restriction therefore added no measurable value and
should not receive further top-one/top-two tuning.

Offline Stockfish 18 was used only as a reproducible development teacher. Its
binary SHA-256 is
`ae4c93fa9676ca7750d0714342fd8a5b1d018000fc6e0f6cedf112067b5ef374`.
No candidate ZIP contains or invokes Stockfish, and the competition runtime has
no network access.

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

The requested multi-experiment batch is complete because experiment 23 was
promoted and passed the release check. Do not repeat the tested one-line
move-ordering, mate-distance, compact-linear, direct-opening-book,
ranked-root-restriction, or PVS/null-move variants.

Before spending a competition slot, compare the new champion against stronger
independent opponents and expand realistic-clock evidence. The next optimiser
batch should start from experiment 23 and test one material change at a time,
preferably a learned move-ordering policy or a larger independently generated
self-play dataset. Re-run the protected release check only after another
statistically accepted champion. Do not upload without an explicit user
instruction.
