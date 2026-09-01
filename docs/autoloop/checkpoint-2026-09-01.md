# Autonomous optimiser checkpoint — 2026-09-01

This checkpoint supersedes `checkpoint-2026-08-31.md`.

Repository: <https://github.com/rohanchakra16/aichessathon-starter>

Competition upload remains disabled. No agent has been uploaded by the
controller or GitHub Actions, and `.autoloop/state.json` has
`live_submission: null`.

## Current champion and internal release artifact

- Champion commit: `fb86594855baaa4a02c83facfc95b2fba885f833`
- Champion source experiment: `exp-0032`
- Last completed experiment: `exp-0035`
- Next experiment number: 36
- Evaluated main commit: `c4d4ea1ab3e99e47ed2c23ae204bf3065c591219`
- Deterministic ZIP SHA-256:
  `91c007cf232433f95dcb4285c3a8df58abc3cb29fa407bd355bdfeedda57c675`
- ZIP size: 11,022 compressed bytes; 35,384 expanded bytes
- Members: `agent.py`, `weights/model.json`
- Release record: `releases/c4d4ea1ab3e9-33479569865.json`
- Release run:
  <https://github.com/rohanchakra16/aichessathon-starter/actions/runs/33479569865>

Release evidence:

- Linux constrained evaluation passed on one CPU, 2 GB memory, no network,
  read-only source, and a 256 MB writable `/tmp`.
- Peak memory: 120,127,488 bytes.
- Import: 0.217 seconds.
- Maximum 120-second stress move: 2.004 seconds under the protected 2.25-second
  real-clock ceiling.
- Short-clock stress cases retained the 0.75-second ceiling and all passed.
- Model ablation changed 30 of 32 opening moves.
- Intact model versus zero-weight ablation: 14 wins, 2 draws, 0 losses; 93.75%.
- Exact 120 s + 0.5 s release games: two wins by checkmate, one with each
  colour, zero failures.

## Experiment 32: confirmed real-clock promotion

Experiment 32 changed only long-clock search behavior:

- maximum iterative depth: 6 to 8;
- clocks below 10 seconds retain the proven old budget;
- 120 seconds permits up to 2.0 seconds per move, scaling down with the clock.

It was neutral in the unchanged fast promotion arena: 19 wins, 24 draws,
21 losses; 48.4375%; interval 41.28%–55.66%. Because the change does not
activate at that clock, it was not promoted from this evidence.

The exploratory exact-clock sample scored 4 wins, 2 draws, 2 losses (62.5%)
with zero failures. That sample was retained but excluded from promotion. A
prospective real-clock gate was then committed before seeing any of its games:
eight different confirmation openings, paired colours, 16 games, 120 s +
0.5 s, and the same one-sided 95% lower-bound acceptance rule.

Prospective result:

- 8 wins, 6 draws, 2 losses;
- 68.75% score;
- interval 54.264%–80.312%;
- decision: accept;
- zero candidate or incumbent failures;
- duration: 1,790.858 seconds.

The deterministic controller automatically cherry-picked the candidate,
updated champion state, recorded the evidence, and pushed it. No user click or
merge approval occurred between evaluation and internal promotion.

## Revised active-learning direction

A new reproducible pipeline generated 64 champion-trajectory games from a
fixed-seed, independently sampled top-three teacher opening distribution. It
never reads the promotion or confirmation opening files.

Dataset evidence:

- 837 champion decisions annotated at fixed Stockfish 18 node budgets;
- 512 contexts selected with both high-regret active selection and broad
  exploration, while reserving coverage from every game;
- 2,342 labeled parent and MultiPV child positions;
- dataset SHA-256:
  `78788fde80f78bb868b3088e2045c15728241b6d9e6bd13cfb6110fdb7f9ee5a`;
- every row carries a game ID;
- validation holds out complete games, preventing same-game position leakage;
- the proven 770 champion weights remain fixed; only 14 bounded strategic
  residual coefficients are trainable.

Stockfish is an offline development teacher only. The dataset and engine are
not packaged, and no competition move uses an external engine.

Formal results against the unchanged champion:

| Experiment | Change | W-D-L | Score | 90% interval | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| 33 | first active strategic residual; implementation later found a reversed passed-pawn mask | 16-34-14 | 51.5625% | 44.34%–58.72% | inconclusive |
| 34 | corrected masks, exact trainer/runtime agreement, zero-weight work pruned | 18-28-18 | 50.0% | 42.81%–57.19% | inconclusive |
| 35 | active examples weighted 16×; exact trainer/runtime agreement | 18-28-18 | 50.0% | 42.81%–57.19% | inconclusive |

Experiments 34 and 35 matched all 2,342 retained trainer positions to about
`1.5e-12` maximum floating-point error before dispatch. All three experiments
passed packaging, ablation, timing, memory, legality, and failure checks.

Additional held-out objectives were rejected before arena dispatch:

- Pairwise teacher-margin residuals reduced margin RMSE by less than 0.4% and
  did not improve ranking accuracy over the champion.
- A 778-weight source/destination move-ordering policy reached at best the
  current ordering's 36.67% top-one accuracy, while its best mean reciprocal
  rank was 0.601 versus the current ordering's 0.630.

These variants were not promoted or presented as formal chess improvements.
Their scripts and tests are retained so they are not rediscovered and repeated.

## Stop condition reached for the current direction

The engine-guided/self-play PSQT-plus-small-strategic-residual and learned
ordering direction now has:

- three consecutive formal non-promotions after the newest champion;
- corrected runtime/trainer equivalence;
- pointwise, weighted pointwise, pairwise-margin, and move-ordering objectives
  tested;
- larger and higher-fidelity label variants already tested in experiments
  26–28 without promotion;
- history/killer ordering and typed-transposition variants already tested in
  experiments 29–30 without promotion.

There is no remaining focused coefficient, weighting, or cheap-ordering variant
in this model family justified by held-out evidence. Further work should be
treated as a new direction, not another tweak to the current one. Plausible new
directions include a materially different efficient model class (for example a
small incrementally updated/NNUE-style evaluator) or a redesigned search with
proper bound-aware transposition storage and selective pruning developed and
benchmarked together. Neither should be mixed into the current champion without
new protected experiments.

## What is autonomous and what still needs approval

After one local controller start, candidate worktree isolation, candidate push,
event-triggered GitHub evaluation, constrained packaging, paired games, fixed
statistical decision, journaling, internal promotion, and push of the result are
automatic. Clock-sensitive promotion also completed automatically after its
precommitted exact-clock gate.

The persistent Mac process must remain awake and online. GitHub Actions and the
local Codex invocation must remain authenticated. Infrastructure failures are
recorded separately and never count as chess evidence.

Competition upload is intentionally absent and still requires a separate,
explicit user decision. The controller contains no competition credentials or
upload call.
