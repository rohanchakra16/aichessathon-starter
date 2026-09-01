# Autonomous optimiser checkpoint — 2026-09-01

This checkpoint supersedes `checkpoint-2026-08-31.md`.

Repository: <https://github.com/rohanchakra16/aichessathon-starter>

Competition upload remains disabled. No agent has been uploaded by the
controller or GitHub Actions, and `.autoloop/state.json` has
`live_submission: null`.

## Current champion and internal release artifact

- Champion commit: `b4cf1218af343df146af3ae6fec93ede5fdcf798`
- Champion source experiment: `exp-0038`
- Last completed formal experiment: `exp-0038`
- Next formal experiment number: 39
- Evaluated main commit: `e6c0e6d10189f57f63037759e90e4b1f9a3f2edc`
- Deterministic ZIP SHA-256:
  `1f63db64f090e010497b236f7cf8ed7c560eb2f20006694770a5bf1763dda121`
- ZIP size: 11,368 compressed bytes; 36,970 expanded bytes
- Members: `agent.py`, `weights/model.json`
- Release record: `releases/e6c0e6d10189-33540401350.json`
- Release run:
  <https://github.com/rohanchakra16/aichessathon-starter/actions/runs/33540401350>

Release evidence:

- Linux constrained evaluation passed on one CPU, 2 GB memory, no network,
  read-only source, and a 256 MB writable `/tmp`.
- Peak memory: 120,156,160 bytes.
- Import: 0.220 seconds.
- Maximum 120-second stress move: 2.002 seconds under the protected 2.25-second
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

## Stop condition reached for the previous residual direction

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

## Compact NNUE-style evaluator direction

The next protected direction implemented a compact king-aware,
two-perspective neural accumulator on top of the champion's frozen 770-weight
tapered evaluator. The network is evaluated locally with NumPy, has eight
hidden units, remains well within the 50 MB package limit, and uses no runtime
engine or network access.

Formal results against the unchanged `exp-0032` champion:

| Experiment | Change | W-D-L | Score | 90% interval | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| 36 | 50% neural-residual blend, four training epochs | 15-35-14 | 50.78125% | 43.57%–57.96% | inconclusive |
| 37 | same network at a 75% residual blend | 12-31-21 | 42.96875% | 35.99%–50.24% | inconclusive |

Both candidates passed the complete constrained CI gate. Experiment 36's
package was 1,974,365 compressed bytes and 5,317,531 expanded bytes, imported
in 0.365 seconds, peaked at 236,101,632 bytes, and completed the real-clock
stress move in 2.006 seconds. Model ablation changed all 32 sampled moves.

Several additional variants were rejected before formal dispatch:

- Doubling the hidden width to 16 doubled model size for only about 0.4% RMSE
  improvement and no held-out ranking gain.
- A high-fidelity-only dataset labelled at 5,000 Stockfish nodes improved
  static validation metrics but scored 1-3-4 (31.25%) in paired play.
- A public recent-master corpus was added as an offline training source. The
  reproducible source is TWIC 1660 (31 August 2026), containing 9,139 games;
  6,000 positions from 2,001 complete game groups were relabelled by
  Stockfish 18 at 5,000 nodes. The archive SHA-256 is
  `f46e172fa2ed5fe53cc465e44f3c8f321685d123006e32b6f531b95dafffc263`
  and the labelled dataset digest is
  `c8ca12507f59fa32366cbd122f19e9825a1d6fc9554f70c84c0c7a960cfc57a1`.
  Games, rather than individual positions, are assigned wholly to training or
  validation, preventing position leakage.
- The eight-epoch hybrid self-play plus master-position model improved both
  active and historical label RMSE but scored 0-11-5 (31.25%) in a 16-game
  paired pre-screen. It is retained on
  `research/nnue-hybrid-e8-offline-rejected`.
- The less-fitted six-epoch hybrid scored 2-9-5 (40.625%): 2-4-2 on held-out
  active positions and 0-5-3 on held-out master positions. It is retained on
  `research/nnue-hybrid-e6-offline-rejected`.

The master games are therefore useful position coverage, not direct move
imitation. Stockfish supplies the labels, while actual paired games remain the
promotion authority.

## Stop condition reached for compact NNUE leaf evaluation

The tested family now covers conservative, medium, and full residual strength;
two network widths; ordinary and high-fidelity engine labels; self-play-only
and master-position hybrid corpora; and both static held-out and actual-play
screens. The sequence contains two formal non-promotions plus three offline
playing failures, while the protected champion remains unchanged.

Static prediction quality consistently failed to translate into stronger
shallow-search play. Further changes to epochs, blend percentages, or nearby
network sizes are no longer well motivated. Any subsequent experiment should
change the search architecture or training objective materially rather than
continue tuning this leaf evaluator.

## Selective-search promotion

Experiment 38 replaced the unsafe exact-score transposition cache with
bound-aware entries, added principal-variation probes, and conservatively
reduced only late quiet moves. Checks, captures, promotions, early ordered
moves, and improving reduced probes retain a full-depth search.

The unchanged fast arena scored 15 wins, 26 draws, and 23 losses (43.75%),
with a 90% interval of 36.74%–51.02%. That result was inconclusive and did not
promote the candidate. The change was independently screened at longer clocks,
where it scored 2-1-1, then entered the frozen prospective real-clock gate
that had been committed before its games were observed.

Prospective competition-clock result against `exp-0032`:

- 7 wins, 8 draws, and 1 loss;
- 68.75% score;
- 90% interval 54.264%–80.312%;
- decision: accept;
- zero candidate or incumbent failures;
- duration: 2,030.116 seconds.

The controller automatically cherry-picked the candidate as the new champion
at `b4cf1218af343df146af3ae6fec93ede5fdcf798`, updated state, and pushed the
evidence. The subsequent release gate also passed, including the constrained
Linux environment, package, static checks, model ablation, short-clock stress,
and two exact 120 s + 0.5 s games won by checkmate with opposite colours.

Focused follow-ups did not improve on the promoted combination:

- exact PVS without late-move reductions scored 2-8-6 (37.5%) in an
  independent fast pre-screen;
- extending check evasions at the quiescence boundary scored 3-8-5 (43.75%);
- transposition best-move ordering plus exactly equivalent evaluator
  optimisations reduced fixed-depth runtime by 43%–59%, but scored 2-3-3
  (43.75%) across two independent longer-clock screens.

These results show that nominally deeper search is not automatically stronger
with the current value model. The promoted PVS/LMR combination remains
protected; the next training cycle should use its own trajectories rather than
continue isolated depth and ordering changes.

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
