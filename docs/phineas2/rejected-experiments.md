# Phineas 2 — rejected P4 candidates

One entry per candidate that was implemented, unit-verified, ablation-tested
against the current internal champion, and NOT accepted. Kept so a rejected
idea is never retested from scratch. The commit itself stays reachable by
hash (not deleted) even after the branch tip moves past it.

## candidate 2: evaluation-scaled repetition preference at the root

- Commit: `6af3ed8f74df342d771cd6b189c856619d626b77` (branched from
  `phineas2-champion-v2-see`, i.e. after P4 candidate 1a/1b (SEE) had already
  been accepted).
- What it did: a small, capped, ordering-only nudge at the root that prefers
  a non-repeating move over an equally-scored repeating one while clearly
  winning, and the reverse while clearly losing (see the commit message for
  the full mechanism and the direct unit verification that it can never
  change which move is judged better -- only which of two REAL-SCORE-TIED
  moves wins the tie-break).
- Ablation vs `phineas2-champion-v2-see` (candidate vs baseline, 20 games,
  30s+0.3s, 10 preregistered openings x2 colours):
  `+8 =5 -7   score=0.525   95% CI [0.336, 0.714]   0 failures`
- Verdict: **REJECTED**. The point estimate is barely above a coin flip and
  the 95% CI comfortably straddles 0.5 in both directions -- this is not a
  demonstrated improvement. Per the user's instruction ("Reject components
  that do not improve strength or that reduce reliability... do not retain a
  feature merely because it appears theoretically sensible"), a mechanism
  that is provably *safe* (cannot make the search choose a worse-scored move)
  is not the same as a mechanism that is *shown to help*, and only the latter
  clears the bar here.
- Why the null result isn't surprising, and why this isn't necessarily "the
  idea was wrong": the mechanism only ever fires on a genuine tie between a
  repeating and a non-repeating root move while the position is clearly
  decided -- inspection of the SEE-candidate ablation's own draws (see
  `phineas2-champion-v2-see`'s commit message) found zero instances of a
  winning position being given away by repetition in that 20-game sample, so
  the failure mode this candidate targets may simply be rare enough that 20
  games of general play doesn't exercise it either way. It is not retested
  from scratch, but the *mechanism* (a safe, root-only, ordering-only
  tie-break bias, never a blanket contempt term) is worth reviving with a
  larger sample, or targeted at constructed near-tie positions specifically,
  if a real game or larger batch surfaces the actual pathology -- to be
  judged on that future evidence, not on theory alone.

### UPDATE: revived and accepted as phineas2-champion-v6-repetition

The predicted trigger arrived. A Step-7 real-clock screen against Stockfish
elo 2200 (`/tmp/claude-501/exact_clock/screen2200-exactclock_elo2200/`,
game 8) showed the candidate's own champion (v5-mobility, no repetition
preference at all) sit at a self-evaluated +666cp in a king+bishop+2-pawns
technical endgame for ten consecutive of its own moves, depths 22-37 --
including *after* promoting a new queen -- while it shuffled into a
threefold-repetition draw it never needed to accept. That is exactly the
failure mode this candidate targets, observed live, not hypothesised.

The unchanged mechanism (commit `6af3ed8`, reapplied verbatim on top of
`phineas2-champion-v5-mobility` as commit `832a700`) was re-ablated at a
larger sample given the stronger justification: 30 games, 30s+0.3s, vs
`phineas2-champion-v5-mobility`:
`+18 =6 -6   score=0.700   95% CI [0.557, 0.843]   0 failures`

This is a clean accept -- CI entirely above 0.5, more than double the sample
of the first attempt. The original rejection was a false negative from too
small a sample on a low-frequency-but-high-value fix, exactly as the "why
the null result isn't surprising" note above anticipated. Tagged
`phineas2-champion-v6-repetition`.
