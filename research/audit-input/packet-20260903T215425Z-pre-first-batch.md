You are the supervising chess-engine researcher for the AI Chessathon bot
Phineas, performing the mandatory between-batch evidence audit from
docs/autoloop/claude-takeover-prompt.md and
docs/autoloop/claude-handoff-2026-09-03.md.

You have read-only tools (Read, Grep, Glob). The working directory is the repo
root. Use them to inspect anything you need: agent.py (the current champion),
experiments/exp-*.json (full records, including game PGNs, for the batch just
run and any earlier experiment), docs/autoloop/*, research/*, .autoloop/state.json,
.autoloop/protected/policy.json. Do not attempt to edit anything.

## Fixed context

Champion commit: 4a0c988009ecae163ac09368f92e4a792dac7568
Live competition upload: older exp-0052-era champion (do not upload; out of scope).
next_experiment in state.json: 90
Consecutive scientific non-improvements before this audit: 0

## Recent experiment registry (newest first, compact)

- exp-0089: accepted; HYPOTHESIS: Adding a cheap fixed 3-centipawn-per-square net-mobility differential to the leaf evaluation gives alpha-beta a monotone gradient toward a
- exp-0088: failed; 
- exp-0087: inconclusive; HYPOTHESIS: Replacing per-move UCI-string tie-breaking in move ordering with a cheap packed-integer key removes throwaway string allocation at every n
- exp-0086: inconclusive; HYPOTHESIS: Aspiration-window iterative deepening reduces root search effort while full-window recovery preserves correctness, enabling deeper searche
- exp-0085: inconclusive; HYPOTHESIS: Per-ply killer-move ordering surfaces empirically strong quiet refutations earlier in alpha-beta, yielding more cutoffs and greater search
- exp-0084: inconclusive; HYPOTHESIS: Depth-preferred transposition-table principal-move reuse improves alpha-beta cutoffs and search depth by searching previously successful m
- exp-0083: rejected; HYPOTHESIS: Prioritizing forcing checks in move ordering improves alpha-beta cutoffs and search depth under the fixed clock without changing evaluatio
- exp-0082: inconclusive; 
- exp-0081: inconclusive; HYPOTHESIS: Depth-aware mate scores prioritize faster forced wins and delay unavoidable losses, improving decisive-line selection without altering lea
- exp-0080: inconclusive; HYPOTHESIS: Exception-safe move unwinding in negamax and quiescence prevents deadline timeouts from corrupting board state, improving reliable timeout
- exp-0079: inconclusive; HYPOTHESIS: Adding zugzwang-guarded null-move pruning lets iterative deepening reach one to two plies deeper within the same clock budget, sharpening 
- exp-0078: inconclusive; HYPOTHESIS: A capped final check-evasion ply prevents static evaluation of unresolved checks, reducing horizon errors without allowing unbounded quies
- exp-0077: inconclusive; HYPOTHESIS: Including non-capturing promotions in quiescence will prevent severe horizon misvaluations of advanced pawns and improve tactical move sel
- exp-0076: inconclusive; Per-ply killer-move ordering surfaces quiet moves that previously produced beta cutoffs, aiming to improve alpha-beta move ordering and effective sear
- exp-0075: failed; 
- exp-0074: failed; 
- exp-0073: failed; 
- exp-0072: failed; 
- exp-0071: failed; 
- exp-0070: rejected; HYPOTHESIS: Ordering quiet moves that previously produced beta cutoffs at the same ply ahead of other quiet moves will sharpen alpha-beta pruning and 

## New experiment record(s) from the batch just run

None yet - this audit sets the direction for the first batch.

Controller stdout (tail):
(none)

## Mechanism families already recorded as exhausted in their tested form

- history / killer / TT-best-move move ordering
- aspiration windows
- guarded null-move pruning
- bounded check extensions / evasions
- general forcing-check priority (decisively worse)
- mate-distance scoring
- quiet promotions in quiescence
- quiescence delta pruning
- draw contempt
- halfmove-clock TT bucketing / early repetition-call guarding
- timeout exception unwinding
- exact quiescence stalemate repair
- move-ordering key micro-optimisation (packed int vs UCI string)
- time-budget / clock allocation changes without a structural search change
- simple six-feature linear king-safety residual (offline: no signal)

Changing only a cap, threshold, depth, margin, table size, ordering weight or
time constant of any of the above is NOT a materially new hypothesis.

## Previous audit

{
  "completed_at": "2026-09-03T21:49:24.803400+00:00",
  "after_experiment": null,
  "decision": "CONTINUE",
  "next_direction": {
    "title": "Non-linear king-danger term in the leaf evaluation",
    "family": "leaf-evaluation",
    "hypothesis": "Adding a cheap non-linear king-danger penalty to _model_evaluate reduces the recurring middlegame checkmate losses without retraining. During the existing per-piece loop in _model_evaluate, OR each piece's attacks_mask (including the kings') into one integer attack-bitboard per colour. After the loop, for each side compute d = chess.popcount(enemy_attack_bb & ring_bb) where ring_bb is the king's adjacent squares (chess.BB_KING_ATTACKS[board.king(colour)]), clamp d to a small cap, and add sign * -KS_WEIGHT * d * d to score. The quadratic shape gives alpha-beta a monotone, sharply increasing reason to keep its own king shielded and to press an attack on an exposed enemy king, which the flat castling-rights scalar cannot express.",
    "rationale": "exp-0089's own retained arena shows 6/32 losses and a clear majority of them are mates around moves 20-45 with an uncastled king after an early queen sortie (reti, sicilian, slav, caro-kann, alekhine games). The champion has no attacker-aware king term at all, only WEIGHTS[CASTLING_OFFSET] as a static bonus. exp-0089 just demonstrated that a single cheap hand-scaled leaf term (mobility) can move the paired-arena score to 0.625 without touching the model. The only prior king-safety attempt was a six-feature LINEAR trained residual that had no offline signal and never reached the arena; a direct non-linear attacker-count penalty computed in agent.py is a materially different mechanism and is explicitly named as an open direction in the handoff. Reusing the attack masks already gathered for the mobility loop makes the added cost a handful of int-ORs and two popcounts per leaf.",
    "guardrails": "Reuse the existing piece loop: accumulate attacks_mask into two Python ints, with no second legal-move generation and no per-leaf object allocation beyond those ints and the ring lookup. Use only documented python-chess API (board.king, chess.BB_KING_ATTACKS or board.attacks_mask, chess.popcount, chess.SquareSet); never read library internals or any path outside agent.py and weights/. Clamp d at roughly 6-8 and fix KS_WEIGHT near 2-3 centipawns so the worst-case penalty stays around 1-1.5 pawn and only in genuine mating attacks, while a typical position stays well under a third of a pawn and far below the model's ~118 cp RMSE, keeping the trained PSQT and the accepted mobility term dominant. Must hold model-ablation minimum_strength_score >= 0.625 with >= 8 divergent moves, and pass the full stress_time_left_ms grid (1/25/100/1000/120000) with required_failures 0; exp-0089's worst single move was 0.807 s against the 2.25 s ceiling at the 120 s clock and 0.042 s against 0.75 s at the 1 s clock, so the extra popcounts must keep leaf cost within roughly 1.3x. No change to _budget_seconds or the iterative-deepening loop. If the arena is inconclusive, do not merely retune KS_WEIGHT, the cap, or the ring definition: move to the next distinct positional term (passed pawns, then rook-on-open-file or bishop pair)."
  },
  "audit_summary": "exp-0089 promoted a cheap fixed net-mobility term into the previously positionally blind tapered-PSQT leaf, scoring 0.625 (interval [0.522, 0.718], 14-12-6) with clean reliability and ablation, and reset the non-improvement streak to 0. The current champion (agent.py at 4a0c988) now has a mature PVS/LMR/TT search plus one mobility term, but its only king feature is a flat castling-rights scalar. The direction is not saturated: the twelve-candidate search/ordering/quiescence/time plateau (exp-0076..exp-0088) has been broken by a leaf-evaluation change, and several cheap positional terms remain untested in the arena. The dominant recurring failure modes in retained games are (1) non-forced threefold-repetition draws from balanced-or-better positions and (2) middlegame checkmates with an uncastled king after an early queen move. The next experiment adds a non-linear king-danger penalty to _model_evaluate, reusing the attack masks already gathered for the mobility loop, with a small hand-chosen quadratic coefficient and an attacker-count cap, keeping the trained model dominant. This is distinct from the exhausted six-feature linear king-safety residual (which had no offline signal and never reached the arena). If it does not help, passed pawns, rook-on-open-file, bishop pair, non-linear queen exposure, and ultimately an offline retrain with positional features remain before true scientific saturation. The protected champion is newer and better-supported than the live exp-0052-era upload, but uploading is out of scope."
}

## Your task

1. Inspect the champion, the new record(s), and whatever retained
   losing/drawn-game evidence or profiling you need. Note recurring failure
   modes (e.g. threefold-repetition draws in balanced or better positions,
   endgame conversion, king-attack defence).
2. Judge whether the current search/move-ordering/quiescence/time-management
   tuning direction is saturated per the handoff's stop rule (>= 5 consecutive
   scientific non-improvements since the newest promotion AND no materially
   different, well-motivated experiment left in that direction).
3. Decide CONTINUE or STOP.
   - CONTINUE if a materially different, well-motivated experiment remains,
     whether in that direction or another (richer leaf evaluation such as
     mobility / pawn structure / passed pawns / bishop pair / rook-on-open-file
     / non-linear king safety; a small opening book designed to not defeat the
     model-move-ablation gate; persisted cross-move search state; something you
     identify from the evidence). Prefer genuinely new directions over more
     tuning.
   - STOP only for: a genuine infrastructure/auth/rules blocker
     (stop_condition "infrastructure_blocker"); or true scientific saturation
     (stop_condition "scientific_saturation") after a full evidence audit with
     no materially new well-motivated experiment anywhere.
4. If CONTINUE, specify the single next hypothesis. Constraints on it:
   - The candidate generator is one `claude -p` run restricted to editing
     agent.py and/or weights/. It has no Bash and cannot run training/ pipelines
     or produce a newly trained weights/model.json. So the hypothesis must be
     self-contained: a code change in agent.py, optionally with small
     hand-chosen or closed-form coefficients, or a change to how existing
     weights are used. If a direction truly needs an offline-trained artifact,
     say so explicitly and scope a self-contained first step the candidate can
     ship now (e.g. a single cheaply-computed feature with a conservative fixed
     coefficient) and flag the training follow-up for the human.
   - It must keep the trained model materially determining leaf evaluation and
     move selection, hold one CPU / 2 GB / no network, keep get_move returning a
     legal UCI move under the real clock, and not repeat a saturated family.
   - Give concrete guardrails (NPS budget, model-ablation-gate safety,
     reliability/flag risk).

## Required structured output

Return exactly one object matching the JSON Schema enforced by the Claude CLI.
Do not wrap it in Markdown, prose, code fences or XML tags. Put the important
analysis in `streak_assessment`, `recurring_failure_modes`, `rationale` and
`audit_summary`. For CONTINUE, `stop_condition` must be null and
`next_direction` must be a complete object. For STOP, `stop_condition` must be
`infrastructure_blocker` or `scientific_saturation`, and `next_direction` may
be null.