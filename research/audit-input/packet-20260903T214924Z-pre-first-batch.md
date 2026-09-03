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
  "completed_at": "2026-09-03T21:31:35.774199+00:00",
  "after_experiment": "exp-0088",
  "decision": "CONTINUE",
  "next_direction": {
    "title": "Cheap net-mobility differential in leaf evaluation",
    "family": "leaf-evaluation",
    "hypothesis": "Adding a small fixed-coefficient net-mobility term to _model_evaluate (side-to-move minus opponent count of squares attacked by non-king pieces, summed once per leaf from board.attacks() bitboards) gives alpha-beta a monotone reason to prefer active, space-gaining moves over passive shuffles, lowering the non-forced threefold-repetition draw rate without retraining the model.",
    "rationale": "agent.py's _model_evaluate has no positional feature of any kind, and exp-0087's retained PGNs show a ~53% draw rate that is almost entirely threefold_repetition from non-forced balanced-or-better positions with piece shuffling, the textbook symptom of a flat evaluator. This mechanism was recommended by the prior audit but only ever failed as a generator-governance slip, so it is untested, not exhausted; the sole prior positional attempt (a six-feature linear king-safety residual) had no offline signal and never reached the arena, so it is not evidence against mobility.",
    "guardrails": "Compute mobility with ~12-20 len(board.attacks(sq)) / chess.popcount lookups once per leaf; never a second full legal-move generation and no per-leaf object allocation, since the leaf is the hot path at every quiescence stand-pat. Use only documented python-chess API (board.attacks, chess.SquareSet, chess.popcount) and never inspect library internals or any path outside agent.py and weights/. Fix the coefficient at roughly 2-4 centipawns per net attacked square so a typical ~10-square delta stays well under a third of a pawn and far below the model's ~118 cp RMSE, keeping the trained PSQT materially dominant: must hold model-ablation minimum_strength_score >= 0.625 and >= 8 divergent moves. Re-profile local NPS and the full stress_time_left_ms grid (1/25/100/1000/120000) before the arena; exp-0087's worst single move was 0.968 s vs the 2.25 s ceiling at the 120 s clock and 0.041 s vs 0.75 s at 1 s, so up to ~1.5-2x leaf cost is tolerable only if verified with required_failures 0. No change to time budgeting or the iterative-deepening loop."
  },
  "audit_summary": "The champion exp-0066 has a mature search but a positionally blind tapered-PSQT leaf, and exp-0087's arena shows ~53% draws that are almost entirely non-forced threefold repetition from balanced-or-better positions with piece shuffling. Twelve consecutive search/ordering/quiescence/time candidates have saturated that direction, while leaf evaluation is unexplored. The next experiment adds a single cheap centipawn-scaled net-mobility differential to _model_evaluate, self-contained in agent.py with a hand-chosen conservative coefficient, keeping the trained model dominant while giving the search a gradient away from repetition; this was the prior audit's pick but only ever failed as a generator-governance violation, so it is genuinely untested. If it does not help, adjacent cheap positional terms (passed pawns, rook-on-open-file, bishop pair, non-linear king safety) and an offline retrain with positional features remain before true scientific saturation."
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