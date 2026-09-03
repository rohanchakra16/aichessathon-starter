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

Champion commit: e9da1e556cd43d0045d4733e9dd313512ab128f4
Live competition upload: older exp-0052-era champion (do not upload; out of scope).
next_experiment in state.json: 88
Consecutive scientific non-improvements before this audit: 12

## Recent experiment registry (newest first, compact)

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
- exp-0069: inconclusive; Restoring exact stalemate scoring at quiescence leaves would preserve exp-0066's terminal-check speed gain without mis-evaluating a drawn terminal pos
- exp-0068: inconclusive; HYPOTHESIS: Bucketing the halfmove clock in the transposition-table key (exact only within 80 plies of the 50-move rule, otherwise zero) will sharply 

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
  "completed_at": "2026-09-03T21:15:51.821455+00:00",
  "after_experiment": null,
  "decision": "CONTINUE",
  "next_direction": {
    "title": "Cheap mobility differential added to leaf evaluation",
    "family": "leaf-evaluation",
    "hypothesis": "Adding a small fixed-coefficient legal-mobility differential (side-to-move legal/pseudo-legal move count minus opponent's) to the learned PSQT score gives the search a non-zero gradient favoring active, space-gaining moves over passive shuffles, reducing the repetition-draw rate seen in exp-0087's retained games without requiring retraining.",
    "rationale": "agent.py's _model_evaluate has no mobility, pawn-structure, or piece-activity term of any kind; the only prior positional-feature attempt (six-feature king-safety residual) tested a different, mechanistically unrelated signal and is not evidence against mobility. exp-0087's PGNs show repeated multi-ply piece shuffles ending in threefold repetition from non-trivial middlegame positions, the textbook symptom of an evaluator that can't tell 'improving' from 'idle' once tactics run out.",
    "guardrails": "Compute mobility via reused/cheap move data, not a second full legal-move generation per leaf (leaf eval runs at every quiescence node/stand-pat, so an extra board.legal_moves() call would roughly double node cost there); prefer pseudo-legal attack-bitboard popcounts or reusing the move list already generated one ply up. Keep the coefficient small and fixed so the trained 770-weight model still dominates the score (must keep minimum_strength_score >= 0.625 against the model-ablated baseline) while still producing >= minimum_move_differences (8) divergent moves under the model-ablation gate. Must not push single-move time over the 0.75s/2.25s reliability ceiling; verify with local NPS profiling before committing to the arena."
  },
  "audit_summary": "The champion (exp-0066) has now fended off 12 consecutive search/ordering candidates spanning every mechanism in that family, and the handoff's own registry confirms that family is exhausted in tested form. The champion's evaluator, however, is a bare tapered PSQT with zero positional features beyond castling rights, and this batch's retained games show a 53% repetition-draw rate consistent with that gap. The next experiment should pivot to leaf-evaluation richness via a cheap mobility differential, a materially different, code-and-evidence-grounded direction that keeps the trained model dominant while giving the search a reason to avoid shuffling."
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

## Required output

Write your analysis, then end your reply with exactly one block:

<AUDIT_DECISION>
{
  "decision": "CONTINUE" | "STOP",
  "stop_condition": null | "infrastructure_blocker" | "scientific_saturation",
  "streak_assessment": "one or two sentences",
  "recurring_failure_modes": ["..."],
  "next_direction": {
    "title": "short name",
    "family": "leaf-evaluation | opening-book | search-state | evaluation-training | other",
    "hypothesis": "one sentence, materially different from every saturated family",
    "rationale": "grounded in champion code, retained games, or the registry",
    "guardrails": "NPS / ablation-gate / reliability considerations"
  },
  "audit_summary": "2-4 sentences capturing the state of the research"
}
</AUDIT_DECISION>

If decision is STOP, next_direction may be null.