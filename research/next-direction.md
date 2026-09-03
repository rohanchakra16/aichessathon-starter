# Next research direction

Set by the Claude evidence audit on 2026-09-03T21:15:51.823027+00:00 (pre-first-batch).
It steers the next candidate's hypothesis only; the controller and its
protected framework remain the sole evaluation and promotion authority.

- Title: Cheap mobility differential added to leaf evaluation
- Family: leaf-evaluation
- Hypothesis: Adding a small fixed-coefficient legal-mobility differential (side-to-move legal/pseudo-legal move count minus opponent's) to the learned PSQT score gives the search a non-zero gradient favoring active, space-gaining moves over passive shuffles, reducing the repetition-draw rate seen in exp-0087's retained games without requiring retraining.
- Rationale: agent.py's _model_evaluate has no mobility, pawn-structure, or piece-activity term of any kind; the only prior positional-feature attempt (six-feature king-safety residual) tested a different, mechanistically unrelated signal and is not evidence against mobility. exp-0087's PGNs show repeated multi-ply piece shuffles ending in threefold repetition from non-trivial middlegame positions, the textbook symptom of an evaluator that can't tell 'improving' from 'idle' once tactics run out.
- Guardrails: Compute mobility via reused/cheap move data, not a second full legal-move generation per leaf (leaf eval runs at every quiescence node/stand-pat, so an extra board.legal_moves() call would roughly double node cost there); prefer pseudo-legal attack-bitboard popcounts or reusing the move list already generated one ply up. Keep the coefficient small and fixed so the trained 770-weight model still dominates the score (must keep minimum_strength_score >= 0.625 against the model-ablated baseline) while still producing >= minimum_move_differences (8) divergent moves under the model-ablation gate. Must not push single-move time over the 0.75s/2.25s reliability ceiling; verify with local NPS profiling before committing to the arena.

Constraints that always hold: edit only agent.py and/or weights/; the
trained model must keep materially determining leaf evaluation and move
selection; one CPU / 2 GB / no network; do not repeat a saturated family
by only changing a parameter.

Audit summary: The champion (exp-0066) has now fended off 12 consecutive search/ordering candidates spanning every mechanism in that family, and the handoff's own registry confirms that family is exhausted in tested form. The champion's evaluator, however, is a bare tapered PSQT with zero positional features beyond castling rights, and this batch's retained games show a 53% repetition-draw rate consistent with that gap. The next experiment should pivot to leaf-evaluation richness via a cheap mobility differential, a materially different, code-and-evidence-grounded direction that keeps the trained model dominant while giving the search a reason to avoid shuffling.
