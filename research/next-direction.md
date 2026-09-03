# Next research direction

Set by the Claude evidence audit on 2026-09-03T21:29:03.647980+00:00 (pre-first-batch).
It steers the next candidate's hypothesis only; the controller and its
protected framework remain the sole evaluation and promotion authority.

- Title: Cheap mobility differential in leaf evaluation
- Family: leaf-evaluation
- Hypothesis: Adding a small fixed-coefficient net-mobility term (side-to-move minus opponent attacked-square count over non-king pieces, computed once per leaf from attack bitboards) to the learned PSQT score gives alpha-beta a monotone reason to prefer active, space-gaining moves over passive shuffles, lowering the threefold-repetition draw rate without retraining.
- Rationale: agent.py's _model_evaluate is a bare tapered PSQT with only castling terms and no positional feature of any kind; exp-0087's retained PGNs show a ~53% draw rate that is overwhelmingly threefold_repetition from non-forced balanced or better middlegames with visible piece shuffling, the textbook symptom of a flat evaluator. The only prior positional-feature attempt was an unrelated king-safety residual with no offline signal that never reached the arena, so it is not evidence against mobility.
- Guardrails: Compute mobility via ~12-20 popcount(board.attacks(sq)) lookups once per leaf, never a second full legal-move generation (leaf eval is the hot path at every quiescence stand-pat). Keep the coefficient fixed at roughly 2-4 centipawns per net attacked square so a typical ~10-square delta stays well under a third of a pawn and far below the model's ~118 cp RMSE, keeping the trained model dominant: must hold minimum_strength_score >= 0.625 and >= 8 divergent moves under the model-ablation gate. Re-profile local NPS and the full stress_time_left_ms grid before the arena; exp-0087's max single move was 0.968 s vs the 2.25 s ceiling at the 120 s clock and 0.041 s vs 0.75 s at 1 s, so a ~1.5-2x leaf-cost increase is tolerable only if verified. No change to time budgeting or the iterative-deepening loop; avoid per-leaf object allocation.

Constraints that always hold: edit only agent.py and/or weights/; the
trained model must keep materially determining leaf evaluation and move
selection; one CPU / 2 GB / no network; do not repeat a saturated family
by only changing a parameter.

Audit summary: The champion exp-0066 has withstood 12 consecutive search/ordering/quiescence/time candidates spanning every mechanism the handoff lists as exhausted, so that direction is saturated. Its evaluator, however, is a positionally blind PSQT, and exp-0087's arena shows a ~53% draw rate that is almost entirely non-forced threefold repetition from balanced or better positions with piece shuffling. The next experiment pivots to leaf-evaluation richness via a single cheap, centipawn-scaled net-mobility differential: a self-contained agent.py change that keeps the trained model materially dominant while giving the search a gradient away from repetition. If it does not help, adjacent cheap positional terms (passed pawns, rook-on-open-file, bishop pair) and, as a human follow-up, an offline retrain with positional features remain as further leaf-evaluation directions before true scientific saturation.
