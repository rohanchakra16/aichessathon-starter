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
next_experiment in state.json: 94
Consecutive scientific non-improvements before this audit: 4

## Recent experiment registry (newest first, compact)

- exp-0093: inconclusive; HYPOTHESIS: Adding a cheap flat doubled-pawn (12 cp) and isolated-pawn (14 cp) penalty to `_model_evaluate`, computed from a per-file pawn tally folde
- exp-0092: inconclusive; HYPOTHESIS: Adding a flat 18 cp open-file / 9 cp half-open-file rook bonus to `_model_evaluate` (via two once-per-call pawn bitboards and a per-rook f
- exp-0091: inconclusive; HYPOTHESIS: Adding a cheap precomputed rank-scaled passed-pawn bonus to the leaf evaluation gives alpha-beta a monotone progress gradient in locked an
- exp-0090: inconclusive; HYPOTHESIS: Adding a cheap quadratic king-danger term to `_model_evaluate` — counting enemy-attacked squares of each king's one-step ring (capped at 6
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
  "completed_at": "2026-09-03T22:31:34.047140+00:00",
  "after_experiment": "exp-0092",
  "decision": "CONTINUE",
  "next_direction": {
    "title": "Doubled and isolated pawn-structure penalty in the leaf evaluation",
    "family": "leaf-evaluation",
    "hypothesis": "Add a cheap fixed pawn-structure penalty to `_model_evaluate`. Inside the existing `for colour, sign in ((side, 1.0), (not side, -1.0))` loop, when `piece_type == chess.PAWN`, accumulate a per-file pawn count into an 8-int list for the current colour (index by `chess.square_file(square)`, increment) at negligible cost, reusing the square iteration already happening for the piece-square lookup. After the piece loop for that colour, compute two structural counts from the 8-int list: (a) doubled_extra = sum(max(0, c - 1) for c in file_counts) and (b) isolated = sum(c for f, c in enumerate(file_counts) if c > 0 and (f == 0 or file_counts[f-1] == 0) and (f == 7 or file_counts[f+1] == 0)). Then `score += sign * (-DOUBLED_PENALTY * doubled_extra - ISOLATED_PENALTY * isolated)` with hand-chosen fixed constants DOUBLED_PENALTY ~= 12 cp and ISOLATED_PENALTY ~= 14 cp, applied flat with no phase scaling and no backward-pawn, hanging-pawn, pawn-majority, protected-passer or half-open-file interaction sub-terms in this first version so the mechanism is tested cleanly. A pawn that is both doubled and isolated simply receives both penalties. This gives alpha-beta a monotone reason to avoid creating its own structural weaknesses, to trade into structurally superior pawn formations, and to target the opponent's weak pawns as a concrete way to make progress in the balanced and slightly-better middlegames and rook endgames that currently end in repetition, and a reason to prefer sound structure in the simplified positions where the king later gets hunted.",
    "rationale": "The dominant retained failure modes are non-forced repetition draws from equal-or-better positions with no progress plan, and lost simplified endgames / king hunts downstream of passive, structurally poor positions. The trained model is piece-square only and has no pawn-structure term at all, so this is the largest missing classical evaluation component, not a refinement of something already present. It has never been tried (absent from the experiment registry and from both exhausted-mechanism lists), and unlike the exp-0092 rook-file term (0.492) and exp-0090 king-danger scalar (0.508) it targets a feature that is genuinely orthogonal to mobility, PSQT and material. It is preferred over the previously pre-authorised bishop-pair term because bishop pair is a near-constant material scalar that does little to create a progress gradient, whereas pawn weaknesses are the canonical lever for converting small edges and for both sides' endgame technique. It is fully self-contained: no move generation, no board.push, no popcount, no new weights artifact and no training pipeline - just two fixed constants and an 8-int tally folded into a loop that already runs.",
    "guardrails": "Magnitude: DOUBLED_PENALTY <= ~15 cp and ISOLATED_PENALTY <= ~16 cp, so a realistic worst case (say two doubled-extra pawns plus three isolated pawns) contributes under ~0.8 pawn, far below the model's ~118-170 cp margin RMSE, and no structural term approaches a minor piece; keep it flat with no phase scaling and no extra sub-terms in v1. NPS / leaf-cost budget: accumulate the per-file counts with one list index and increment per pawn inside the existing pawn branch of the piece loop, then one O(8) post-loop pass per side with no chess.popcount, no board.push, no second legal-move generation and no per-leaf allocation beyond two fixed-size 8-int lists; target added leaf cost within ~1.15x. exp-0091/exp-0092 reliability stress showed the worst single move at the 120 s clock was ~0.55 s against the 2.25 s ceiling and ~0.04 s against 0.75 s at the 1 s clock, so re-verify the full stress_time_left_ms grid (1 / 25 / 100 / 1000 / 120000) with required_failures 0 and init under 50 s. Must hold model-move-ablation minimum_strength_score >= 0.625 with >= 8 divergent moves at the 3 s ablation clock. Use only documented python-chess API (chess.square_file); never read library internals; edit only agent.py (weights/ unchanged). No change to _budget_seconds, the iterative-deepening loop, move ordering, quiescence, or the transposition table. If the paired arena is inconclusive, do NOT re-tune the two constants, add phase scaling, or bolt on backward-pawn / hanging-pawn / pawn-majority / passed-pawn-interaction sub-terms and call it a new experiment; move to the bishop-pair term, and if that is also inconclusive, flag an offline positional-feature retrain of weights/model.json (pawn-structure, king-shelter and rook features) for the human before declaring scientific saturation."
  },
  "audit_summary": "Champion is agent.py at commit 4a0c988 (recorded as exp-0089): PVS/LMR negamax with exchange-aware capture ordering and losing-capture pruning in quiescence, a depth-keyed transposition table, partial root-iteration salvage and a fast draw/terminal path, over a tapered learned piece-square model, plus the accepted fixed 3 cp/square net-mobility leaf term and a flat castling-rights scalar. The batch just run produced exp-0092: a flat 18 cp open-file / 9 cp half-open-file rook bonus added to _model_evaluate, which returned an inconclusive paired arena (17-29-18, score 0.492, CI [0.420, 0.564]) with CI, reliability stress, model-move-ablation, and resource/policy gates all clean and zero agent failures. It was not promoted, so rook-file activity is not in the champion. This is the third consecutive scientific non-improvement since the exp-0089 promotion (exp-0090 king-danger 0.508, exp-0091 passed pawns 0.531, exp-0092 rook files 0.492), against a stop threshold of five. Reviewing exp-0092's retained games, the failure modes are unchanged from the previous two audits: 28 of 64 games end in non-forced threefold repetition from equal-or-better positions with aimless piece shuffling and unadvanced passed pawns, and all 18 losses are checkmates skewed toward the candidate playing White via long king-hunts after passive simplification. The search/ordering/quiescence/time family is exhausted in its tested form (verified against the handoff exhausted list), but the leaf-evaluation direction is not: only mobility, king-danger, passed pawns and rook files have reached the arena, and pawn structure - the single largest classical evaluation term missing from a piece-square-only model - has never been tried and is absent from the registry and both exhausted lists. Decision: CONTINUE, stop_condition null. Next experiment adds a cheap flat doubled-pawn (~12 cp per extra pawn on a file) and isolated-pawn (~14 cp per pawn with no friendly pawn on adjacent files) penalty to _model_evaluate, computed from an 8-int per-file tally folded into the existing piece loop with no popcount, no board.push and no move generation, keeping the trained model dominant (worst realistic contribution under ~0.8 pawn vs the model's ~118-170 cp margin RMSE) and touching no search, time-management or protected surface. If inconclusive, the bishop-pair term remains, followed by flagging an offline positional-feature retrain of weights/model.json for the human before declaring saturation. Uploading remains out of scope; the live upload is a separate older exp-0052-era artifact and this audit does not touch it."
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