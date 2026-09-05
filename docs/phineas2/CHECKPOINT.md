# Phineas 2 — major architecture rebuild

Branch: `phineas2` (worktree). Never touch `main` / the champion while this runs.
Started 2026-09-04. Goal: reliable engine at ~1800 (min) / 2000 (competitive) / 2200+ (stretch)
under 120+0.5, one core, that clearly beats the current protected champion (exp-0089, `4a0c988`).

## CURRENT STATUS (2026-09-05) — v9 is submission-decision-ready

**Internal champion: `phineas2-champion-v9-deeperbook`** (tag; HEAD of the `phineas2` branch,
commit `849dd6b` / engine at `bb4ad10`). Nothing on `main` has been touched; the protected champion
is intact as the safe fallback. Nothing has been uploaded.

Validation, all at the real 120s+0.5s clock, one core, alternating colours over preregistered openings,
Phineas 2 in its own fresh process per game (mirrors the competition protocol):

| opponent | games | score | 95% CI | failures |
|---|---|---|---|---|
| **current protected champion** | 16 | **0.969** (+15 =1 -0, all wins by checkmate) | [0.909, 1.000] | 0 |
| Stockfish 18 @ UCI_Elo 1800 | 30 | 0.567 | [0.395, 0.738] | 0 |
| Stockfish 18 @ UCI_Elo 2000 | 30 | 0.500 | [0.354, 0.646] | 0 |
| Stockfish 18 @ UCI_Elo 2200 | 30 | 0.483 | [0.327, 0.640] | 0 |

- **The mandate's core success criterion — statistically meaningful head-to-head superiority over the
  protected champion — is met emphatically** (0.969, CI clear of 0.9).
- Against the strength-limited Stockfish ladder, Phineas 2 sits around 2000-2100-equivalent: solidly
  ahead of 1800, essentially dead-even across 2000-2200 (the ladder's ~200-Elo resolution is too coarse
  to separate those cleanly). Roughly at, not clearly past, the ~2200 bar the user has since said is
  needed for a London-final seat.
- Reliability: 0 crashes / flags / illegal moves across ~400+ validation games; cold-start import
  ~13-14s (budget 90s); submission zip 4.4 MB unzipped (cap 50 MB); shipped imports are numpy, numba
  (guarded), python-chess + chess.syzygy, and stdlib only — no network, no subprocess, no third-party
  engine; `ruff` + `mypy` clean repo-wide; `pytest tests/autoloop` 82/83 (the one failure is the
  champion-specific offline-retraining splice test, which a from-scratch agent.py was never routed
  through — not a regression, documented).

**Open decision for the user:** submit v9 now, or continue to the one remaining Step 8 lever (a
retrained compact evaluator — a larger, riskier, separate data-generation + training effort, the item
most likely to lift the whole 2000-2200 band rather than patch one more failure mode). Everything
below is the detailed trail.

## Repo reconciliation (2026-09-04 ~13:25)
- Champion: exp-0089 `4a0c988009ecae163ac09368f92e4a792dac7568`. exp-0090..0099 all failed to displace it.
- exp-0099 (Variant C material calib, commit `090f18c`, branch `autoloop/candidate-0099`): fast arena
  17-37-10 / 0.5547 / **inconclusive**. NOT promoted. Candidate branch preserved.
- exp-0099 real-clock confirmation (`controller.py --clock-promotion exp-0099`) is **RUNNING** in the
  user's terminal (controller.lock held by PID 6815, worktrees `.autoloop/worktrees/clock-exp-0099-*`,
  no `confirmations/exp-0099-prospective-real-clock.json` yet). DO NOT launch a duplicate; it is one-shot.
- Live submission: v3, sha `55fec84c6dab`, tied to the exp-0089 champion. Do not upload without approval.
- Untracked user files preserved: `.claude/ benchmark_phineas.py champions_tournament.py phineas_ui/`.

## Rules (fetched from aichessathon.com, authoritative)
- 120s + 0.5s/move, 90s init, 1 core, 2GB, no net/GPU. 300 plies -> adjudication. FIDE draws (python-chess).
- Env has torch(CPU), numpy, python-chess, onnxruntime, numba preinstalled. `requirements.txt` ignored.
- **Third-party engines prohibited** incl. any wrapper/port/translation. A classical search you wrote is a
  FULL entry. AI help to *write* the code is fine.
- **Learned nets**: must be trained by us; training on engine-labelled positions is ALLOWED; may NOT start
  from a published net; may NOT ship an engine-eval lookup DB. onnx/safetensors/pt are fine (not "native").
- Opening books (`chess.polyglot`) and syzygy tablebases (`chess.syzygy`) permitted as shipped data <=50MB.
- Source must be judge-readable; no obfuscation; no native binaries in the zip.

## KEY BLOCKER for the >2000 path — RESOLVED 2026-09-04
User authorized ONE narrow protected change: add `numba==0.67.0` + `llvmlite==0.49.0` (its required
compatible dep) to `.autoloop/protected/Dockerfile`, matching the real competition runtime. Done in
commit `8e33e03` on `main` (own commit, previous version in history, protected_hash recomputed to
`6b2de3b47d6c2460...`, 83 protected tests still pass). torch/onnxruntime intentionally NOT added —
only on demonstrated + separately-approved need. Numba is now the primary hot-path target; the
pure-Python fallback (`P2_NO_NUMBA=1`) stays as the correctness reference and a safety net if JIT
init ever fails in the container.

## Current-engine profile (champion, under concurrent load; idle ~1.5-2x faster)
| position    | depth@2s | nps    | q-frac | TT hit% |
|-------------|----------|--------|--------|---------|
| middlegame  | **4**    | 30k    | 51%    | 31%     |
| tactical    | **4**    | 12k    | 70%    | 16%     |
| endgame     | 8 (cap)  | 80k    | 40%    | 55%     |

cProfile (2.0s midgame, 7.28M calls): `_model_evaluate` ~40%, SEE (`_static_exchange`+`_least_valuable_
recapture`) ~25% (move-gen based, very slow), python-chess move gen ~30%. TT key includes `depth` ->
near-zero cross-depth reuse. `MAX_DEPTH=8` hard cap. Time budget fixed 2.0s regardless (broken).

## Why ~1400-1600: depth 4 in tactical/midgame positions. Need depth 7-9 for 1800, 9-12 for 2000+.

## Plan (staged; correctness is an absolute gate at every stage)
- P0  scaffolding: phineas2 branch, dev test/bench harness, perft harness, Stockfish sparring harness.
- P1  fast board: numba flat-array bitboard, incremental Zobrist + phase, make/unmake. DONE (`4205f10`).
      GATE MET: exact perft startpos d6 / kiwipete d5 / pos3-6 d5-d7; differential vs python-chess on
      1600 random + dedicated edge FENs; make/unmake restores every field incl Zobrist. ~8M nps perft
      jitted, ~4.3s JIT warm-up. (PST arrays / incremental eval deferred to P4; phase is tracked live.)
- P2  search core: iterative deepening PVS + aspiration; TT (hash-keyed, depth-in-entry, bucketed,
      no full clear); ordering = TT move / SEE captures (bitboard SEE) / 2 killers / history / countermove.
      GATE: WAC/ECM tactical suite solve rate up vs champion; deterministic node screens.
- P3  pruning: null-move (zugzwang-guarded), LMR, futility/razoring, mate-distance, 1-ply check ext.
      quiescence: SEE-filtered captures + queen promotions + 1 check ply + delta pruning.
- P4  eval: tapered PST (reuse 770 weights as init) + phase-aware king safety / passed pawns / rook files
      / cheap mobility; repetition contempt + progress term for winning-position conversion.
- P5  time mgmt: clock-bank + increment allocation; panic extension on fail-low / big eval drop.
- P6  learned eval (optional): small linear/NNUE trained on Stockfish WDL+eval over diverse + loss
      positions; incremental or tiny so it's fast; onnx OR pure-numpy so it works in the stripped CI.
- Validation ladder per stage: local tactics -> short H2H vs champion -> 64-100 paired games ->
  exact-clock -> Stockfish ladder 1600 -> 1800 -> 2000 -> 2200.

## STATUS
- [x] reconciliation, rules, profiling, plan
- [x] protected Dockerfile numba parity (`8e33e03`, main)
- [x] P0 scaffolding + P1 numba bitboard core, perft-validated (`4205f10`)
- [x] P2 search core (`67d23b8`): njit iterative-deepening PVS, hash TT (depth-in-entry, generation-
      aged, no full clear), TT/MVV-LVA/killer/history ordering, quiescence. Eval = faithful port of
      the shipped model (tapered PST + castling + bias; mobility term dropped, per the A/B/C study).
      GATE MET: 12-game H2H vs the champion at 8s+80ms, +12 =0 -0, all by checkmate.
- [x] P3 pruning (`50db414`): mate-distance, reverse-futility/static-null, null-move (zugzwang-guarded
      via phase + non-PV + not-in-check), frontier futility, LMR with verification re-search.
      Effect: same middlegame FEN reaches depth 13 in ~6.5s vs depth 8 before (branching factor
      ~5-6 -> ~2.5). Re-ran the H2H gate to confirm no regression: +11 =1 -0, score 0.958.
- [x] time-management hardening (`d164769`): a dev Stockfish-sparring run (see below) flag-lost a
      game; root cause was a flat "+400ms" floor plus reliance on the increment to cover ~5ms of
      fixed per-call overhead outside the timed search. Fixed with a bank-proportional hard reserve
      and an explicit overhead subtraction. Verified flag-free over two synthetic full-length stress
      games (8000+80ms and 3000+30ms) driving agent.get_move() directly against a shrinking clock.
- [x] first Stockfish calibration (dev-only, NOT a competition-clock claim — 15s+150ms local
      Stockfish 18, UCI_LimitStrength): elo=1400 +3=0-1 (the pre-fix flag loss), elo=1600 +6=0-0 all
      checkmates, elo=1800 +2=1-3 roughly even. Crossover currently sits around SF-1700-1800-ish at
      this fast clock -- encouraging (matches/exceeds the ~1800 minimum target) but NOT yet a
      competition-clock or statistically adequate-sample claim. No crashes, no flags, no illegal
      moves across ~30 dev games total after the time-management fix.
- [x] MAJOR COURSE CORRECTION per user instruction (2026-09-04): frozen the P1-P3 + time-fix state as
      an immutable reference, tag `phineas2-baseline-p1p3` = commit `466d6d2` (pushed to origin).
      Every subsequent change is now a separately-committed, separately-ablated candidate compared
      against this tag, not a bundled rewrite. "~1700-1800-ish" language retired -- not a valid claim
      from 6-game fast-clock samples; only real 120+0.5 games at adequate sample size count.
- [x] reliability verification (`3b06f01`): no Docker in this sandbox, so "fresh container" is
      approximated by fresh OS subprocesses (the variable that matters -- a from-scratch process
      paying the numba JIT cost once -- is exercised directly, not skipped). 5x fresh-process cold
      start: import ~12.2-12.6s (<<60s budget), first move exactly the requested budget (not JIT
      lag), no crashes. Submission zip: 8 files, 24KB/88KB (unzipped), <<50MB. Import scan of shipped
      files: numpy + guarded numba + stdlib only, no python-chess, no network, no subprocess; the
      only "stockfish" string anywhere is inert JSON provenance in model.json. Pure-Python fallback
      (P2_NO_NUMBA=1) verified to still produce a legal move end-to-end. ruff/mypy clean repo-wide.
      pytest tests/autoloop: 82/83 -- the one failure is expected and not a regression (it checks the
      champion-specific offline-retraining residual-splice anchor, which a from-scratch agent.py was
      never routed through; not modified, still correctly validates the champion on main).
- [x] exact-clock validation harness (`3b06f01`): scripts/p2_worker.py (one-game subprocess, mirrors
      "one process per game", exposes non-contract LAST_INFO telemetry) + scripts/
      p2_exact_clock_validation.py (drives it vs a strength-limited local Stockfish 18 at a real
      clock, saves PGN + JSON per run: W/D/L, score, 95% CI, by-colour, move-time/depth stats,
      failures, winning-position draws, over a fixed preregistered opening list).
      METHODOLOGICAL FIX: Worker/play_game take a `target_dir` (default: live worktree); a validation
      run must point --target-dir at a separate, detached `git worktree add` checkout of the tag/
      commit under test (e.g. /tmp/claude-501/p2-baseline for the frozen baseline) -- never the live
      worktree while P4 development is ongoing there, since every game spawns a fresh subprocess that
      re-imports agent.py from disk and would silently pick up in-progress edits.
- [x] Step 3 (user-mandated): 14-game 120+0.5 baseline vs Stockfish elo 1800, isolated frozen-baseline
      checkout (`/tmp/claude-501/p2-baseline` = phineas2-baseline-p1p3). First attempt crashed on a PGN-
      export bug for non-startpos openings (fixed in `1c34034`, verified offline with no engine cost
      before rerunning). Result: **+5 =3 -6, score=0.464, 95% CI [0.233, 0.696], 0 failures.** Roughly
      even at the real clock -- materially humbler than the fast-clock screening suggested, exactly the
      outcome the user warned was possible ("we cannot assume the real clock benefits Phineas more than
      strength-limited Stockfish"). By colour: white 3-2-2 (0.571), black 2-1-4 (0.357) -- weaker as
      black in this sample. This is the reference point Step 6's larger batch will be compared to.
- [x] P4 candidate 1a (`bf04f4b`): weights/p2core.see -- bitboard SEE, standard swap algorithm,
      handles en passant / promotion / discovered attackers. 7 hand-verified unit tests
      (scripts/p2_see_test.py), all pass.
- [x] P4 candidate 1b (`bf4ba9e`): wired SEE into move ordering and quiescence (see the commit for the
      mechanism). ACCEPTED as **phineas2-champion-v2-see** after ablation vs phineas2-baseline-p1p3:
      +12 =3 -5, score=0.675, 95% CI [0.488, 0.862], 0 failures.
      METHODOLOGICAL FIX during this candidate: the exact-clock/ablation harness always imported
      agent.py from the *live* worktree, so an in-flight validation run picks up concurrent P4 edits on
      every fresh per-game subprocess. Fixed by adding a `target_dir`/`--candidate-dir`/`--baseline-dir`
      parameter and always pointing runs at a separate, detached `git worktree add --detach <tag>`
      checkout -- never the branch tip while it's being edited. This discipline was then violated once
      more by accident while drafting candidate 3 during candidate 2's ablation (caught after 1 game,
      no bad data used) and is now followed strictly: never edit the live tree while any ablation or
      validation job targeting it is in flight.
- [x] P4 candidate 2 (repetition preference at the root): implemented, verified in isolation
      (a synthetic fabricated-repetition test confirms the ordering nudge touches exactly the intended
      move by exactly the intended amount, in both directions, and is a no-op at direction=0), ablated
      vs phineas2-champion-v2-see: +8 =5 -7, score=0.525, CI [0.336,0.714], 0 failures.
      **REJECTED** -- not a demonstrated improvement (see docs/phineas2/rejected-experiments.md).
      Branch tip reset past it; the commit (`6af3ed8`) remains reachable by hash, not deleted.
- [x] P4 candidate 3 (`bd6e8f6`): king safety (pawn-shield proxy) + development, both phase-scaled the
      same direction as the mg/eg PST blend. ACCEPTED as **phineas2-champion-v3-kingsafety-dev**:
      +11 =3 -6, score=0.625, 95% CI [0.431, 0.819], 0 failures.
- [x] P4 candidate 4 (`9b45cb4`): passed pawns, phase-scaled the *opposite* direction from candidate 3
      -- more weight toward the endgame, since converting a passed pawn is an endgame concern. A flat
      PST cannot express this (depends on where the *other* pawns are). ACCEPTED as
      **phineas2-champion-v4-passedpawn**: +12 =4 -4, score=0.700, 95% CI [0.525, 0.875], 0 failures --
      first candidate whose CI lies entirely above 0.5. King activity was considered alongside this one
      and deliberately deferred: PST_EG already differentiates king squares by endgame value, so an
      explicit centralisation term risked double-counting rather than adding a new signal.
- [x] P4 candidate 5 (`f6208fb`, last per the user's order, highest risk given the champion's own
      mobility term was previously found net harmful): properly-weighted bitboard mobility --
      per-piece-type weights (knight/bishop 4cp, rook 2cp, queen 1cp per reachable non-own square,
      reusing the search's own attack generators, no second per-square pass) instead of one flat
      weight dominated by queen-mobility noise, and no unrelated material-calibration bug for it to
      fight (PST_MG/PST_EG already carry the trained material faithfully). ACCEPTED as
      **phineas2-champion-v5-mobility**: +17 =0 -3, score=0.850, 95% CI [0.694, 1.000], 0 failures --
      the strongest single-candidate result of the sequence.
- [x] **P4 COMPLETE.** All five of the user's named priority items resolved: (1) SEE accepted,
      (2) repetition preference rejected then revived (see below), (3) king safety + development
      accepted, (4) passed pawns accepted, (5) mobility accepted.
- [x] Step 6: 30-game 120+0.5 confirmation vs Stockfish elo 1800 on phineas2-champion-v5-mobility:
      **+16 =2 -12, score=0.567, 95% CI [0.395, 0.738], 0 failures.** Real improvement over the Step 3
      baseline (0.464), but the CI still straddles 0.5 -- not yet a statistically airtight superiority
      claim. By colour: white 0.700, black 0.400 -- confirms the weak-as-Black pattern is real (note:
      this is the opposite phrasing from the user's step 8 "weak White openings" -- the actual evidence
      says weak as Black, flagged explicitly rather than silently reconciled).
- [x] Step 7 ladder screening, done as small real-clock checks rather than fast-clock ones after the
      fast-clock elo-1800 screen (0.417) turned out to badly understate the real-clock result (0.567) --
      Stockfish's UCI_LimitStrength evidently doesn't scale with time the way Phineas's own search does,
      so a fast-clock screen systematically misclassifies this matchup:
        - elo 2000, 8 games @ 120+0.5: +7 =0 -1, score=0.875, 95% CI [0.646, 1.000] -- clearly competitive.
        - elo 2200, 8 games @ 120+0.5: +3 =1 -4, score=0.438, 95% CI [0.116, 0.759] -- roughly even,
          n too small to call either way. One of these 8 games (below) directly motivated candidate 2's
          revival.
- [x] **P4 candidate 2 REVIVED AND ACCEPTED** (commit `832a700`, tag **phineas2-champion-v6-repetition**):
      the elo-2200 screen's game 8 showed v5-mobility's own search sit at self-evaluated +666cp in a
      won king+bishop+2-pawns endgame for ten consecutive moves (depths 22-37, including after
      promoting a new queen) while shuffling into a threefold-repetition draw -- the exact failure mode
      the rejected candidate targeted, observed live. Reapplied the unchanged mechanism from `6af3ed8`
      and re-ablated at a larger sample (30 games vs 20 the first time): **+18 =6 -6, score=0.700, 95%
      CI [0.557, 0.843], 0 failures** -- clean accept, CI entirely above 0.5. Full writeup in
      docs/phineas2/rejected-experiments.md (kept there under the original entry, marked as revived,
      rather than moved -- the rejection-then-revival history is itself the useful record).
      **Current internal champion: phineas2-champion-v6-repetition, HEAD of the `phineas2` branch.**
- Cross-checked against the actual champion's last 15 live rated-round games (user-supplied PGNs/logs,
  2026-09-04): champion scored 6/15 = 0.40 live (white 0.375, black 0.286 -- same weak-as-Black
  direction independently confirmed), with 4/15 games (27%) ending in threefold repetition and two
  fast tactical collapses (mate in 20 and mate by move 31) resembling exactly the king-safety/tactical
  weaknesses Phineas 2's P4 work targeted. Not code-relevant (Phineas 2 shares no implementation with
  the champion) but useful independent corroboration that the diagnosed problem classes are real and
  that repetition-of-winning-positions is a recurring pattern worth the P4 candidate-2 attention it got.
- [x] Re-screened v6 vs elo 2200 at a larger real-clock sample (16 games, doubling the earlier 8):
      +4 =3 -9, score=0.344, 95% CI [0.137, 0.551]. Combined with the earlier 8-game screen (both on
      v6 code, before and after -- the first 8 predate v6, technically mixed vintage, but the direction
      is consistent): 24 games total, W=7 D=4 L=13, score=0.375. This is now a fairly clear "not yet
      competitive at 2200" signal -- per step 7, do NOT spend a large batch confirming 2200 right now.
      Inspected the 3 draws in the 16-game screen: game 2 is the mechanism working *correctly* (Black's
      own score was ~-29966, i.e. a near-mate losing score, and it correctly steered into a repetition
      draw rather than accepting the loss -- exactly "still takes a draw from a losing position"); game
      12 plateaus at a genuine 0 (drawn king-and-pawn-ish ending, not a bug); but game 9 (White, +268cp
      rook-shuffle repeated away down to +90 before the draw) shows the v6 fix is not a complete
      solution -- it can only pick among moves the search already judged tied, and some technical
      endgame conversions are apparently still beyond the search's horizon to find a genuinely better
      alternative to shuffling. A real, expected limitation, not a regression; the ablation still shows
      net benefit. Per-colour split in this batch reversed from Steps 3/6 (white 0.1875, black 0.5 here)
      -- likely small-sample (n=8/colour) noise given openings differ per colour slot, not a stable
      finding; do not treat "weak as White" as established from this alone.
- [x] Full 30-game 120+0.5 confirmation at elo 2000, v6-repetition: **+10 =10 -10, score=0.500, 95% CI
      [0.354, 0.646], 0 failures.** Regression to the mean from the 8-game screen's 0.875 -- the small
      sample badly overestimated. This is the number to trust: Phineas 2 currently plays almost exactly
      Stockfish-2000-equivalent strength at the real clock. Colour split this time: white 0.500, black
      0.500 -- perfectly even, directly contradicting the weak-as-Black pattern from Steps 3/6 and the
      live cross-check. Combined with the elo-2200 batches' colour reversal (v6: white 0.1875, black
      0.5), the colour effect is evidently NOISE across these sample sizes, not a stable structural
      weakness -- retracting the earlier "confirmed weak-as-Black" framing; it was real in two samples
      and absent/reversed in two others.
      The ladder is now coherent and monotonic: elo 1800 -> 0.567 (30 games), elo 2000 -> 0.500 (30
      games), elo 2200 -> 0.375 (24 games pre-tablebase). Phineas 2 sits at roughly the 2000 mark itself,
      not yet the 2200 the user has since confirmed is the actual qualification bar for the London final
      (top of the live leaderboard already past 2100 after day one, 235 teams, seats decided by 24-40
      game... by Swiss ranking, not a fixed threshold, but 2200+ is the realistic target to be safe).
- [x] Step 8 candidate 1 (`2d5d128`): Syzygy 3-4-piece WDL+DTZ tablebases (weights/syzygy/, ~4.3MB,
      weights/p2tb.py). Directly targets the diagnosed conversion-failure pattern (a large advantage
      shuffled into a draw in a simplified endgame) with an exact, provably-safe-by-construction fix
      (maximises our own WDL category first, so a win can never become a draw/loss under this mechanism;
      DTZ breaks ties toward real progress). ACCEPTED as **phineas2-champion-v7-tablebase** after
      ablation vs v6-repetition: +17 =5 -8, score=0.650, 95% CI [0.496, 0.804], 0 failures.
      Re-screened v7 at elo 2200 (16 games @ 120+0.5) to see whether it closed the 2200 gap:
      **+4 =2 -10, score=0.312, 95% CI [0.103, 0.522].** Not better than v6's number there, though
      notably `winning_position_draws: 0` this time (down from 1) -- the specific pathology the
      tablebase targets did stop showing up in this sample, consistent with the ablation's positive
      result, but it was evidently a minor contributor to the overall 2200 gap, not the primary one.
      Most 2200 losses are ordinary being-outplayed, not thrown-away wins. Accepted regardless: the
      ablation shows a genuine, safe improvement in general play even though it didn't move the 2200
      needle specifically -- these are different questions (does it help vs. does it close THIS gap).
- [x] Step 8 candidate 2 (`074cf22`): 126-entry opening book, 8 plies deep, branch 2, generated offline
      via scripts/p2_book_gen.py (local Stockfish as a development oracle only, ranking python-chess's
      own legal moves -- nothing Stockfish-derived beyond the resulting position->move pairs ships).
      Every shipped move independently re-verified legal from its keyed position. ACCEPTED as
      **phineas2-champion-v8-book** after ablation vs v7-tablebase: **+20 =6 -4, score=0.767, 95% CI
      [0.638, 0.895], 0 failures** -- the strongest single-candidate result of the entire P4/Step-8
      sequence, CI comfortably clear of 0.5. (Ablation openings already run several plies deep before
      either side moves, limiting the book's usable depth there -- the effect size suggests real value
      beyond opening quality alone, plausibly including the clock time an instant ~0.1ms lookup banks
      versus the multi-second search it replaces, compounding over a full game.)
      Screened v8 at elo 2200 (16 games @ 120+0.5): **+5 =5 -6, score=0.469, 95% CI [0.266, 0.671].**
      Meaningfully better than v6 (0.344) and v7 (0.312) at the same level -- the cumulative Step 8 work
      is closing the 2200 gap, not just improving general play. Still technically <0.5 and n=16, but no
      longer "clearly outmatched"; per step 7 this now justifies continued investment at 2200 rather
      than writing it off.
- [x] Step 8 candidate 3 (`bb4ad10`): deepened the book to 238 entries / 14 plies (root-heavy branch:
      2 for the first 6 plies, 1 beyond that so depth grows without the node count exploding). Every
      entry independently re-verified legal. ACCEPTED as **phineas2-champion-v9-deeperbook** after
      ablation vs v8-book: +19 =2 -9, score=0.667, 95% CI [0.504, 0.829], 0 failures -- CI lower bound
      essentially at 0.5, same bar as SEE and king-safety earlier.
      Screened v9 at elo 2200 (16 games @ 120+0.5): **+9 =1 -6, score=0.594, 95% CI [0.361, 0.826].**
      First screen in the whole 2200 sequence with a point estimate clearly above 0.5. Full trend across
      the last four internal-champion versions at this exact level, same 16-game screen each time:
      v6=0.344 -> v7=0.312 -> v8=0.469 -> v9=0.594. Real, monotonic (since v7) improvement, not noise --
      each step corresponds to an accepted, ablation-confirmed candidate. Still n=16 per point, CI still
      wide, not yet a statistically airtight "beats 2200" claim.
- [x] Full 30-game 120+0.5 confirmation at elo 2200, v9-deeperbook: **+11 =7 -12, score=0.483, 95% CI
      [0.327, 0.640], 0 failures.** Regresses toward the mean from the 16-game screen's 0.594, same
      pattern as the elo-2000 confirmation earlier (0.875 screen -> 0.500 confirmed). This is the number
      to trust: **Phineas 2 (v9) plays essentially dead-even with Stockfish-2200-equivalent at the real
      competition clock** -- not a proven "beats 2200" claim (CI still spans meaningfully below 0.5),
      but no longer a real gap either, and a large, genuine improvement from the pre-P4 baseline (elo
      1800 was 0.464 back at Step 3).
      **Full honest ladder, each figure a real n=30 (or n=24 for the pre-book 2200 figure) confirmation,
      not a small screen:** elo 1800 = 0.567, elo 2000 = 0.500, elo 2200 = 0.483. This reads less like "a
      wall at 2200" and more like Phineas 2's real strength sitting close to 2000-2100 across this whole
      band, with the ladder resolution (roughly 200-Elo steps) too coarse to distinguish 2000 from 2200
      precisely -- consistent with the top of the live leaderboard sitting "just past 2100" after day one.
- [ ] NEXT: per the pre-registered decision criterion above, this result (regression toward the mean
      rather than holding the 16-game trend) means the retrained compact evaluator is the next lever
      worth pulling if further gains are wanted -- it is the only remaining Step 8 item and the one most
      likely to move the whole band up rather than patch one more specific failure mode. It is also
      materially bigger and riskier than anything done so far (a full data-generation + training
      pipeline, not a self-contained module), so treating it as a separate, explicitly-scoped effort
      rather than folding it into this session's momentum. In the meantime: phineas2-champion-
      v9-deeperbook is a fully validated, reliable, substantially-stronger-than-the-champion candidate,
      ready for a submission decision at its current strength (~2000-2100-equivalent) if the user wants
      to stop here rather than continue chasing 2200+.
