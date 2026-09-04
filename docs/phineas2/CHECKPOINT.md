# Phineas 2 — major architecture rebuild

Branch: `phineas2` (worktree). Never touch `main` / the champion while this runs.
Started 2026-09-04. Goal: reliable engine at ~1800 (min) / 2000 (competitive) / 2200+ (stretch)
under 120+0.5, one core, that clearly beats the current protected champion (exp-0089, `4a0c988`).

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
      (2) repetition preference rejected (docs/phineas2/rejected-experiments.md), (3) king safety +
      development accepted, (4) passed pawns accepted, (5) mobility accepted. Current internal champion:
      **phineas2-champion-v5-mobility**, HEAD of the `phineas2` branch.
- [ ] NEXT: Step 6, a larger 24-40 game exact-clock (120+0.5) confirmation vs Stockfish elo 1800,
      compared against the Step 3 baseline number above (phineas2-baseline-p1p3: score=0.464). Then
      screen 2000 only if competitive at 1800, 2200 only if competitive at 2000 -- do not spend a large
      batch at a level a small screen already shows is clearly outmatched. Then Step 8 (opening book,
      tablebase, retrained compact evaluator) if the ladder still has room to climb.
