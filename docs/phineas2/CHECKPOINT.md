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

## KEY BLOCKER for the >2000 path
`.autoloop/protected/Dockerfile` (the promotion-CI image) installs only chess/mypy/numpy/pytest/ruff —
**NO numba/onnxruntime/torch**. So a numba- or onnx-based agent CRASHES in the protected CI even though
the real competition env has them. Decision needed from user: add numba(+onnxruntime) to that Dockerfile
(protected change) OR keep Phineas 2 pure-Python (chess+numpy only) for the promotable path.
=> Building pure-Python first (no approval needed, hits the 1800 min). numba = optional later upgrade.

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
- P1  fast board: pure-Python bitboard board, incremental Zobrist + PST + phase, make/unmake.
      GATE: perft exact match vs python-chess to depth 5-6 on a standard suite; >=150k nps target.
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
- [ ] P0 scaffolding
