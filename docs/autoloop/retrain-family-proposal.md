# Proposal: learned-evaluator retraining as a new research family

Status: **APPROVED and IMPLEMENTED (2026-09-04).** The loop is NOT restarted and
`exp-0097` has NOT run — that waits on the one-time dataset generation, the
sha256 pin, and explicit user approval.

As-built (decisions 1-5 from the review applied — datasets committed to git,
deterministic no-LLM `claude-retrain` path, full ~10-feature set, reused pairwise
move-ranking margin objective, user runs dataset generation):

| Change | File | Protected |
| --- | --- | --- |
| `train_positional_evaluator.py` — 10 bounded features, game-grouped CV, report-only validation, `AGENT_FEATURE_SOURCE` single source of truth, `--check-agent-consistency` | `training/train_positional_evaluator.py` (new) | via `protected_hash` |
| One-time dataset generator (Stockfish-18 teacher, whole-game train/validation split, `MANIFEST.json`) | `training/generate_positional_teacher_dataset.py` (new) | no |
| Deterministic `claude-retrain` generator + `--retrain-entrypoint` + `splice_residual_block` + `retrain_generate` | `controller.py` | **yes** |
| `retrain` policy block (whitelist, seed, pinned-dataset slots, leakage controls, frozen-base flag) | `.autoloop/protected/policy.json` | **yes** |
| Policy + trainer + splice assertions | `tests/autoloop/test_policy.py`, `tests/autoloop/test_positional_training.py` (new) | **yes** |
| `learned-evaluator-retrain` family routing + audit-prompt update + `retrain_blocker()` | `claude_supervisor.py` | no |

Verified: 82/82 protected tests pass, `mypy` (agent.py + harness) clean, `ruff`
clean; the splice is idempotent and keeps agent.py mypy-strict + ruff clean; a
retrain run is byte-deterministic; the residual coefficients live in the zeroed
tail of `weights/model.json` so the model-ablation gate neutralises them with
everything else.

---

## Original proposal follows

The exp-0096 audit stopped with `scientific_saturation` of the *self-contained
hand-weighted* search and leaf-evaluation families and explicitly recommended
"an offline retrain of `weights/model.json` with trained positional and endgame
features (king activity, pawn structure, passed pawns, and a tempo/progress
signal)". This document is the smallest safe extension that lets the supervisor
run that family through the unchanged controller gates.

---

## 1. What training infrastructure already exists

| Piece | File | Role |
| --- | --- | --- |
| Champion evaluator | `weights/model.json` (schema 5, `model_kind: pairwise_finetuned_tapered_piece_square_evaluator`) | 770 weights = 6 pieces x 64 squares x {midgame, endgame} + 2 castling, plus `bias`. `layout` describes the block structure. `training` block records teacher (`Stockfish 18`, binary sha256), dataset sha256, script + sha256, ridge penalty, seeds, `protected_opening_list_used: false`. `cross_validation` records baseline vs candidate margin-RMSE / MRR / top-1. |
| Pairwise fine-tune trainer | `training/train_pairwise_psqt_finetune.py` | Produced the current model. `pairwise_design()` builds teacher-ranked move-pair differences; `game_folds()` holds whole games out; ridge `fit_delta()`; `rmse()`. |
| **Residual trainer (direct precedent)** | `training/train_king_safety_residual.py` | Freezes the 770 base weights, fits **6 bounded** king-safety residual features (`FEATURE_NAMES`, `LOWER_BOUNDS`/`UPPER_BOUNDS`) by **game-grouped cross-validation on the training set only**, then reports metrics on a **separate independent validation set that is never used for selection**. Emits `weights/model.json` schema 6, `model_kind: frozen_psqt_with_king_safety_residual`, weights = `[*base_770, *residual_6]`, `layout.king_safety_offset: 770`, full provenance. This is the template the new family generalises. |
| Dataset loaders | `training/train_active_residual_evaluator.py` (`load_active_dataset`, `baseline_prediction`), `training/train_active_psqt_finetune.py` (`game_folds`, `ranking_totals`) | Dataset = JSON with `kind: "champion_disagreement_active_learning_dataset"`, `protected_opening_list_used: False`, `game_grouped: True`, rows `{game_id, parent_ply, source, fen, label}`, `rows_count`, `dataset_sha256` (verified on load). `baseline_prediction` hard-asserts `len(weights) == 770`. |
| Teacher dataset generators | `training/generate_active_learning_dataset.py`, `generate_historical_dataset.py`, `generate_stockfish_hard_positions.py` | Use `chess.engine.SimpleEngine` with Stockfish (`shutil.which("stockfish")`, present at `/opt/homebrew/bin/stockfish`) as an **offline teacher** to label positions. Never shipped in the agent. |
| Other trainers (context) | `train_nnue_evaluator.py`, `train_compact_evaluator.py`, `train_selfplay_evaluator.py`, `train_active_move_ordering.py`, `train_active_psqt_finetune.py`, `train_pairwise_psqt_finetune.py`, `train_linear_evaluator.py`, `train_stockfish_evaluator.py` | A full offline-training toolbox already exists. `train_nnue_evaluator.py` even has a two-perspective sparse net with an antisymmetry test. |
| Training tests | `tests/autoloop/test_king_safety_training.py`, `test_nnue_training.py`, `test_active_learning.py`, `test_compact_training.py`, ... | Symmetry, determinism, coefficient-bound, and no-leakage-metadata assertions per trainer. |
| Ablation gate | `.autoloop/protected/evaluate.py` `create_ablated()` / `model_move_ablation()` / `model_strength_ablation()` | Zeroes **`[0.0] * len(weights)` and `bias = 0.0`** — already weight-count agnostic. Candidate must diverge from the zeroed model on >= 8 / 32 openings and score >= 0.625 over 16 games. |

**Two gaps:**

* The candidate generator is one `claude -p` run with **no Bash**, so it cannot run a trainer. Training must happen locally in the controller process (the supervisor's terminal) before the branch is pushed.
* The frozen teacher datasets referenced by sha256 in `model.json` are **not in the repo** (`DEFAULT_INCLUDES = ("agent.py", "weights")`; nothing under `training/data/`). They must be regenerated once and pinned.

---

## 2. Exact change needed to permit retraining experiments

Four changes. Three touch protected files and need your sign-off; one is data.

### 2a. `controller.py` — a third generator `claude-retrain` (protected, ~60 lines, additive)

* New CLI flag `--retrain-entrypoint PATH` on `controller.py`. When set, `one_iteration` forces `generator = "claude-retrain"` for that experiment and threads the entrypoint through. The **stall-cadence scheduler is untouched** (`generator_for_stall_count` still returns `claude-code` for every count — `test_generator_schedule_*` stays green). Retraining is strictly out-of-band, chosen by the supervisor, never by the streak counter.
* `generate_candidate()` gains `elif generator == "claude-retrain": metadata = retrain_generate(...)`.
* `retrain_generate(worktree, experiment_id, policy, records, entrypoint)` is **fully deterministic — no LLM in the candidate path**:
  1. `retrain = policy["retrain"]`; assert `retrain["enabled"]` and `entrypoint in retrain["allowed_entrypoints"]`.
  2. Assert `git diff --quiet <champion> -- training/` in the worktree (no training-code change smuggled into the candidate). `protected_hash` also records `training/*.py` per experiment.
  3. For each `retrain["datasets"]` entry: assert the file exists and its sha256 equals the pin, else `InfrastructureError` (a missing dataset is not a scientific non-improvement).
  4. **Splice the canonical feature block into `agent.py`** between `# === BEGIN learned residual features ===` / `# === END learned residual features ===` markers, taking the exact pure-Python source from the trainer module's `AGENT_FEATURE_SOURCE` constant. Markers are inserted once (first retrain) at a fixed point in `_model_evaluate`; later retrains replace the block in place. Deterministic text operation, no model call.
  5. Run `uv run python <entrypoint> --base-model weights/model.json --output weights/model.json --training-dataset <pin> --validation-dataset <pin> --seed <retrain["seed"]>` with `timeout = retrain["timeout_seconds"]`, `cwd = worktree`. The trainer `import agent`, builds its design matrix from **`agent.positional_features(board)`** (the block just spliced), fits the bounded residual, and rewrites `weights/model.json`.
  6. `status_paths(worktree)` must be a subset of `{agent.py, weights/model.json}` — both already in `candidate_allowed_paths`, so `path_allowed` passes with no policy change to that list. Commit `experiment <id>: retrained evaluator`.
  7. Return metadata: `generator: "claude-retrain"`, `retrain_entrypoint` + sha256, `training_dataset_sha256`, `validation_dataset_sha256`, `seed`, `selected_cross_validation`, `independent_validation`, `feature_names`, `teacher_name`, `external_engine_used: true`, `feature_consistency_ok: true`.
* Everything downstream — `changed_paths` allow-check, `github_evaluate`, CI (`candidate-evaluate.yml`), `decide`, `persist`, promotion, clock-sensitive promotion — is **byte-for-byte unchanged**.

### 2b. `.autoloop/protected/policy.json` — a `retrain` block (protected)

```json
"retrain": {
  "enabled": true,
  "allowed_entrypoints": ["training/train_positional_evaluator.py"],
  "timeout_seconds": 2400,
  "seed": 20260904,
  "datasets": {
    "training":   {"path": "training/data/positional_teacher_train.json",     "sha256": "<pinned once generated>"},
    "validation": {"path": "training/data/positional_teacher_validation.json", "sha256": "<pinned once generated>"}
  },
  "leakage_controls": {
    "require_protected_opening_list_unused": true,
    "require_game_grouped": true,
    "require_disjoint_train_validation_games": true,
    "validation_is_report_only": true
  },
  "residual": { "max_features": 16, "coefficient_abs_cap": 60.0, "base_weights_frozen": true }
}
```

New assertions in `tests/autoloop/test_policy.py` (protected test file): `retrain.enabled` is a bool; every `allowed_entrypoints` path starts with `training/` and exists; both datasets carry a 64-hex sha; `residual.base_weights_frozen` is `true`; and `path_allowed("training/train_positional_evaluator.py", policy) is False` (candidates still cannot edit trainers — only the controller's own `retrain_generate` invokes them).

### 2c. `training/train_positional_evaluator.py` — the trainer (protected via `protected_hash`, NEW)

Generalises `train_king_safety_residual.py`. **Base 770 PST weights frozen; only the residual block is fit.** Feature blocks (all side-relative, hard-bounded, phase-tapered where sensible):

| Feature block | Signal | Gate |
| --- | --- | --- |
| King centralisation | own king Chebyshev distance to the four centre squares | endgame-tapered |
| Enemy-king tropism | `-(dist(own_king, enemy_king)) * sign(material_diff)` | `|material_diff| >= 300 cp` **and** `phase <= 10` (decided endgames only) |
| Passed pawns | own vs enemy count + rank-weighted advancement | none |
| Pawn structure | doubled, isolated, connected/phalanx counts (own vs enemy) | none |
| Tempo / progress | `-min(halfmove_clock, 80) * sign(material_diff)` | `|material_diff| >= 300 cp` **and** `phase <= 10` |

* Reuses `load_active_dataset`, `pairwise_design`, `game_folds`, `baseline_prediction`, `file_sha256`, `dataset_digest`.
* Ridge penalty chosen by 4-fold **game-grouped** CV **on the training set only**; independent validation set is **report-only**.
* Deterministic: `np.linalg.solve` closed form + fixed seed for any fold shuffle.
* Coefficients hard-clipped to `+/- coefficient_abs_cap`; block sized `<= max_features`.
* Exposes `positional_features(board) -> np.ndarray`, `FEATURE_NAMES`, and `AGENT_FEATURE_SOURCE` (the pure-Python runtime version). Includes `--check-agent-consistency`: recompute a sample of vectors via `agent.positional_features` and **abort with non-zero exit on any mismatch** (feature-drift guard).
* Emits `weights/model.json` schema 7, `model_kind: frozen_psqt_with_positional_endgame_residual`, `weights = [*base_770, *residual_k]`, `layout.positional_offset: 770`, `layout.positional_feature_names`, `residual_coefficients`, and a full `training` + `cross_validation` + `independent_validation` + `feature_consistency` block.

New `tests/autoloop/test_positional_training.py`: startpos features are symmetric (zero vector); coefficient bounds obeyed; CV is reproducible; tropism/tempo features are exactly zero outside their gate; the trainer refuses a dataset whose metadata is not `protected_opening_list_used: False` / `game_grouped: True`.

### 2d. `training/data/` + `training/data/MANIFEST.json` — the frozen datasets (NOT protected, NOT shipped in the zip)

Generated once on your machine, committed for reproducibility (`DEFAULT_INCLUDES` excludes `training/`, so this adds nothing to the 50 MB submission):

* `generate_active_learning_dataset.py` (and/or `generate_historical_dataset.py`) with Stockfish 18 as teacher, over a fixed source game corpus, fixed node budget, fixed seed, `protected_opening_list_used=False`.
* Split by whole `game_id` into **disjoint** `positional_teacher_train.json` / `positional_teacher_validation.json`.
* `MANIFEST.json` records: generator script + sha256, teacher name + binary sha256, node budget, RNG seed, source corpus + sha256, row/game counts, and the two dataset sha256s (which then get pinned into `policy.json` 2b).

### 2e. `claude_supervisor.py` — route the family (NOT protected, ~40 lines)

* Recognise `next_direction.family == "learned-evaluator-retrain"`.
* On CONTINUE with that family, run `controller.py --retrain-entrypoint <policy.retrain.allowed_entrypoints[0]> --iterations 1` for that batch instead of the plain command. `research/next-direction.md` is still written (informational).
* Audit prompt gains one paragraph: retraining is now an available family; select it now (the self-contained leaf-eval family is exhausted); declare `scientific_saturation` only once retraining with materially different feature families / teacher targets is *also* exhausted; refuse the family if `policy.retrain.datasets` files are absent (report an infrastructure blocker instead).
* `supervisor-log.jsonl` and the persisted audit record `family`, and for retrain batches the entrypoint + dataset shas + resulting CV metrics.

---

## 3. Files that become editable / generated

| Path | Kind | By whom | Shipped in zip? |
| --- | --- | --- | --- |
| `agent.py` | edited (feature block spliced between markers) | `retrain_generate` (deterministic splice) | yes |
| `weights/model.json` | regenerated (schema 7, `[*770, *residual]`, provenance) | the trainer | yes |
| `training/train_positional_evaluator.py` | **new**, part of `protected_hash` | you (committed to main once, reviewed) | no |
| `.autoloop/protected/policy.json` | `retrain` block added | you (reviewed) | no |
| `controller.py` | `retrain_generate` + `--retrain-entrypoint` | you (reviewed) | no |
| `tests/autoloop/test_policy.py`, `test_positional_training.py` | assertions added / new | you (reviewed) | no |
| `training/data/*.json`, `training/data/MANIFEST.json` | **new**, generated once | you, locally | no |
| `claude_supervisor.py` | family routing | me (not protected) | no |
| `research/audits/*`, `research/next-direction.md` | audit trail | supervisor | no |

The candidate still only ever commits `agent.py` + `weights/model.json`. `candidate_allowed_paths` is **unchanged**.

---

## 4. How a trained candidate flows through the protected gates

It is a normal candidate branch — no new evaluation path:

1. `retrain_generate` (local, in `.autoloop/worktrees/exp-00NN`): splice `agent.py` -> run trainer -> `weights/model.json` -> commit `{agent.py, weights/model.json}`.
2. `changed_paths(champion, sha)` -> both `path_allowed` -> proceed.
3. `github_evaluate` pushes `autoloop/candidate-00NN` -> `candidate-evaluate.yml` -> constrained Docker (`--network=none --read-only --cpus=1 --memory=2g --pids-limit=128`):
   * `evaluate.py`: ruff, mypy, smoke, resources, package/size, **model-move ablation** (zero all weights + bias; candidate must still diverge on >= 8 / 32 openings — safe, the 770 PST weights are untouched and dominate), **model-strength ablation** (candidate >= 62.5% vs the zeroed model over 16 games), **reliability stress** (init < 50 s; per-move <= 0.75 s, <= 2.25 s at the 120 s clock; `stress_time_left_ms` grid; `required_failures: 0`).
   * `arena.py`: 64-game sequential paired arena vs `HEAD^`, Wilson CI, sequential `statistical_decision`.
4. `decide(ci, match, policy)`: `accepted` only on `statistical_decision == "accept"`; otherwise `inconclusive` / `rejected`. **Unchanged.**
5. `persist`: experiment JSON retained pass or fail, `protected_hash` recorded, promotion (ff-merge to `main`, `submission_candidate` cleared) only on `accepted`. **Unchanged.**
6. Clock-sensitive promotion (prospective real-clock arena) path **unchanged**.

The retrained model earns promotion only by winning the same 64-game arena at the same significance after clearing the same ablation and reliability gates. No gate is relaxed.

---

## 5. Data-leakage / validation-contamination prevention

1. **Protected openings never in training data.** `load_active_dataset` already asserts `protected_opening_list_used: False`. The new trainer additionally refuses if any training FEN's transposition key matches a position reachable within the opening length from `.autoloop/protected/openings.json` **or** `confirmation-openings.json`.
2. **Train / validation are whole-game disjoint.** Datasets split by `game_id`; trainer asserts `set(train ids) & set(validation ids) == set()` (also pinned via `leakage_controls.require_disjoint_train_validation_games`).
3. **Model selection never sees validation.** Ridge penalty chosen by game-grouped CV *inside the training set*; the independent set yields report-only metrics — exactly `train_king_safety_residual.independent_validation`'s discipline.
4. **CV folds hold whole games out together** (`game_folds`), so no sibling position leaks a validation game into training.
5. **Dataset pinning.** `policy.json` pins both sha256; `retrain_generate` refuses to run on a mismatch; `dataset_digest` re-checked on load.
6. **Feature-drift guard.** `--check-agent-consistency` aborts the trainer if `agent.positional_features` disagrees with the trainer's own computation, so coefficients are always fit against the exact runtime semantics.
7. **Determinism.** Closed-form ridge + fixed `policy.retrain.seed`; recorded in `model.json`.
8. **Full provenance in `model.json`.** Teacher name + binary sha256, generator script + sha256, both dataset sha256s, seed, penalty grid + choice, CV + independent metrics, base-model sha256, `external_engine_used: true`.
9. **Arena / ablation openings are the frozen protected set** — disjoint from training by (1), so arena results are genuinely out-of-distribution.
10. **Failed retrains are retained** like any experiment (`persist` always writes; `protected_hash` records the trainer + policy state).

---

## 6. First retraining experiment

**exp-0097 — family `learned-evaluator-retrain`.** Fit a frozen-PST positional + endgame residual (~10 bounded features: own/enemy passed-pawn count & advancement; doubled / isolated / connected pawns; own-king centralisation, endgame-tapered; enemy-king tropism `x sign(material_diff)` gated to `|material_diff| >= 300 cp and phase <= 10`; and a tempo/progress term on the same gate) on the frozen Stockfish-18 teacher move-pair dataset with 4-fold game-grouped CV for the ridge penalty, confirmed on the disjoint independent validation set.

*Hypothesis.* The champion's flat piece-square leaf evaluation cannot express progress once a side is materially ahead, producing 25 / 64 threefold-repetition draws from winning positions and 17 / 17 losses as king-hunt checkmates after passive simplification (exp-0090 through exp-0096). A jointly-trained positional / endgame residual gives the leaf a *learned* progress and king-driving gradient that deeper search cannot recover, targeting conversion directly — and, unlike exp-0094 through exp-0096, the coefficients come from teacher data rather than hand-tuning.

*Must clear.* Model-move ablation >= 8 / 32 (expect ~28 / 32, unchanged), strength ablation >= 62.5%, reliability stress `required_failures: 0` with per-leaf cost within ~1.05x (features are cheap bitboard ops folded into the existing piece loop), then `statistical_decision == "accept"` on the 64-game arena vs exp-0089.

*If inconclusive.* The next audit may try one more materially different retrain — a different feature family (king-zone attackers, mobility-by-piece), or a different teacher target (WDL-blended eval instead of move-ranking margins) — but **not** a bare re-tune of the same features / penalty (anti-tuning rule). Project-level `scientific_saturation` only once retraining with materially different feature families *and* targets is also exhausted.

---

## Open decisions for you

1. **Datasets in git?** Recommend yes — commit `training/data/*.json` + `MANIFEST.json` (not shipped in the zip; maximally reproducible). Alternative: keep local, pin sha + manifest only.
2. **LLM in the retrain path?** This proposal uses a **deterministic splice** (no `claude -p` in `retrain_generate`) for bit-for-bit reproducibility. Confirm you prefer that over an `agent.py`-only `claude -p` wiring step.
3. **Feature set for exp-0097** — the ~10 above, or a narrower first cut (e.g. passed pawns + king centralisation + tropism only)?
4. **Teacher target** — reuse the existing move-ranking pairwise margin objective (consistent with the current model), or introduce a WDL/eval-blended target in a later experiment?
5. Who runs the one-time dataset generation (needs Stockfish 18 with the pinned binary sha256)?
