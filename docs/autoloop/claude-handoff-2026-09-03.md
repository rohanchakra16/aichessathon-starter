# Claude-only continuation checkpoint — 3 September 2026

## Current authority and state

- Repository: `rohanchakra16/aichessathon-starter`
- Local root: `/Users/phantomvenom/Documents/Codex/2026-08-30/referenced-chatgpt-conversation-this-is-an/outputs/aichessathon-starter`
- Protected internal champion: `exp-0066`, commit
  `e9da1e556cd43d0045d4733e9dd313512ab128f4`
- Last completed experiment: `exp-0086`
- Next experiment: `exp-0087`
- Live competition upload: v2, the older `exp-0052` champion, checksum prefix
  `e46046749bf4`
- Competition upload remains outside the autonomous controller.

Always re-read `.autoloop/state.json`; it is authoritative if later promotions
make any identifier in this dated checkpoint stale.

## Why exp-0066 remains champion

Claude replaced `board.outcome(claim_draw=True)` at every search node with
cheaper terminal/draw checks. Against the previous champion it scored 20 wins,
10 draws and 2 losses: 78.125%, with a 68.58–85.39% confidence interval. It
passed the protected exact-artifact, legality, clock, resource and model gates.

Eleven scientifically completed candidates from `exp-0076` through `exp-0086`
failed to displace it. All eleven had zero arena reliability failures:

| Experiment | Mechanism | W-D-L | Score | 90% interval | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| 0076 | per-ply killer moves | 15-32-17 | 48.4% | 41.3–55.7% | inconclusive |
| 0077 | quiet promotions in quiescence | 16-32-16 | 50.0% | 42.8–57.2% | inconclusive |
| 0078 | capped final check-evasion ply | 14-38-12 | 51.6% | 44.3–58.7% | inconclusive |
| 0079 | guarded null-move pruning | 18-29-17 | 50.8% | 43.6–58.0% | inconclusive |
| 0080 | exception-safe timeout unwinding | 17-32-15 | 51.6% | 44.3–58.7% | inconclusive |
| 0081 | depth-aware mate scores | 11-36-17 | 45.3% | 38.2–52.6% | inconclusive |
| 0082 | skip impossible early repetition probes | 16-32-16 | 50.0% | 42.8–57.2% | inconclusive |
| 0083 | prioritize forcing checks | 9-33-22 | 39.8% | 33.0–47.1% | rejected |
| 0084 | depth-preferred TT principal-move reuse | 16-26-22 | 45.3% | 38.2–52.6% | inconclusive |
| 0085 | duplicate killer-move attempt | 17-34-13 | 53.1% | 45.9–60.2% | inconclusive |
| 0086 | aspiration-window retry | 16-35-13 | 52.3% | 45.1–59.5% | inconclusive |

`exp-0070` was invalidated by a temporary generator-governance test mismatch;
its exact killer-move candidate was fairly re-evaluated as `exp-0076`.
`exp-0085` duplicated that mechanism because the recovered record originally
lacked hypothesis metadata. That metadata is now repaired and the concise
evidence window is 48 experiments. `exp-0086` also overlaps the earlier
aspiration-window `exp-0057`. Do not treat either duplicate as a new mechanism.

## Exhausted mechanism families

Treat these as exhausted in their tested form. Changing only a cap, threshold,
depth, margin, table size, ordering weight or time constant is not a new
hypothesis:

- history, killer and TT-best-move ordering (`exp-0050`, `exp-0056`,
  `exp-0060`, `exp-0076`, `exp-0084`, `exp-0085`)
- aspiration windows (`exp-0057`, `exp-0086`)
- guarded null-move pruning (`exp-0058`, `exp-0079`)
- bounded check extensions/evasions (`exp-0055`, `exp-0062`, `exp-0078`)
- general forcing-check priority (`exp-0083`, decisively worse)
- mate-distance scoring (`exp-0048`, `exp-0065`, `exp-0081`)
- quiet promotions in quiescence (`exp-0049`, `exp-0077`)
- quiescence delta pruning (`exp-0067`)
- draw contempt (`exp-0063`)
- halfmove-clock TT bucketing and early repetition-call guarding
  (`exp-0068`, `exp-0082`)
- timeout exception unwinding (`exp-0080`)
- exact quiescence stalemate repair (`exp-0069`)
- the simple six-feature linear king-safety residual; offline metrics showed no
  meaningful signal, so do not place it in the arena unchanged

Read retained JSON records for exact evidence. The protected champion already
contains accepted exchange-aware capture ordering/pruning, PVS/LMR, partial
root-iteration salvage, and the fast draw/terminal path. Do not independently
rediscover those accepted mechanisms.

## How Claude should continue

Use an interactive local Claude Code session at the repository root as the
semantic supervisor. The deterministic controller is configured Claude-only
and limits each candidate to `agent.py` and permitted `weights/` files.

Run a bounded batch such as:

```sh
./.venv/bin/python controller.py --iterations 3
```

After each batch, inspect the new retained evidence. A batch boundary is not a
stopping condition: immediately start another batch without user interaction
when a materially different, well-motivated hypothesis remains. Do not use
`--continuous`; it knows how to stop on infrastructure failure or a stop file,
but it cannot make the semantic exhaustion judgment.

The existing eleven-result streak exceeds the numeric stopping minimum. Before
starting `exp-0087`, therefore, perform an explicit evidence audit. Continue
only with a genuinely different mechanism that is justified by the champion's
code, retained losing games, or reproducible profiling. If no such mechanism
exists in the current search/evaluator direction, declare that direction
saturated rather than manufacturing a parameter variant. A materially new
offline evaluator-training or loss-driven feature programme may be proposed as
a next direction, but do not silently expand candidate permissions or alter
governance to create it.

Keep the Mac awake, the local session open and the internet connected. Do not
enable permission bypass. GitHub Actions runs the expensive protected games.

## Stop, validate and submit

Continue from the newest promoted champion. After any promotion, reset the
scientific non-improvement streak and continue. Stop only when either:

1. a genuine infrastructure, authentication, rules or reproducibility blocker
   prevents safe progress; or
2. at least five consecutive scientifically completed candidates have failed
   since the newest promotion and no materially different, well-motivated
   experiment remains in the current evaluator/search direction.

Generator and infrastructure failures are not scientific non-improvements.
When a stopping condition is genuinely met, run:

```sh
./.venv/bin/python controller.py --release-check
```

The release check does not upload. It creates a reproducible candidate under
the real competition clock and resource envelope. Never upload without a new,
explicit instruction from the user.

The authoritative specification says rated ladder rounds use the latest valid
upload. Uploads close on 11 September at 11:00 and submissions lock at 12:00.
The subsequent 11-round qualification Swiss uses the locked submission.

## Preserve user files

`benchmark_phineas.py` and `phineas_ui/` are user-owned untracked files. Do not
delete, stage, overwrite or reformat them.
