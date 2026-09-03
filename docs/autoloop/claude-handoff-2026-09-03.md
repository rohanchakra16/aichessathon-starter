# Claude-only continuation checkpoint — 3 September 2026

## Current authority and state

- Repository: `rohanchakra16/aichessathon-starter`
- Local root: `/Users/phantomvenom/Documents/Codex/2026-08-30/referenced-chatgpt-conversation-this-is-an/outputs/aichessathon-starter`
- Protected internal champion: `exp-0089`, commit
  `4a0c988009ecae163ac09368f92e4a792dac7568`
- Last completed experiment: `exp-0093`
- Next experiment: `exp-0094`
- Live competition upload: v3, the current `exp-0089` champion, checksum prefix
  `55fec84c6dab`; dashboard status `VALID` and `Active`
- Competition upload remains outside the autonomous controller.

Always re-read `.autoloop/state.json`; it is authoritative if later promotions
make any identifier in this dated checkpoint stale.

## Why exp-0089 remains champion

`exp-0066` first earned promotion by replacing the expensive
`board.outcome(claim_draw=True)` search-node probe. `exp-0089` later added a
cheap 3-centipawn-per-square net-mobility term to the learned leaf evaluator
and beat `exp-0066` 14 wins, 12 draws and 6 losses: 62.5%, with a 52.23–71.76%
confidence interval. It passed exact-artifact, legality, clock, resource and
model-ablation gates with zero failures.

Four subsequent positional terms did not clear promotion: `exp-0090`
king-danger (18-29-17, 50.8%), `exp-0091` passed pawns (21-26-17, 53.1%),
`exp-0092` rook file activity (17-29-18, 49.2%), and `exp-0093` doubled/isolated
pawns (14-34-16, 48.4%). The next audit must treat these mechanisms as tested,
recompute the streak as four, and choose a materially different hypothesis.

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

Use the external supervisor from a normal macOS Terminal. Do not ask an
interactive Claude Desktop sandbox to run `gh`; that environment's TLS proxy
certificate is not trusted by GitHub CLI. The supervisor keeps GitHub-dependent
controller work in the normal Terminal environment and invokes Claude Code
programmatically for both evidence audits and candidates.

Run:

```sh
./.venv/bin/python claude_supervisor.py --continuous
```

The first action on every process start is a fresh Claude evidence audit, so a
stale `research/next-direction.md` is never executed blindly. A batch boundary
and a promotion are not stop conditions. The supervisor continues until Claude
finds genuine scientific saturation, an infrastructure blocker occurs, the
Claude subscription becomes unavailable, or `.autoloop/supervisor.stop` is
created.

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
