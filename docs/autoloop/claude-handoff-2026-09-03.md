# Claude-only continuation checkpoint — 3 September 2026

## Current authority and state

- Repository: `rohanchakra16/aichessathon-starter`
- Local root: `/Users/phantomvenom/Documents/Codex/2026-08-30/referenced-chatgpt-conversation-this-is-an/outputs/aichessathon-starter`
- Protected internal champion: `exp-0066`, commit
  `e9da1e556cd43d0045d4733e9dd313512ab128f4`
- Last completed experiment: `exp-0069`
- Next experiment: `exp-0070`
- Live competition upload: v2, the older `exp-0052` champion, checksum prefix
  `e46046749bf4`
- Competition upload remains outside the autonomous controller.

Always re-read `.autoloop/state.json`; it is authoritative if later promotions
make any identifier in this dated checkpoint stale.

## Why exp-0066 is champion

Claude replaced `board.outcome(claim_draw=True)` at every search node with
cheaper terminal/draw checks. Against the previous champion it scored 20 wins,
10 draws and 2 losses: 78.125%, with a 68.58–85.39% confidence interval. It
passed the protected exact-artifact, legality, clock, resource and model gates.

The exact quiescence-stalemate repair in exp-0069 scored 16 wins, 34 draws and
14 losses (51.56%, interval 44.34–58.72%) and was not promoted. Retain that
negative result; do not repeat it as a parameter variation.

## Recent exhausted mechanisms

- TT best-move ordering: neutral (`exp-0060`)
- bounded check extensions: neutral (`exp-0055`, `exp-0062`)
- draw contempt: neutral (`exp-0063`)
- mate-distance scoring: neutral (`exp-0065`)
- quiescence delta pruning: neutral/slightly negative (`exp-0067`)
- halfmove-clock TT bucketing: positive hint but inconclusive (`exp-0068`)
- simple six-feature linear king-safety residual: offline metrics showed no
  meaningful signal; do not put it into the arena unchanged

Read the retained JSON records for exact evidence. A different constant, cap or
threshold is not a new hypothesis.

## How to continue without Codex credits

No ordinary Claude chat is required. The deterministic controller already
starts a fresh restricted Claude Code process for each candidate. From Terminal:

```sh
cd "/Users/phantomvenom/Documents/Codex/2026-08-30/referenced-chatgpt-conversation-this-is-an/outputs/aichessathon-starter"
claude auth status
./.venv/bin/python controller.py --iterations 6
```

Keep the Mac awake, Terminal open and the internet connected. GitHub Actions
runs the expensive protected games; Claude is used for hypothesis selection and
candidate implementation. The controller is configured Claude-only and still
limits each Claude candidate to the repository files and its budget.

If an interactive Claude session is preferred, open a local Claude Code session
at the repository root and say: `Read CLAUDE.md and continue one bounded
six-experiment batch. Do not upload anything.` The session should invoke the
controller, not edit `main` directly.

Do not use `--continuous` for the handoff. Six-experiment batches provide a
clean checkpoint and limit accidental usage. Do not enable permission bypass;
the existing restricted controller invocation is sufficient.

## Stop, validate and submit

Continue from the newest promoted champion. After a promotion, reset the
non-improvement streak. Stop when either:

1. a genuine infrastructure/rules blocker appears; or
2. at least five consecutive scientifically completed candidates fail to
   improve the champion and Claude can identify no materially new,
   well-motivated experiment in the current evaluator/search direction.

At that point run the separate release evaluation:

```sh
./.venv/bin/python controller.py --release-check
```

That does not upload. It produces a reproducible submission candidate under the
real competition clock and resource envelope. Upload only after explicit user
approval and only if it is newer and better than the live v2 submission.

The authoritative specification says rated ladder rounds use the latest valid
upload. Uploads close on 11 September at 11:00 and submissions lock at 12:00.
The subsequent 11-round qualification Swiss uses those locked submissions, so
do not expect to change the bot during qualification.

## Preserve user files

`benchmark_phineas.py` and `phineas_ui/` are user-owned untracked files. Do not
delete, stage, overwrite or reformat them.

