# Autonomous internal optimisation

The persistent controller proposes one isolated candidate at a time, evaluates
the exact submission ZIP on GitHub, runs a paired constrained-Linux champion
arena, and records an immutable decision before starting the next experiment.

Candidate-editable paths are deliberately narrow:

- `agent.py`
- `weights/`

The controller rejects changes to the official `harness/`, acceptance policy,
workflows, controller, and policy tests before pushing a candidate.

## State and evidence

- `.autoloop/state.json`: internal champion and experiment cursor.
- `.autoloop/protected/policy.json`: fixed gates and decision boundaries.
- `experiments/`: accepted, rejected, inconclusive, failed, and infrastructure
  records.
- GitHub Actions artifacts: exact-package compliance and reliability results.
- Candidate branches: retained experiment source.

The first protected gate builds and extracts the exact ZIP inside a pinned
Python 3.12 Linux image. The container enforces a one-core quota, 2 GB memory,
128 processes, no network, a read-only workspace, and a 256 MB `/tmp`. It
records the image ID and cgroup telemetry, then checks the API, contents, size,
dependency syntax, source indicators, initialization, legal moves across edge
cases, low-clock responses, lint/type/policy checks, two-colour smoke games,
and learned-model move-selection ablation.

`make release-zip` builds the same byte-reproducible artifact used by protected
evaluation. It fixes ZIP timestamps and permissions so identical submission
source produces an identical SHA-256 across checkouts and hosts.

The GitHub arena builds and extracts both exact submission ZIPs, uses 32 frozen
openings with paired colours, and retains each PGN in the experiment record.
It evaluates at declared 16-game looks from 32 through 64 games. A one-sided
95% Wilson boundary above or below an even score accepts or rejects; overlapping
evidence remains inconclusive. Both agents play under the same Linux limits on
the same shared Actions runner. This is useful match evidence, but the host is
not a dedicated fixed physical machine, so tiny nodes-per-second differences
remain out of scope.

`python3 controller.py --release-check` dispatches a separate protected gate.
It adds a 16-game learned-model strength ablation and two full 120 s + 0.5 s
two-colour games, persists the result and Linux image/resource evidence, and
sets `submission_candidate` only when every absolute gate passes. This creates
an internal nomination; it does not upload anything.

## Run

From a clean `main` checkout with GitHub and Codex authenticated:

```sh
python3 controller.py --iterations 2
```

For a persistent no-click loop on the Mac, keep the host awake and run:

```sh
caffeinate -dimsu python3 controller.py --continuous
```

Create `.autoloop/controller.stop` to request a clean stop after the current
experiment. An infrastructure or authentication failure stops continuous mode
instead of immediately retrying and consuming resources.

To evaluate the current champion for internal submission-candidate status:

```sh
python3 controller.py --release-check
```

No code path in the controller uploads to the competition. The protected policy
sets `competition_upload_enabled` to `false`, and there are no submission
credentials in the repository or workflow.
