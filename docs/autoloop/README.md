# Autonomous internal optimisation

The persistent controller proposes one isolated candidate at a time, evaluates
the exact submission ZIP on GitHub, runs a paired local champion arena, and
records an immutable decision before starting the next experiment.

Candidate-editable paths are deliberately narrow:

- `agent.py`
- `requirements.txt`
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

The first protected gate builds and extracts the exact ZIP, then checks its API,
contents, size, dependency syntax, source indicators, initialization, legal
moves across edge cases, low-clock responses, lint/type checks, and two-colour
smoke games.

The local arena is intentionally an early bootstrap rule, not the final release
rule. Before any submission candidate is nominated it must be replaced or
supplemented with a frozen opening suite, fixed-host resource enforcement,
larger paired matches, declared statistical stopping boundaries, learned-model
ablation, and real-clock release games.

## Run

From a clean `main` checkout with GitHub and Codex authenticated:

```sh
python3 controller.py --iterations 2
```

No code path in the controller uploads to the competition. The protected policy
sets `competition_upload_enabled` to `false`, and there are no submission
credentials in the repository or workflow.

