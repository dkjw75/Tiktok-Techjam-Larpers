# Competition-valid research agent: implementation progress

Last updated: 2026-08-30 (Asia/Singapore)

## Objective and source precedence

Implement the audited autonomous recommender-system research agent on top of
`agentv1.0`, while treating the uploaded organizer starter kit as the benchmark
authority. The governing order is:

1. Organizer `evaluate.py`
2. Organizer starter-kit README and benchmark contract
3. Repository `AGENTS.md`
4. `agentv1.0` implementation and tests
5. `docs/agent-architecture.md`
6. `PHASES.md` and other plans

The competition test split remains outside research and model-selection flows.

## Branch and publication state

- Local implementation branch: `codex/competition-valid-agent`
- Base implementation branch: `agentv1.0`
- Remote repository: `dkjw75/Tiktok-Techjam-Larpers`
- GitHub publication is blocked: the available GitHub credential belongs to
  `smellywesley`, which does not have write permission to the repository, and
  `gh` has no authenticated account. No push or PR has been performed.
- The working tree is intentionally uncommitted while the audit gates are open.

## Organizer-contract verification completed

- Compared `evaluate.py`, `baseline.py`, `data.py`, and `submit.py` against the
  uploaded starter-kit ZIP after line-ending normalization.
- All four match the organizer copies. Normalized `evaluate.py` SHA-256:
  `ecfde28392eb14fec4f488083251df50624e1af2b86278b962daecfb42d195de`.
- The official KuaiRand-Pure archive was downloaded and extracted into the
  ignored `KuaiRand-Pure/data` directory.
- Observed canonical split sizes:
  - train: 1,141,112 rows
  - validation: 124,909 rows
- No official test score was completed. A later defective finalization attempt
  accessed test labels and was killed before any output or metric persisted;
  that transaction is abandoned and disclosed in `AUDIT_REPORT.md`.

## Major implementation work completed

### Benchmark fidelity and comparison validity

- Added typed comparison-validity and fidelity contracts.
- Enforced organizer metrics, validation-only selection, strict improvement
  threshold semantics, and matched comparison lineage.
- Classified a run that reaches the 40-epoch ceiling as truncated and therefore
  non-comparable, rather than treating it as a valid early-stopped result.
- Restored the best validation checkpoint instead of using the last epoch.
- Added the organizer FM as a validation-only matched comparator.
- Prevented low-fidelity screening runs from becoming champions.

### Autonomous search lifecycle

- Added direction-local pairwise search, screening batches, full-fidelity
  promotion, proposal criticism, evidence review, and region management.
- Separated cycles, logical trials, runner attempts, valid comparisons, and
  planning rejections in state accounting.
- Added atomic cycle-capacity and wall-clock checks, terminal plateau behavior,
  duplicate prevention, and append-only recovery evidence.
- Added three-seed matched confirmation before a production candidate may
  replace the champion.

### Execution isolation and data boundary

- Moved production candidates to isolated subprocess workers with process-tree
  timeout handling.
- Added durable attempt reservation, retry, collision, interruption recovery,
  and completed-attempt reuse.
- Staged only train and validation rows for workers; worker job documents do not
  contain the raw dataset path.
- Replaced unsafe pickle staging with gzip-compressed JSON.
- Bound staged inputs to dataset and staging-code hashes and validated schema,
  date boundaries, and binary labels before worker use.
- Added dataset, evaluator, preprocessing, staging, feature-schema, model-code,
  seed, and comparison-group lineage metadata.

### LLM planning and operational fallback

- Added Responses-API parsing, refusal handling, retries, structured-output
  validation, and explicit deterministic fallback.
- With no `OPENAI_API_KEY`, the CLI records degraded deterministic planning
  rather than silently exiting or pretending an LLM was used.

### Finalization and reporting foundations

- Added explicit one-time final-evaluation authorization.
- Added a finalization lock, state machine, fingerprint, test-access boundary,
  checkpoint validation, submission generation, and fail-closed retry behavior.
- Added final seed confirmation and finalization commands.
- Updated the root and research-agent documentation; marked `PHASES.md` as
  historical where it conflicts with implemented behavior.

## Real-data evidence obtained

- Organizer FM subprocess smoke, one epoch:
  - runtime: 22.422 seconds
  - validation predictions: 124,909
  - separate worker PID confirmed
  - comparison lineage recorded
- One autonomous validation-only cycle completed in `runs_agent_real`:
  - three screen trials and one full promotion
  - screen primaries approximately 0.596992, 0.598755, and 0.598824
  - promoted full run early-stopped after 7 epochs
  - full validation primary: 0.598823632
  - delta versus published baseline 0.6016: -0.002776
  - candidate correctly rejected; organizer baseline remains champion
  - state after cycle: 1 cycle, 4 logical iterations, 1 valid comparison,
    1 consecutive non-improvement
- Seed confirmation was correctly not triggered for the rejected candidate.

## Verification completed so far

- Earlier full suite: 91/91 tests passed before the latest hardening changes.
- Latest focused gate after seed/cache hardening: 19/19 tests passed.
- Earlier static gates:
  - Ruff passed for `research_agent` and `tests`
  - mypy passed with project-compatible import settings
  - `git diff --check` passed

These full gates must be rerun after all remaining changes; the earlier results
are not being represented as final evidence.

## Current hardening completed in this resume

- Fixed a misplaced return that caused first-time seed confirmation resolution
  to mutate state but return `None`.
- Seed certificates now require persisted runner evidence, fixed candidate and
  comparator identities, fixed seeds, finite scores, valid matched lineage,
  recomputed mean/win/decision values, exact attempt accounting, and an
  in-artifact checkpoint for confirmed candidates.
- Re-resolving the same certificate is idempotent and does not double-count
  comparisons or attempts.
- Added tests for replay idempotence, tampering/non-finite rejection, test-date
  stage poisoning, and corrupted-stage regeneration.

## Remaining work, in required order

1. Resume any persisted `pending_confirmation` before proposing a new cycle.
2. Increase full-promotion plus matched-seed wall-clock reserve from 3,600 to
   the actual 4,200-second worst-case envelope.
3. Remove speculative screen ancestry by proposing/executing screen candidates
   sequentially and allowing only evaluated records to become parents.
4. Implement periodic screen-to-full calibration required by comparison-validity
   policy, persist its evidence, and react when screen ranking is unreliable.
5. Bind run manifests and finalization to actual dataset/evaluator/preprocessing/
   schema hashes; reject resume/finalize when immutable inputs drift.
6. Hash final submission/report outputs and verify those hashes before completed
   finalization reuse; finish stale-lock recovery.
7. Decide and test safe finalization semantics for terminal search exhaustion.
8. Overlay seed resolutions and comparison-group evidence in reports.
9. Fix the remaining planner type annotation and run Ruff, mypy, the full unit
   suite, organizer parity, `git diff --check`, and a fresh production smoke.
10. Run another independent architecture/validity/code review wave, fix all
    high-severity findings, then commit locally.
11. Resume additional validation-only research cycles only after the integrity
    gates are green. Do not cross the test boundary without explicit final
    confirmation.
12. Push the branch and open a PR once a GitHub account with write permission is
    authenticated.

## Brutal current assessment

The implementation is materially safer and more competition-aligned than the
starting branch, but it is not ready for a five-hour unattended run yet. The
missing screen/full calibration and immutable-resume/finalization binding are
validity blockers, not documentation polish. Launching the long run before they
are fixed could spend the budget producing evidence that cannot be defended.
