# Competition Research Agent

This package implements the autonomous KuaiRand-Pure ranking research loop. It
keeps the organizer's `evaluate.py`, `baseline.py`, `data.py`, and `submit.py`
unchanged and treats the organizer evaluator as authoritative.

## Safety and validity guarantees

- Validation is the only research-selection split. Test is accessible only to
  the explicit finalization transaction.
- Low-fidelity trials rank screen candidates only and never update champion
  state.
- Full comparisons use 40 epochs and patience 4. Reaching the epoch ceiling is
  recorded as truncation and is not a comparable result.
- Candidate and incumbent must share dataset, evaluator, preprocessing,
  feature-schema, seed, and stopping-budget evidence. Model-code hashes are
  recorded separately because model code may legitimately differ.
- A single-seed improvement remains provisional. Promotion requires a mean
  delta strictly greater than `0.002` across seeds 0, 1, and 2, with the margin
  holding on at least two seeds.
- Organizer-baseline comparisons run the unchanged NumPy FM rule on the same
  seeds. Later champions reuse their persisted matched-seed evidence.
- Three valid non-improvements, 50 logical trials, or six hours stops research.
  Failures and infrastructure retries do not masquerade as scientific
  non-improvements.

## Execution and recovery

`runner.py` reserves append-only logical experiments and attempts. Production
candidates run in subprocesses with process-tree hard timeouts. Transient
failures receive one bounded retry under the same logical trial. Completed
attempts are reused, deterministic failures fail closed, and stale attempts are
preserved as interrupted evidence.

Before a worker launches, `data_boundary.py` stages train/validation-only rows.
The worker job never contains the raw dataset path. Candidate output must cover
the complete canonical validation split in exact row order.

State is reconciled from append-only iterations and promotion resolutions on
resume. Experiment identifiers also consider durable runner allocations, so an
orchestrator crash cannot silently reuse an orphaned ID.

## Search loop

1. The LLM planner selects one approved research direction and explains it.
2. The deterministic search controller selects exact bounded configurations.
3. The critic rejects unsafe work before any candidate execution.
4. Up to three equal-fidelity screens are compared within the cycle.
5. The best rankable screen is promoted to full fidelity.
6. A provisional full improvement receives matched-seed confirmation.
7. The controller accepts, rejects, preserves the incumbent, and updates the
   stopping state.

Pairwise directions use direction-local configuration ancestry. This permits a
bounded learning-rate/L2 envelope after a loss-family bootstrap even when that
bootstrap did not replace the global champion. The comparison incumbent remains
the global champion throughout.

## Commands

```powershell
python -m research_agent.run_research run --cycles 50 --artifact-dir runs_llm
python -m research_agent.run_research resume --cycles 10 --artifact-dir runs_llm
python -m research_agent.run_research status --artifact-dir runs_llm
python -m research_agent.run_research confirm-seeds --artifact-dir runs_llm
python -m research_agent.run_research finalize --artifact-dir runs_llm --confirm-final-evaluation
```

`--cycles` bounds the cycles attempted by that invocation; it does not replace
the competition's 50-trial safety limit. A paused run must be resumed until it
reaches a valid terminal state.

## Evidence

The artifact directory contains append-only events, iterations, metric tables,
per-attempt worker artifacts, configuration patches, state, environment
manifest, promotion certificates, final seed certificate, finalization
transaction and `research_log.md`. The report includes cycle/trial/retry counts,
wall time, LLM token totals, fallback count, environment information,
comparison validity, termination evidence, lineage, seed scores and manual
interventions.
