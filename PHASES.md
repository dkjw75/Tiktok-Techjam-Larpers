# Historical Implementation Phases

This file records the original phased plan. The implementation has now advanced
beyond it; the authoritative operating instructions are `README.md`,
`research_agent/README.md`, the organizer evaluator, and persisted run evidence.
In particular, champion promotion now requires matched three-seed confirmation,
and the competition stopping limit is 50 logical trials rather than the early
single-run controller described below.

Yes. We’ll implement it incrementally, with each phase producing something testable before moving on.

The eventual operating loop will be:

```text
Research agent chooses a hypothesis
              ↓
Safety checks approve or reject it
              ↓
Controller creates an isolated experiment
              ↓
Runner trains the candidate
              ↓
Metrics layer calls evaluate.py
              ↓
Controller accepts or rejects the candidate
              ↓
Logger records every material action
              ↓
Experiment store preserves the evidence
              ↓
Markdown report presents the run
```

## Phase 0: Prepare the environment

Before building the agent:

1. Identify or create the project’s Python environment.
2. Ensure NumPy is available.
3. Confirm the official FM baseline still reproduces.
4. Record the Python and NumPy versions.
5. Confirm the dataset remains outside Git.
6. Add generated checkpoints and large run artifacts to `.gitignore`.

This prevents us from blaming the new agent when the underlying environment is broken.

## Phase 1: Create the project structure

The initial structure will be approximately:

```text
research_agent/
├── controller.py
├── runner.py
├── metrics.py
├── safety.py
├── logger.py
├── store.py
├── reporter.py
├── state.py
├── templates/
│   ├── learning_rate.json
│   ├── regularization.json
│   └── pairwise_loss.json
└── models/
    └── pairwise_fm.py

tests/
├── test_decisions.py
├── test_logger.py
├── test_metrics.py
├── test_recovery.py
├── test_safety.py
└── test_runner.py

research_runs/
├── experiments.jsonl
├── metrics.csv
├── research_log.md
├── manual_interventions.md
├── patches/
└── runs/
```

`baseline.py` and `evaluate.py` will remain unchanged.

## Phase 2: Build the experiment store and logger

We build logging before running experiments so that no research event happens without evidence.

### Event logger

Every material action gets an append-only JSONL event:

```json
{
  "timestamp": "...",
  "experiment_id": "exp_001",
  "action": "training_started",
  "details": {}
}
```

Examples of logged actions:

- Hypothesis selected
- Safety check passed or failed
- Candidate created
- Training started
- Epoch completed
- Evaluation started
- Metrics received
- Candidate accepted or rejected
- Error detected
- Retry attempted
- Previous candidate restored
- Human intervention received

### Experiment record

At the end of each iteration, a complete record is written containing:

- Hypothesis and rationale
- Parent experiment
- Controlled change
- Complete configuration
- Code or configuration diff
- Seed
- Runtime and resource budget
- GAUC
- nDCG@5
- Primary score
- Difference from the parent
- Decision and reason
- Errors and recovery
- Manual interventions

### Markdown reporter

The reporter will read the JSONL records and generate `research_log.md`. Markdown will not be written independently because that could cause it to disagree with the structured record.

## Phase 3: Build the metrics layer

The metrics layer will be deliberately small.

It will:

1. Receive user IDs, labels, and candidate scores.
2. Reject missing, NaN, or infinite scores.
3. Confirm all arrays have equal length.
4. Call the unchanged `evaluate.py`.
5. Return GAUC, nDCG@5, and primary.
6. Record which evaluator version was used.

It will never redefine or recalculate the benchmark metrics itself.

Routine experiments will evaluate validation only. Test evaluation will require an explicit final-validation command.

## Phase 4: Build the experiment runner

The runner is not an LLM agent. It is a controlled program.

For every experiment it will:

1. Receive an approved configuration.
2. Create a unique run directory.
3. Load training and validation data.
4. Train the requested candidate.
5. Save per-epoch results.
6. Save the candidate checkpoint.
7. Return validation predictions to the metrics layer.
8. Capture errors and exit status.

Each experiment will have a separate directory:

```text
research_runs/runs/exp_001/
├── plan.json
├── config.json
├── events.jsonl
├── epoch_metrics.csv
├── final_metrics.json
├── decision.json
├── stdout.log
├── error.json
└── checkpoint.npz
```

The runner will execute as a separate process so the controller can enforce a timeout and survive a runner crash.

## Phase 5: Build deterministic safety checks

Before training, the safety layer will check:

- The template is approved.
- Only one conceptual factor changed.
- No external dataset was added.
- Training uses only the training split.
- Feature creation does not use future labels.
- Test labels are not used.
- `evaluate.py` was not modified.
- Scores and configurations are valid.
- The runtime budget is acceptable.
- The proposal is not an exact duplicate of a previous experiment.
- A new dependency or model family has human approval.

A failed safety check is logged and the experiment is not launched.

## Phase 6: Build the controller

The controller coordinates the other components.

Its state will contain:

```text
current accepted experiment
current best validation score
consecutive non-improving experiments
remaining experiment budget
next approved template
active experiment, if any
```

For each iteration it will:

1. Read the contract, architecture, current best, and history.
2. Choose one template.
3. Form one hypothesis.
4. Run safety checks.
5. Ask the runner to execute it.
6. Receive metrics.
7. Apply the decision rule.
8. Update or retain the current best.
9. Write the complete record.
10. Check the stopping rule.

### Initial decision rule

- Improvement greater than `0.002`: accept.
- Improvement between `0` and `0.002`: inconclusive; do not promote.
- No improvement: reject.
- Contract failure, crash, or invalid metrics: fail.
- A failed or rejected candidate never replaces the accepted candidate.

## Phase 7: Implement the Research/Orchestrator Agent

Implement the agent that decides **what is worth investigating**, using the architecture in `docs/agent-architecture.md` as its operating model.

The agent must read:

- the fixed benchmark contract
- the persisted controller state
- append-only experiment history and metric summaries
- previous errors, recoveries, and manual interventions
- the architecture guidance before proposing work

For each research cycle, it produces a structured **research direction**, not a hard-coded experiment:

```text
hypothesis
rationale
approved research direction
search-space definition
success/failure evidence to look for
suggested evaluation budget
```

The agent must not select a stream of precise numeric values or directly run code. It proposes the scientific question and explains why the history supports it.

## Phase 8: Implement the Search Controller

Implement the layer between a research direction and individual candidate trials.

The Search Controller must:

1. Read the agent's structured research direction.
2. Select an exploration or exploitation strategy.
3. Define or update a search region.
4. Allocate a compute budget and evaluation fidelity.
5. Select exact candidate configurations with deterministic search logic.
6. Send only safety-checkable proposals to the existing controller.
7. Preserve parent lineage, search region, strategy, and budget in the experiment store.

The LLM/orchestrator chooses **what** to investigate. The Search Controller and its search algorithm choose **which exact configurations** to evaluate.

Start with dependency-free deterministic sampling. TPE, Bayesian optimization, ASHA, Hyperband, or other search libraries require explicit human review before they are added.

## Phase 9: Implement Multi-Fidelity Execution and Recovery

Upgrade execution from a single full run to evidence-based resource allocation.

The runner and Search Controller should support:

- cheap initial evaluations with limited epochs or another approved low-cost budget
- promotion of promising candidates to higher-fidelity runs
- explicit comparison of early and final performance before aggressive pruning
- hard time-limit handling and preserved failure artifacts
- bounded retries with the recovery action and reason recorded
- restoration of the accepted candidate pointer after every rejected or failed branch

A candidate must never be promoted solely because it was cheap to run. Promotion must be based on recorded validation evidence.

## Phase 10: Implement Agent Review and Critic Checks

After a batch of candidate trials, the Research/Orchestrator Agent reviews the evidence and decides whether to:

- continue the current research direction
- refine a promising search region
- increase exploration
- request a new research direction
- mark the current region as plateaued
- request a human review when the next step needs new authority

Add a deterministic critic around each agent proposal. The critic must verify benchmark compliance, data-flow restrictions, duplicate avoidance, budget limits, and required logging before the Search Controller receives the proposal.

The agent and critic must have narrow interfaces. They must not receive permission to modify `evaluate.py`, bypass `data.py`, use test labels for optimization, or silently add dependencies.

## Phase 11: Implement Plateau Handling and Diverse Search Regions

Add explicit search-state management from the architecture:

```text
BOOTSTRAP → EXPLORING → PROMISING → EXPLOITING
                         ↓
                      PLATEAU
                         ↓
                 RESTARTING / EXPLORING
```

The controller must maintain more than one plausible search region when the budget permits and reserve part of the budget for exploration.

When progress plateaus, the agent should reassess the hypothesis itself rather than repeatedly making near-identical changes. A restart is a recorded research decision, not an unlogged failure.

## Phase 12: Validate Finalists and Produce Submission Evidence

Only after the research loop identifies promising candidates:

1. Re-run finalists with independent seeds where the budget permits.
2. Compare validation results and reliability against the accepted baseline.
3. Select one winner using validation evidence.
4. Generate the final prediction file.
5. Run `submit.py --check`.
6. Use test evaluation only as final confirmation, never as the routine optimization objective.
7. Generate the final Markdown research report, JSONL/CSV evidence, code patches, resource/token summary, and manual-intervention summary.

## Ongoing rules from Phase 7 onward

- No fixed sequence of hypotheses or experiment templates is embedded in the agent.
- Every proposed direction is evidence-based and logged before execution.
- Every exact trial is selected by the Search Controller, not manually enumerated by the LLM.
- The root `AGENTS.md`, `data.py`, `evaluate.py`, and `docs/agent-architecture.md` remain the governing constraints.
- New dependencies, a substantially different model family, or work outside the approved catalogue require human review.
