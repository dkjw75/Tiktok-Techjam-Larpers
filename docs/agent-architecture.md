# Autonomous MLE Agent Architecture


## Document role and precedence

This is the project's architecture reference for the autonomous research system. It describes the target design and longer-term direction; it is not a second benchmark contract.

- The repository-root `AGENTS.md`, the Starter Kit README, and `evaluate.py` remain authoritative.
- If this document conflicts with those sources, the root rules and fixed benchmark contract win.
- Before designing, implementing, or modifying the research agent, consult this document.
- Implement the architecture in phases. The initial version should establish the smallest reliable closed loop before advanced orchestration or search is added.
- Advanced search, multiple specialist agents, multiple islands, and new model families are future extensions requiring evidence and human review.
- Concrete experiment templates are implementation configuration, not part of this architecture document. Define, approve, and version them separately.

## Project implementation constraints

### Canonical data interface

- Treat the datasets returned by `data.py` as the project's canonical cleaned and prepared data.
- All controllers, runners, PyTorch datasets, and models must obtain benchmark data through `data.py`; they must not read the raw KuaiRand CSV files directly.
- Keep split construction, label preparation, metadata joining, feature encoding, and missing/unknown handling centralized in `data.py`.
- If preprocessing must change, change it deliberately in `data.py`, log the exact diff, run leakage and data-contract checks, and keep the change isolated as part of the experiment.
- Do not perform additional silent cleaning, deduplication, row removal, label rewriting, or split reconstruction downstream.

### Model implementation framework

- Implement candidate model architectures and training loops in PyTorch.
- Represent models as `torch.nn.Module` components and use explicit, reproducible PyTorch training and checkpoint logic.
- Preserve the existing NumPy FM in `baseline.py` as the immutable official reference; do not convert or overwrite that reference implementation.
- Before comparing new architectures, verify that the PyTorch baseline path consumes the same `data.py` outputs and uses the unchanged official evaluator.
- PyTorch is approved for this project. Any additional modelling or search dependency still requires human review under the root project rules.

## Autonomous MLE Experimentation System

This document defines the architecture, responsibilities, operating rules, and implementation guidance for an autonomous machine-learning experimentation system designed to improve recommender-system performance metrics such as **GAUC** and **nDCG@5**.

The central design principle is:

> **Agents reason about what is worth investigating; specialized search algorithms efficiently determine which exact configurations to test.**

The system must explicitly balance **exploration** and **exploitation**, use **multi-fidelity evaluation** to conserve compute, and maintain multiple promising search regions so that it does not become trapped in a local optimum.

---

## 1. System Goal

Given a recommender-system training pipeline and an evaluation function, the system should autonomously discover pipeline changes that improve a target metric while minimizing:

- GPU/CPU compute
- wall-clock time
- unnecessary model training
- LLM token usage
- repeated experiments
- overfitting to a single validation split

For the KuaiRand-Pure benchmark, the optimization target is fixed:

```text
primary = (GAUC + nDCG@5) / 2
```

GAUC and nDCG@5 must also be reported separately. The system must not substitute a different primary objective.

For the fixed objective:

\[
x^* = \arg\max_x f(x)
\]

where:

- `x` is a complete experiment configuration
- `f(x)` is the resulting evaluation score

The true function `f(x)` is expensive and unknown before experimentation. The system therefore combines reasoning agents with efficient black-box optimization.

---

## 2. High-Level Architecture

```text
                         ┌───────────────────────────┐
                         │       LLM AGENT LAYER      │
                         │                           │
                         │ Diagnose                  │
                         │ Generate hypotheses       │
                         │ Decide what to investigate│
                         │ Detect plateaus           │
                         │ Review evidence           │
                         └─────────────┬─────────────┘
                                       │
                              Hypothesis / Search Task
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
                    ▼                                     ▼
             Feature / Data                         Model / Training
                 Agent                                  Agent
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │    SEARCH CONTROLLER      │
                         │                           │
                         │ Exploration/exploitation  │
                         │ Strategy selection        │
                         │ Compute allocation        │
                         │ Region management         │
                         └─────────────┬─────────────┘
                                       │
                ┌──────────────────────┼──────────────────────┐
                ▼                      ▼                      ▼
          EXPLORATION              EXPLOITATION          MULTI-FIDELITY
                │                      │                      │
        Random / Diverse         BO / TPE / Local        ASHA / Hyperband
        Search / Restarts          Search / UCB             / Pruning
                │                      │                      │
                └──────────────────────┼──────────────────────┘
                                       ▼
                         ┌───────────────────────────┐
                         │    EXPERIMENT RUNNER      │
                         │                           │
                         │ Build config              │
                         │ Train model               │
                         │ Evaluate                  │
                         │ Record artifacts          │
                         └─────────────┬─────────────┘
                                       ▼
                             GAUC / nDCG@5 / Loss
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │      EXPERIMENT DB        │
                         │                           │
                         │ Config                    │
                         │ Hypothesis                │
                         │ Metrics                   │
                         │ Search region             │
                         │ Budget                    │
                         │ Parent/lineage            │
                         │ Uncertainty               │
                         │ Compute cost              │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                                  LLM REVIEW
                                       │
                              ┌────────┴────────┐
                              ▼                 ▼
                         Improving           Stuck
                              │                 │
                              ▼                 ▼
                         Exploit            Explore / Restart
                              │                 │
                              └────────┬────────┘
                                       ▼
                                     LOOP
```

---

## 3. Core Design Principle: Separate Strategy from Search

The LLM should **not** be the low-level hyperparameter optimizer.

Bad pattern:

```text
LLM → suggest LR = 0.000743
LLM → suggest LR = 0.000691
LLM → suggest LR = 0.000812
LLM → repeat
```

This consumes tokens while using a language model for a task that classical optimization algorithms handle better.

Preferred pattern:

```text
LLM:
    "Negative sampling appears to be a bottleneck."

Search Controller:
    "Search negative-sampling parameters in this defined space."

TPE/BO/ASHA:
    Select exact configurations and allocate evaluation budget.

Experiment Runner:
    Train and evaluate.

LLM:
    Review the accumulated evidence and decide what concept to investigate next.
```

### Responsibilities by level

| Layer | Main question |
|---|---|
| LLM agent | **What should we investigate? Why?** |
| Search controller | **Which search strategy and how much compute?** |
| Search algorithm | **Which exact configuration should be evaluated next?** |
| ASHA/Hyperband | **Which trials deserve more compute?** |
| Experiment runner | **Run the experiment correctly.** |
| Metrics layer | **How good was the experiment?** |
| Experiment DB | **What have we learned so far?** |

---

## 4. Agent Layer

### 4.0 Autonomous proposal policy

The production research loop must not require the LLM to select from a fixed
catalogue of experiment names. The LLM receives the benchmark contract,
canonical data interface, prior evidence, verified capabilities, and failure
history, then proposes one novel, in-scope, controlled experiment. Safety is
enforced by deterministic validation and isolated execution rather than by a
menu of pre-authored directions.

The default operating mode is self-extending autonomy: the LLM may formulate
and implement complete FM-family candidates—model/training logic, ranking
objectives, sampling, and leakage-safe feature logic—that use only `data.py`
outputs. Each candidate runs only in its own workspace and receives prepared
train/validation rows, safe PyTorch primitives, and a bounded configuration;
it does not receive raw paths or test data. A deterministic host statically
rejects unsafe source, preflights it on a small real-data slice, enforces
budgets, and retains the immutable baseline and evaluator. A new dependency,
substantially different model family, or an experiment outside the approved
project scope still requires human review under the root rules.

Autonomous screening must use one parity-locked configuration except for the
single declared hypothesis change. Short screens are reported separately from
full evaluations. Once at least three comparable autonomous screens exist, the
controller promotes the strongest unpromoted candidate from the global screen
leaderboard—not merely from the same research label—to the full benchmark
budget. Invalid proposals and failed candidate preflights are visible but do
not consume full-evaluation budget. Every run must end with an explicit
completion, budget, review, recovery-failure, or crash status event.

Cross-run learning uses a shared, append-only evidence index. At startup, a
run imports compact copies of completed iteration records and material
candidate failures from earlier run folders; it never edits those original
artifacts. The planner receives a bounded summary of prior metrics, failures,
and repeatedly weak methods before proposing its next hypothesis. This summary
must guide novelty, but it is not a fixed catalogue or a ban list: a method may
be revisited when the LLM states the material difference from the failed prior
attempt. Each completed iteration and generated-candidate failure is appended
to the index immediately for use by later iterations and future runs.

### 4.1 Research/Orchestrator Agent

The main LLM agent is responsible for high-level experimental reasoning.

It should:

1. Read current experiment history.
2. Identify trends and plateaus.
3. Compare GAUC and nDCG@5 behavior.
4. Generate hypotheses.
5. Select a research direction.
6. Define a reasonable search space.
7. Decide whether the current direction is still worth pursuing.
8. Request additional exploration when evidence is weak.
9. Request a restart or change in research direction when the search plateaus.
10. Record a concise rationale for each major search decision.

It should **not** manually micromanage every individual trial.

### 4.2 Specialist Agents

The system may use several specialist agents rather than one general agent.

#### Feature/Data Agent

Investigates:

- user features
- item features
- historical behavior windows
- recency features
- interaction aggregation
- missing-value handling
- data leakage risks
- feature normalization
- feature crossing

Example hypothesis:

> Recent user interactions may contain more predictive signal than long-term aggregate behavior.

#### Model Architecture Agent

Investigates:

- embedding architecture
- number of layers
- hidden dimensions
- attention mechanisms
- residual connections
- ranking heads
- model family changes

Example hypothesis:

> The current model may be under-expressive for the user/item interaction structure.

#### Training Agent

Investigates:

- learning rate
- optimizer
- scheduler
- regularization
- dropout
- batch size
- epochs
- warmup

#### Sampling Agent

Investigates:

- negative-sample ratio
- random negatives
- popularity-based negatives
- hard negatives
- sampling temperature
- candidate filtering

#### Evaluation Agent

Checks:

- metric correctness
- train/validation/test leakage
- user-level aggregation
- sample weighting
- reproducibility
- metric variance across runs

A system should not optimize a metric it cannot trust.

---

## 5. Search Controller

The Search Controller is the central decision layer between high-level hypotheses and low-level optimization.

It decides:

- exploration vs exploitation allocation
- search algorithm
- search region
- evaluation fidelity
- whether to continue, pause, or restart a search
- which candidates should receive additional compute

### 5.1 Search states

The controller should maintain an explicit search state.

Recommended states:

```text
BOOTSTRAP
EXPLORING
PROMISING
EXPLOITING
PLATEAU
RESTARTING
VALIDATING
FINISHED
```

### 5.2 Example state transitions

```text
BOOTSTRAP
   ↓
EXPLORING
   ↓
PROMISING
   ↓
EXPLOITING
   ↓
┌───────────────┐
│               │
│ improvement   │ plateau
│               │
▼               ▼
EXPLOITING     PLATEAU
                 │
          ┌──────┴──────┐
          ▼             ▼
      increase       NEW REGION /
      exploration       RESTART
          │             │
          └──────┬──────┘
                 ▼
             EXPLORING
```

---

## 6. Exploration Strategies

Exploration exists to discover useful regions the current model of the search space does not yet understand.

### 6.1 Random Search

Sample configurations from the allowed search space without trying to favor the current best region.

Use when:

- little information is available
- the search space is large
- a baseline is needed
- avoiding early bias is important

Random search is a deliberate hedge against premature commitment.

### 6.2 Diverse Search

Prefer configurations that are substantially different from prior experiments.

Conceptually:

\[
\text{exploration score}(x)=\text{distance}(x,\text{previous experiments})
\]

Useful distance functions may be defined separately for numeric and categorical variables.

Purpose:

> Explore regions that have not already been repeatedly sampled.

### 6.3 Random Restarts

When the search is stuck, create a new seed configuration far from the current incumbent and begin another local/search trajectory.

Restarts should not be treated as failure. They are a mechanism for escaping local optima.

### 6.4 Bayesian Uncertainty Exploration

If using a Bayesian surrogate, select some experiments because uncertainty is high.

For Upper Confidence Bound:

\[
UCB(x)=\mu(x)+\kappa\sigma(x)
\]

where:

- `μ(x)` = predicted performance
- `σ(x)` = uncertainty
- `κ` = exploration strength

A larger `κ` encourages more exploration.

### 6.5 Evolutionary Exploration

Maintain multiple candidate configurations and mutate them.

Exploration is preserved by:

- mutation
- population diversity
- occasional random immigrants
- maintaining multiple candidate lineages

---

## 7. Exploitation Strategies

Exploitation uses known information to improve promising candidates.

### 7.1 Bayesian Optimization

Use a surrogate model to estimate promising configurations and prioritize expensive evaluations.

Best suited when:

- experiments are expensive
- the search space is reasonably structured
- the number of evaluations is limited

The optimizer should update after observed experiment results.

### 7.2 TPE / Tree-structured Parzen Estimator

TPE is useful when the search space includes:

- categorical variables
- conditional parameters
- discrete values
- irregular parameter spaces

The practical mechanism is to learn which parameter configurations are associated with stronger outcomes and sample preferentially from promising regions.

### 7.3 Local Search

Perturb a strong configuration by small amounts.

Example:

```text
Current best:
embedding = 128
lr = 0.0007

Neighborhood:
embedding = 96
embedding = 112
embedding = 144
embedding = 160
```

This is useful for final refinement.

---

## 8. Multi-Fidelity Search

Never assume every configuration deserves a full training run.

The system should evaluate candidates at progressively higher fidelity.

Example:

```text
1000 candidates
      ↓
cheap training / few epochs
      ↓
100 candidates
      ↓
medium training
      ↓
20 candidates
      ↓
long training
      ↓
5 candidates
      ↓
full training
      ↓
best candidates
```

### 8.1 ASHA / Hyperband

Successive-halving style methods should be used to stop weak candidates early.

Example:

```text
100 trials × 1 epoch
      ↓
keep 30
      ↓
30 trials × 3 epochs
      ↓
keep 10
      ↓
10 trials × 10 epochs
      ↓
keep 3
      ↓
3 trials × full budget
```

This is one of the strongest compute-saving mechanisms in the architecture.

### 8.2 Important constraint

Early performance must be reasonably predictive of final performance.

If a model often starts poorly but eventually wins, overly aggressive early stopping can eliminate the true optimum.

Therefore the system should periodically test whether early-fidelity ranking correlates with final-fidelity ranking.

---

## 9. Exploration vs Exploitation Controller

The controller should dynamically allocate search budget.

A simple initial policy may be:

```text
Early stage:
80% exploration / 20% exploitation

Promising stage:
30% exploration / 70% exploitation

Strong improvement:
20% exploration / 80% exploitation

Plateau:
60% exploration / 40% exploitation
```

These numbers are starting points, not hard-coded truths.

### 9.1 Signals for exploitation

Increase exploitation when:

- recent experiments consistently improve the target metric
- multiple nearby configurations are strong
- Bayesian uncertainty is low around the promising region
- improvement per compute is high
- independent runs reproduce the improvement

### 9.2 Signals for exploration

Increase exploration when:

- improvement has plateaued
- recent candidates are too similar
- surrogate uncertainty is high outside the current region
- all experiments are concentrated around one local region
- changes to hyperparameters no longer produce meaningful gains
- GAUC and nDCG@5 disagree persistently
- the current hypothesis has repeatedly failed

---

## 10. Preventing Local Optima

The system cannot mathematically guarantee that it will never reach a local optimum. It should instead use several independent safeguards.

### 10.1 Maintain multiple search regions

Do not maintain only a single "best configuration".

Maintain an archive of promising regions or lineages:

```text
Region A: embedding ≈ 64
Region B: embedding ≈ 128
Region C: different sampling strategy
Region D: different architecture
```

A weaker current region should not automatically be deleted if it remains scientifically plausible.

### 10.2 Keep an exploration quota

Reserve an explicit portion of compute for unexplored regions.

Example:

```text
70% → exploit known-good regions
20% → investigate uncertain regions
10% → completely new/random directions
```

### 10.3 Plateau detection

Monitor recent improvement.

For a rolling window:

\[
\Delta f = f_{best,t}-f_{best,t-N}
\]

If `Δf` remains below a practical threshold while compute continues to rise, mark the search as plateauing.

Do not endlessly optimize a saturated region.

### 10.4 Random restart

When a search region plateaus, initialize a new region using a randomized or deliberately distant configuration.

### 10.5 Conceptual restart

A hyperparameter plateau does not necessarily mean that the numerical values are wrong.

It may mean the **hypothesis is wrong**.

Example:

```text
Many learning-rate experiments
        ↓
very small metric improvement
        ↓
AGENT:
"Learning rate probably isn't the main bottleneck."
        ↓
Investigate negative sampling / features / architecture
```

This is a key reason to retain the LLM layer.

### 10.6 Diversity constraints

The controller should measure configuration diversity.

Do not allow 50 consecutive experiments to be near-identical unless there is strong evidence that local refinement is still productive.

---

## 11. Multiple Search Islands

A robust implementation should be able to run multiple search islands in parallel.

```text
                 SEARCH SPACE

      ┌───────────────────────────────────┐
      │                                   │
      │  Island A      Island B            │
      │  BO/TPE        BO/TPE              │
      │                                   │
      │          Island C                 │
      │          Evolutionary             │
      │                                   │
      │  Island D                         │
      │  Random exploration               │
      │                                   │
      └───────────────────────────────────┘
```

Each island can explore a different:

- model family
- feature hypothesis
- sampling strategy
- hyperparameter region
- optimizer

### Island migration

Periodically, strong discoveries can be shared across islands.

Example:

```text
Island C discovers:
new negative sampling strategy → nDCG +0.04

        ↓
share discovery

Island A/B/D may test that strategy
```

Do not force every island to converge immediately. Diversity is valuable.

---

## 12. Experiment Database

Every experiment must be recorded in a structured form.

Recommended record:

```yaml
experiment_id: exp_00142
parent_id: exp_00131
hypothesis_id: hyp_00027
region_id: region_03
search_strategy: tpe

config:
  learning_rate: 0.0007
  embedding_dim: 128
  dropout: 0.10
  negative_samples: 20

budget:
  epochs: 10
  gpu_hours: 1.8

metrics:
  gauc: 0.721
  ndcg_at_5: 0.337
  train_loss: 0.841

status: completed
seed: 42

lineage:
  parent_experiment: exp_00131
  mutation_type: learning_rate

notes:
  hypothesis: "More aggressive negative sampling may improve top-k ranking."
```

The database is the system's shared memory.

---

## 13. Experiment Lineage

Experiments should form a graph rather than an unstructured list.

```text
exp_001
   │
   ├── exp_002
   │     └── exp_006
   │
   ├── exp_003
   │
   └── exp_004
         └── exp_007
```

This allows the agent to understand:

- which changes caused improvement
- which branches failed
- which search regions originated from which hypotheses
- whether gains depend on a particular parent configuration

Every experiment should ideally have:

- parent experiment
- hypothesis
- search strategy
- resource budget
- final metrics

---

## 14. Metric Handling

### 14.1 Primary objective

The benchmark primary objective is fixed as the arithmetic mean of GAUC and nDCG@5. Neither component may replace it for candidate acceptance. The agent should still inspect the two component metrics separately when interpreting results.

### 14.2 Secondary metrics

Track GAUC and additional diagnostics simultaneously.

Example:

```text
Primary:    (GAUC + nDCG@5) / 2
Components: GAUC and nDCG@5
Diagnostic: log loss / calibration / coverage / diversity
```

### 14.3 Do not blindly maximize one metric

Possible outcome:

```text
GAUC ↑
nDCG@5 ↓
```

This is not necessarily contradictory. The model may improve overall ranking discrimination while becoming worse at placing the most useful items at the very top.

The agent should investigate metric disagreement rather than averaging everything blindly.

### 14.4 Statistical reliability

Small metric improvements may be noise.

Important candidates should be re-run across multiple seeds or evaluation samples where feasible.

The system should distinguish:

```text
apparent improvement
        vs.
reproducible improvement
```

---

## 15. Token-Efficient Agent Operation

LLM tokens are a scarce resource relative to deterministic search operations.

### Good use of tokens

- analyze experiment trends
- propose high-level hypotheses
- detect saturation
- identify promising research directions
- interpret metric conflicts
- select or modify search spaces
- decide whether to change the problem formulation

### Bad use of tokens

- manually enumerate hundreds of numeric configurations
- rewrite identical experiment instructions
- reason independently about every low-level trial
- perform calculations that a search library can execute
- repeatedly inspect unchanged experiment history

### Batch operation

Prefer:

```text
LLM reasoning pass
      ↓
define search task
      ↓
50–500 automated trials
      ↓
aggregate results
      ↓
LLM review pass
```

rather than:

```text
LLM → one trial
LLM → one trial
LLM → one trial
LLM → one trial
...
```

This can dramatically reduce token consumption.

---

## 16. Coarse-to-Fine Optimization

The system should usually move from cheap/broad search to expensive/narrow search.

```text
                    LARGE SEARCH SPACE
                           │
                           ▼
                  Random / TPE / cheap
                           │
                    many candidates
                           ▼
                       ASHA
                           │
                    remove weak trials
                           ▼
                       BO / TPE
                           │
                   promising regions
                           ▼
                    Local refinement
                           │
                           ▼
                    Full training
                           │
                           ▼
                   independent validation
```

The fundamental rule is:

> **Spend the minimum amount of computation needed to make the next decision.**

---

## 17. Suggested End-to-End Loop

```text
1. Establish baseline.

2. Validate evaluation code.
   - GAUC
   - nDCG@5
   - train/validation/test separation

3. Run a broad bootstrap search.
   - random search
   - diverse sampling
   - cheap fidelity

4. Identify several promising regions.

5. Start multiple search islands.
   - TPE/BO
   - local refinement
   - evolutionary branch

6. Use ASHA/Hyperband to eliminate weak trials.

7. Increase compute for promising candidates.

8. Track improvement per compute.

9. Maintain explicit exploration.

10. Detect plateaus.

11. If plateaued:
    - increase exploration
    - start a new region
    - perform a random restart
    - ask the agent whether the hypothesis itself is wrong

12. Promote strong candidates to full-budget training.

13. Re-run finalists with independent seeds.

14. Compare against the baseline.

15. Record the winning configuration and evidence.

16. Begin the next research cycle.
```

---

## 18. Example Search Cycle

Assume baseline:

```text
GAUC    = 0.710
nDCG@5  = 0.310
```

### Cycle 1: broad exploration

Random/TPE evaluates many cheap candidates.

Best region:

```text
embedding ≈ 128
learning rate ≈ 5e-4
```

### Cycle 2: exploitation

BO refines around that region.

```text
nDCG@5:
0.310 → 0.319 → 0.325 → 0.327
```

### Cycle 3: plateau

Additional local tuning gives:

```text
0.327 → 0.327 → 0.326 → 0.327
```

Controller marks the region as saturated.

### Cycle 4: exploration

A separate island investigates negative sampling.

```text
random negatives → 0.322
hard negatives   → 0.334
```

Now the agent has evidence that sampling, not another tiny learning-rate adjustment, may be the next major direction.

### Cycle 5: combined refinement

Search:

```text
embedding
learning rate
negative sampling
```

using BO/TPE with ASHA.

Potential result:

```text
nDCG@5 = 0.341
```

The improvement is then independently validated.

---

## 19. Controller Decision Policy

A practical controller can use a simple scoring framework.

For each candidate/search region estimate:

```text
Expected value
+ uncertainty value
+ diversity value
+ historical success
--------------------------------
compute cost
```

The controller then chooses experiments that have strong expected information or performance per unit compute.

A conceptual utility function is:

\[
U(x)=\frac{E[\text{improvement}(x)] + \lambda\,\text{uncertainty}(x) + \gamma\,\text{diversity}(x)}{\text{estimated compute}(x)}
\]

This is a design abstraction rather than a required exact formula.

---

## 20. Recommended Practical Stack

A practical initial implementation can use:

```text
LLM orchestration
        ↓
Experiment manager
        ↓
Optuna / TPE / Bayesian optimization
        ↓
ASHA / Hyperband pruning
        ↓
PyTorch training framework
        ↓
GAUC / nDCG@5 evaluator
        ↓
Experiment database
```

The system does not need every algorithm on day one.

### Minimal useful version

Start with:

```text
LLM Agent
   ↓
TPE/Random search
   ↓
ASHA
   ↓
Experiment DB
   ↓
LLM review
```

Then add:

- multiple search islands
- BO with uncertainty-based exploration
- evolutionary search
- automatic restarts
- more sophisticated plateau detection
- multi-objective optimization

---

## 21. Safety and Reproducibility Rules

The system must never silently:

- change train/validation/test splits
- change metric definitions
- alter preprocessing without logging it
- bypass `data.py` by loading raw benchmark CSV files in a controller, runner, or model
- compare results produced by incompatible evaluation code
- reuse test-set results for optimization
- overwrite experiment history
- claim an improvement without recording the exact configuration

Every promoted model must have:

1. exact code/version
2. exact configuration
3. data/version identifier
4. metric implementation/version
5. random seed(s)
6. training budget
7. evaluation results
8. parent experiment/hypothesis

The test set should be treated as a **final confirmation**, not as the routine search objective.

---

## 22. Mandatory Action and Iteration Logging

Logging is part of the evaluated autonomous behavior, not an optional dashboard feature. The structured run artifacts are the source of truth, and the Markdown log is the human-readable report generated from them.

The agent must record every material action that changes or evaluates research state, including:

- selecting or rejecting a hypothesis
- creating or modifying a candidate
- changing a configuration
- starting, stopping, timing out, or retrying training
- invoking evaluation
- accepting, rejecting, or restoring a candidate
- detecting a policy, leakage, data-quality, or resource problem
- receiving a manual intervention

Each iteration record must include:

1. Experiment and parent identifiers.
2. The hypothesis: what the agent intended to try and why.
3. The single controlled change and complete configuration.
4. The code diff applied, stored as a patch or an explicit no-code-change configuration diff.
5. Validation GAUC, nDCG@5, primary score, and delta from the accepted parent.
6. Runtime, configured resource budget, random seed, and available compute/token usage.
7. Decision: accepted, rejected, inconclusive, or failed, with the reason.
8. Every error or recovery event and how the agent handled it.
9. Any restoration of the last accepted candidate.
10. Manual interventions associated with the iteration.

Required outputs:

- append-only JSONL for detailed events and iteration records
- CSV for the metric and decision summary
- Markdown for the readable per-iteration run log
- patch files for code changes
- a short final manual-intervention summary

The manual-intervention summary must report:

- total intervention count, including an explicit zero
- experiment or timestamp for each intervention
- what the human requested or changed
- why intervention was necessary
- its effect on the run or decision

Routine user observation, reading logs, or approving the initial run budget is not an intervention unless it changes an active experiment. The agent must never invent missing metrics, token counts, resource measurements, diffs, errors, recoveries, or interventions; unavailable values must be recorded as unavailable.


---

## 23. Success Criteria

The system is successful when it can demonstrate all of the following:

### Performance

- Find configurations that reliably improve the baseline.
- Preserve improvements under independent re-runs.

### Efficiency

- Use substantially fewer expensive full-training runs than naive grid/random search.
- Stop poor experiments early.
- Minimize LLM token usage by batching search operations.

### Robustness

- Continue exploring after local improvements plateau.
- Maintain multiple plausible search regions.
- Recover from poor early decisions.

### Scientific quality

- Maintain complete experiment lineage.
- Produce reproducible results.
- Distinguish noisy improvements from reliable improvements.
- Keep test evaluation isolated from routine optimization.

---

## 24. Guiding Principle

The system should behave like a scientist with specialized laboratory equipment:

```text
LLM / Agents
    = formulate hypotheses and interpret evidence

Search Algorithms
    = systematically choose experiments

ASHA / Hyperband
    = stop wasting resources on weak experiments

Experiment Runner
    = perform the actual experiment

GAUC / nDCG@5
    = measure the result

Experiment DB
    = preserve scientific memory

Exploration / Restarts / Multiple Islands
    = prevent tunnel vision and local-optimum lock-in
```

The final objective is not merely:

> **"Find the best hyperparameters."**

It is:

> **"Build a closed-loop system that continuously forms better hypotheses, tests them efficiently, learns from the results, explores alternative explanations, and converges toward better models without wasting compute or LLM tokens."**
