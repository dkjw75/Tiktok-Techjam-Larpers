# Agent Architecture Conformance

The runnable agent follows `agent-architecture.md` as follows.

- **LLM agent layer:** `LLMResearchTeam` has independent planner, critic, coding, and evidence-review calls.
- **Strategy vs search:** LLM roles state the research idea and controlled change; the runner/evaluator remain deterministic. Generated source may not choose data splits or metrics.
- **Canonical data:** generated candidates receive `PreparedData` only from `ExperimentRunner`, which gets it through `data.py`; test rows are never exposed during research.
- **Isolation:** generated `candidate.py` is stored under the run artifact's `candidate_workspaces/`, never written into the shared implementation.
- **Safety:** generated code has no imports, filesystem builtins, network tools, subprocesses, or dynamic execution. The deterministic validator protects evaluator/baseline, split choice, external data, dependencies, and model-family review.
- **Evidence:** planner/critic/coder/reviewer actions, LLM usage metadata, candidate source diff, metrics, recovery, and decisions are append-only artifacts.
- **Search and stopping:** accepted results use validation primary; three non-improvements trigger a new LLM proposal; the run ends at primary 0.65 or the twenty-experiment hard budget.

Future architecture extensions such as TPE/ASHA, multiple parallel islands, and independently reproducible finalist seeds are intentionally not claimed as implemented.
