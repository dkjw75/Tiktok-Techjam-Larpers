"""Command-line entrypoint for a complete autonomous research run."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .controller import ExperimentController
from .critic import ProposalCritic
from .fidelity import FidelityManager
from .finalize import finalize_run
from .logger import ResearchLogger
from .loop import AutonomousResearchLoop
from .models.torch_fm import run_torch_fm_candidate
from .agent_team import LLMResearchTeam
from .broad_loop import BroadAutonomousLoop
from .llm_planner import LLMPlanningError, OpenAIPlanner
from .regions import SearchRegionManager
from .review import EvidenceReviewer
from .runner import ExperimentRunner
from .safety import SafetyValidator
from .search import SearchController
from .store import ArtifactStore
from .contracts import BENCHMARK_CONTRACT


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the validation-only KuaiRand autonomous research loop.")
    parser.add_argument("--cycles", type=int, default=20, help="maximum research cycles to attempt (capped at the safety budget)")
    parser.add_argument("--artifact-dir", default="runs", help="append-only artifact directory")
    parser.add_argument("--data-dir", default=str(BENCHMARK_CONTRACT.data_dir))
    args = parser.parse_args()
    if args.cycles <= 0:
        parser.error("--cycles must be positive")

    contract = replace(BENCHMARK_CONTRACT, data_dir=Path(args.data_dir))
    store = ArtifactStore(args.artifact_dir)
    logger = ResearchLogger(store)
    validator = SafetyValidator(max_runtime_seconds=600.0)
    controller = ExperimentController(
        logger=logger,
        runner=ExperimentRunner(logger, contract=contract),
        validator=validator,
        contract=contract, max_iterations=min(args.cycles, contract.max_experiments),
    )
    logger.log_action("research_run_started", details={"max_cycles": args.cycles, "data_interface": "data.py", "selection_split": contract.validation_split})
    try:
        planner = OpenAIPlanner.from_environment()
    except LLMPlanningError as exc:
        logger.log_action("llm_configuration_failed", details={"error": str(exc), "recovery": "add OPENAI_API_KEY to .env and rerun"})
        raise SystemExit(str(exc))
    loop = BroadAutonomousLoop(controller, logger, LLMResearchTeam(planner.client))
    try:
        results = loop.run(min(args.cycles, contract.max_experiments))
    except LLMPlanningError as exc:
        logger.log_action("llm_planning_failed", details={"error": str(exc), "recovery": "run paused; correct the LLM configuration and resume"})
        raise SystemExit(str(exc))
    logger.log_action("research_run_finished", details={"cycles_completed": results, "stop_reason": controller.state.stop_reason})
    try:
        final = finalize_run(store, contract=contract)
    except Exception as exc:
        logger.log_action("finalization_failed", details={"error": f"{type(exc).__name__}: {exc}", "recovery": "no automatic retry available"})
        raise
    print(f"Selected: {final.selected_experiment_id} (validation primary {final.selection_primary:.4f})")
    if final.test_metrics:
        print(f"Final test confirmation: {final.test_metrics['primary']:.4f}")
    else:
        print("No candidate cleared the validation improvement threshold; wrote the official baseline submission.")
    print(f"Log: {final.report_path}")
    print(f"Submission: {final.submission_path}")


if __name__ == "__main__":
    main()
