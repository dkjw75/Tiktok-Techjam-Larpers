"""Command-line entrypoint for a complete autonomous research run."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .architecture_context import ArchitectureContext, ArchitectureContextError
from .controller import ExperimentController
from .finalize import finalize_run
from .logger import ResearchLogger
from .agent_team import LLMResearchTeam
from .broad_loop import BroadAutonomousLoop
from .llm_planner import LLMPlanningError, OpenAIPlanner
from .runner import ExperimentRunner
from .safety import SafetyValidator
from .store import ArtifactStore
from .research_memory import ResearchMemory
from .contracts import BENCHMARK_CONTRACT


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the validation-only KuaiRand autonomous research loop.")
    parser.add_argument("--cycles", type=int, default=20, help="maximum research cycles to attempt (capped at the safety budget)")
    parser.add_argument("--artifact-dir", default="runs/current", help="append-only artifact directory")
    parser.add_argument("--data-dir", default=str(BENCHMARK_CONTRACT.data_dir))
    parser.add_argument(
        "--mode", choices=("autonomous",), default="autonomous",
        help="full isolated-candidate autonomous research workflow",
    )
    args = parser.parse_args()
    if args.cycles <= 0:
        parser.error("--cycles must be positive")

    workspace_root = Path(__file__).resolve().parents[1]
    runs_root = (workspace_root / "runs").resolve()
    artifact_dir = Path(args.artifact_dir)
    artifact_path = (workspace_root / artifact_dir).resolve() if not artifact_dir.is_absolute() else artifact_dir.resolve()
    try:
        artifact_path.relative_to(runs_root)
    except ValueError:
        parser.error("--artifact-dir must be inside the project's runs/ folder")

    contract = replace(BENCHMARK_CONTRACT, data_dir=Path(args.data_dir))
    store = ArtifactStore(artifact_path)
    logger = ResearchLogger(store)
    research_memory = ResearchMemory(runs_root)
    memory_status = research_memory.bootstrap(exclude_run=store.root)
    logger.log_action("cross_run_memory_loaded", details=memory_status)
    architecture_path = Path(__file__).resolve().parents[1] / "docs" / "agent-architecture.md"
    try:
        architecture = ArchitectureContext.from_file(architecture_path, contract)
    except ArchitectureContextError as exc:
        logger.log_action("architecture_context_failed", details={"error": str(exc), "recovery": "restore docs/agent-architecture.md before running research"})
        raise SystemExit(str(exc))
    store.write_root_json("architecture_context.json", architecture.artifact_record())
    validator = SafetyValidator(max_runtime_seconds=600.0)
    controller = ExperimentController(
        logger=logger,
        runner=ExperimentRunner(logger, contract=contract),
        validator=validator,
        contract=contract, max_iterations=min(args.cycles, contract.max_experiments),
        research_memory=research_memory,
    )
    logger.log_action("research_run_started", details={"max_cycles": args.cycles, "data_interface": "data.py", "selection_split": contract.validation_split, "architecture_source": str(architecture_path), "architecture_sha256": architecture.source_sha256})
    try:
        planner = OpenAIPlanner.from_environment()
    except LLMPlanningError as exc:
        logger.log_action("llm_configuration_failed", details={"error": str(exc), "recovery": "add OPENAI_API_KEY to .env and rerun"})
        raise SystemExit(str(exc))
    team = LLMResearchTeam(planner.client, architecture)
    loop = BroadAutonomousLoop(controller, logger, team, research_memory=research_memory)
    try:
        results = loop.run(min(args.cycles, contract.max_experiments))
    except LLMPlanningError as exc:
        logger.log_action("llm_planning_failed", details={"error": str(exc), "recovery": "run paused; correct the LLM configuration and resume"})
        logger.log_action("research_run_finished", details={"status": "stopped_after_recovery_failure", "mode": args.mode, "reason": str(exc)})
        raise SystemExit(str(exc))
    except Exception as exc:
        logger.log_action("research_run_finished", details={"status": "unexpected_crash", "mode": args.mode, "reason": f"{type(exc).__name__}: {exc}"})
        raise
    status = "budget_reached" if controller.state.stop_reason else "completed"
    logger.log_action(
        "research_run_finished",
        details={"status": status, "cycles_completed": len(results) if isinstance(results, list) else results, "mode": args.mode, "stop_reason": controller.state.stop_reason},
    )
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
