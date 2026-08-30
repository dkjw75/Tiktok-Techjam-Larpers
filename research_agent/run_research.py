"""Command-line entrypoint for a complete autonomous research run."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import replace
from pathlib import Path

from .controller import ExperimentController
from .critic import ProposalCritic
from .fidelity import FidelityManager
from .logger import ResearchLogger
from .loop import AutonomousResearchLoop
from .llm_planner import LLMPlanningError, OpenAIPlanner, ResilientPlanner
from .planner import EvidencePlanner, ResearchPlanner
from .regions import SearchRegionManager
from .review import EvidenceReviewer
from .runner import ExperimentRunner
from .safety import SafetyValidator
from .search import SearchController
from .seed_validation import confirm_promotion_candidate
from .store import ArtifactStore
from .contracts import BENCHMARK_CONTRACT
from .manifest import ensure_run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Operate the validation-only KuaiRand research agent.")
    parser.add_argument(
        "command",
        nargs="?",
        choices=(
            "run",
            "resume",
            "status",
            "confirm-seeds",
            "finalize",
            "recover-finalization",
        ),
        default="run",
    )
    parser.add_argument("--cycles", type=int, default=50, help="maximum research cycles to attempt (capped at the safety budget)")
    parser.add_argument("--artifact-dir", default="runs", help="append-only artifact directory")
    parser.add_argument("--data-dir", default=str(BENCHMARK_CONTRACT.data_dir))
    parser.add_argument("--submission-path")
    parser.add_argument(
        "--confirm-final-evaluation",
        action="store_true",
        help="explicitly authorize the one-time test/submission finalization boundary",
    )
    parser.add_argument(
        "--confirm-finalization-recovery",
        action="store_true",
        help="explicitly authorize recovery of an interrupted finalization with no outputs",
    )
    parser.add_argument(
        "--finalization-recovery-reason",
        default="",
        help="audit-log reason for authorizing an interrupted finalization retry",
    )
    args = parser.parse_args()
    if args.cycles <= 0:
        parser.error("--cycles must be positive")

    contract = replace(BENCHMARK_CONTRACT, data_dir=Path(args.data_dir))
    store = ArtifactStore(args.artifact_dir)
    persisted = store.read_root_json("state.json")

    if args.command == "status":
        if persisted:
            ensure_run_manifest(store, contract, create=False)
        print(json.dumps(persisted or {"status": "not_started"}, indent=2, sort_keys=True))
        return

    if args.command == "finalize":
        from .finalize import finalize_run

        ensure_run_manifest(store, contract, create=False)
        final = finalize_run(
            store,
            contract=contract,
            submission_path=args.submission_path,
            confirm_final_evaluation=args.confirm_final_evaluation,
        )
        print(f"Selected: {final.selected_experiment_id} (validation primary {final.selection_primary:.4f})")
        if final.test_metrics:
            print(f"Final test confirmation: {final.test_metrics['primary']:.4f}")
        else:
            print("No validation candidate beat the official baseline; wrote its submission.")
        print(f"Log: {final.report_path}")
        print(f"Submission: {final.submission_path}")
        return

    if args.command == "recover-finalization":
        from .finalization_recovery import recover_interrupted_finalization

        ensure_run_manifest(store, contract, create=False)
        recovered = recover_interrupted_finalization(
            store,
            confirm_recovery=args.confirm_finalization_recovery,
            reason=args.finalization_recovery_reason,
            submission_path=args.submission_path,
        )
        print(json.dumps(recovered.as_dict(), indent=2, sort_keys=True))
        return

    if args.command == "confirm-seeds":
        from .seed_validation import confirm_selected_candidate

        ensure_run_manifest(store, contract, create=False)
        logger = ResearchLogger(store)
        result = confirm_selected_candidate(
            store,
            ExperimentRunner(logger, contract=contract),
            contract=contract,
        )
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
        return

    if args.command == "run" and persisted:
        parser.error("artifact directory already contains state; use the resume command")
    if args.command == "resume" and not persisted:
        parser.error("cannot resume: artifact directory has no state.json")

    ensure_run_manifest(
        store,
        contract,
        create=args.command == "run",
        environment=_environment_metadata() if args.command == "run" else None,
    )

    from .models.dispatch import run_candidate

    logger = ResearchLogger(store)
    validator = SafetyValidator(max_runtime_seconds=2400.0)
    runner = ExperimentRunner(logger, contract=contract)
    controller = ExperimentController(
        logger=logger,
        runner=runner,
        validator=validator,
        contract=contract,
        max_iterations=contract.max_experiments,
        require_seed_confirmation=True,
    )
    logger.log_action("research_run_started", details={"max_cycles": args.cycles, "data_interface": "data.py", "selection_split": contract.validation_split})
    try:
        planner: ResearchPlanner = ResilientPlanner(
            OpenAIPlanner.from_environment(),
            EvidencePlanner(seed=0),
        )
    except LLMPlanningError as exc:
        planner = EvidencePlanner(seed=0)
        logger.log_action(
            "llm_configuration_failed",
            details={
                "error": str(exc),
                "recovery": "explicit deterministic planner fallback; add OPENAI_API_KEY to resume LLM planning",
                "planner_mode": "deterministic_fallback",
                "degraded": True,
            },
        )
    loop = AutonomousResearchLoop(
        controller=controller,
        logger=logger,
        planner=planner,
        search=SearchController(seed=0),
        critic=ProposalCritic(validator),
        reviewer=EvidenceReviewer(),
        fidelity=FidelityManager(),
        regions=SearchRegionManager(),
        candidate=run_candidate,
        promotion_confirmer=lambda record: confirm_promotion_candidate(
            store,
            runner,
            record,
            candidate=run_candidate,
            contract=contract,
        ).as_dict(),
    )
    try:
        results = loop.run(min(args.cycles, contract.max_experiments))
    except LLMPlanningError as exc:
        logger.log_action("llm_planning_failed", details={"error": str(exc), "recovery": "run paused; correct the LLM configuration and resume"})
        raise SystemExit(str(exc))
    # Fold this invocation's real compute into the carried-forward total before
    # persisting, so a paused run does not silently burn its research budget.
    controller.checkpoint_active_runtime()
    logger.log_action("research_run_finished", details={"cycles_completed": len(results), "stop_reason": controller.state.stop_reason, "active_runtime_seconds": controller.state.active_runtime_seconds})
    print(json.dumps(controller.state.as_dict(), indent=2, sort_keys=True))
    if controller.state.stopped:
        print("Research reached a terminal state without test access. Confirm seeds before finalization.")
    else:
        print("Research paused at the requested cycle limit; resume to continue.")


def _environment_metadata() -> dict[str, object]:
    try:
        import torch  # type: ignore[import-not-found]

        torch_environment: dict[str, object] = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        }
    except ModuleNotFoundError:
        torch_environment = {
            "torch_version": "unavailable",
            "cuda_available": False,
            "cuda_device_count": 0,
        }
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor() or "unavailable",
        **torch_environment,
    }


if __name__ == "__main__":
    main()
