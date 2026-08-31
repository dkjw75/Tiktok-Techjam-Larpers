from __future__ import annotations

import tempfile
import unittest

from research_agent.agent_team import BroadProposal, Critique
from research_agent.broad_loop import BroadAutonomousLoop
from research_agent.controller import ExperimentController
from research_agent.logger import ResearchLogger
from research_agent.runner import CandidateOutput, ExperimentRunner
from research_agent.safety import SafetyValidator
from research_agent.state import ResearchState
from research_agent.store import ArtifactStore


class FakeTeam:
    def propose(self, _history, _state, _capabilities=()):
        return BroadProposal("A generated candidate may improve ranking quality.", "Test one isolated implementation change.", "training", "generated_candidate", "fm", False, ()), {"model": "fake"}
    def critique(self, _proposal):
        return Critique("approved", "safe"), {"model": "fake"}
    def code(self, _proposal):
        return "def run_candidate(prepared, config, run_dir):\n    return run_torch_fm_extension(prepared, config, run_dir)\n", {"model": "fake"}
    def verify_capability(self, _proposal, _source):
        return {"decision": "verified", "rationale": "matches"}, {"model": "fake"}
    def review(self, _history):
        return {"decision": "explore", "rationale": "continue"}, {"model": "fake"}
    def decide_capability_recovery(self, _proposal, _failure):
        return {"decision": "abandon", "rationale": "not needed for this test"}, {"model": "fake"}


class RejectingTeam(FakeTeam):
    def code(self, _proposal):
        return "def run_candidate(prepared, config, run_dir):\n    return None\n", {"model": "fake"}


class BroadLoopTests(unittest.TestCase):
    def test_broad_proposal_flows_through_critic_coder_runner_and_reviewer(self):
        def loader(_): return {"train": [("x",)], "valid": [("x",)], "test": [("x",)]}
        def candidate(_prepared, _config, _run_dir): return CandidateOutput(["u", "u"], [1, 0], [0.1, 0.9])
        # Fake generated source uses this safe helper after it is injected by the test.
        with tempfile.TemporaryDirectory() as directory:
            store, logger = ArtifactStore(directory), ResearchLogger(ArtifactStore(directory))
            controller = ExperimentController(logger=logger, runner=ExperimentRunner(logger, data_loader=loader), validator=SafetyValidator(max_runtime_seconds=600), state=ResearchState(current_best_primary=0.5))
            # Patch the extension runtime only within this test process.
            import research_agent.models.torch_fm as fm
            original = fm.run_torch_fm_extension
            fm.run_torch_fm_extension = candidate
            try:
                BroadAutonomousLoop(controller, logger, FakeTeam()).run(1)
            finally:
                fm.run_torch_fm_extension = original
            actions = [item["action"] for item in store.read_events()]
            self.assertIn("llm_broad_hypothesis_proposed", actions)
            self.assertIn("isolated_candidate_preflight_passed", actions)
            self.assertIn("llm_evidence_review_completed", actions)
            self.assertEqual(store.read_capabilities(), [])
            self.assertIn("isolated_candidate_preflight_completed", actions)

    def test_repeated_broad_hypothesis_is_suppressed_and_ids_are_not_reused(self):
        def loader(_): return {"train": [("x",)], "valid": [("x",)], "test": [("x",)]}
        def candidate(_prepared, _config, _run_dir): return CandidateOutput(["u", "u"], [1, 0], [0.1, 0.9])
        with tempfile.TemporaryDirectory() as directory:
            store, logger = ArtifactStore(directory), ResearchLogger(ArtifactStore(directory))
            controller = ExperimentController(
                logger=logger,
                runner=ExperimentRunner(logger, data_loader=loader),
                validator=SafetyValidator(max_runtime_seconds=600),
                state=ResearchState(current_best_primary=0.5),
            )
            import research_agent.models.torch_fm as fm
            original = fm.run_torch_fm_extension
            fm.run_torch_fm_extension = candidate
            try:
                BroadAutonomousLoop(controller, logger, FakeTeam()).run(2)
            finally:
                fm.run_torch_fm_extension = original

            proposals = [event for event in store.read_events() if event["action"] == "llm_broad_hypothesis_proposed"]
            self.assertEqual(len(proposals), 1)
            self.assertEqual(proposals[0]["experiment_id"], "exp_002")
            self.assertTrue(any(event["action"] == "hypothesis_suppressed_as_duplicate" for event in store.read_events()))

    def test_unusable_generated_code_is_not_registered_or_run(self):
        with tempfile.TemporaryDirectory() as directory:
            store, logger = ArtifactStore(directory), ResearchLogger(ArtifactStore(directory))
            controller = ExperimentController(
                logger=logger,
                runner=ExperimentRunner(logger, data_loader=lambda _: {"train": [], "valid": [], "test": []}),
                validator=SafetyValidator(max_runtime_seconds=600),
                state=ResearchState(current_best_primary=0.5),
            )

            BroadAutonomousLoop(controller, logger, RejectingTeam()).run(1)

            actions = [item["action"] for item in store.read_events()]
            self.assertIn("isolated_candidate_preflight_failed", actions)
            self.assertIn("candidate_abandoned", actions)
            self.assertEqual(store.read_capabilities(), [])
            generated_iterations = [
                record for record in store.read_iterations()
                if record["search_strategy"] == "llm_isolated_candidate"
            ]
            self.assertEqual(generated_iterations, [])
