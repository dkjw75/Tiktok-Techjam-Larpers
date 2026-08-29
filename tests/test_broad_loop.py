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
    def propose(self, _history, _state):
        return BroadProposal("A generated candidate may improve ranking quality.", "Test one isolated implementation change.", "training", "generated_candidate", "fm", False, ()), {"model": "fake"}
    def critique(self, _proposal):
        return Critique("approved", "safe"), {"model": "fake"}
    def code(self, _proposal):
        return "def run_candidate(prepared, config, run_dir):\n    return run_torch_fm_candidate(prepared, config, run_dir)\n", {"model": "fake"}
    def verify_capability(self, _proposal, _source):
        return {"decision": "verified", "rationale": "matches"}, {"model": "fake"}
    def review(self, _history):
        return {"decision": "explore", "rationale": "continue"}, {"model": "fake"}


class BroadLoopTests(unittest.TestCase):
    def test_broad_proposal_flows_through_critic_coder_runner_and_reviewer(self):
        def loader(_): return {"train": [("x",)], "valid": [("x",)], "test": [("x",)]}
        def candidate(_prepared, _config, _run_dir): return CandidateOutput(["u", "u"], [1, 0], [0.9, 0.1])
        # Fake generated source uses this safe helper after it is injected by the test.
        with tempfile.TemporaryDirectory() as directory:
            store, logger = ArtifactStore(directory), ResearchLogger(ArtifactStore(directory))
            controller = ExperimentController(logger=logger, runner=ExperimentRunner(logger, data_loader=loader), validator=SafetyValidator(max_runtime_seconds=600), state=ResearchState(current_best_primary=0.5))
            # Patch the standard candidate helper only within this test process.
            import research_agent.models.torch_fm as fm
            original = fm.run_torch_fm_candidate
            fm.run_torch_fm_candidate = candidate
            try:
                BroadAutonomousLoop(controller, logger, FakeTeam()).run(1)
            finally:
                fm.run_torch_fm_candidate = original
            actions = [item["action"] for item in store.read_events()]
            self.assertIn("llm_broad_hypothesis_proposed", actions)
            self.assertIn("coding_subagent_completed", actions)
            self.assertIn("llm_evidence_review_completed", actions)
