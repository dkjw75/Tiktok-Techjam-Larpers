import tempfile
import unittest

from research_agent.controller import ExperimentController
from research_agent.critic import ProposalCritic
from research_agent.fidelity import FidelityManager
from research_agent.logger import ResearchLogger
from research_agent.loop import AutonomousResearchLoop
from research_agent.planner import EvidencePlanner
from research_agent.review import EvidenceReviewer
from research_agent.runner import CandidateOutput, ExperimentRunner
from research_agent.safety import SafetyValidator
from research_agent.search import SearchController
from research_agent.state import ResearchState
from research_agent.store import ArtifactStore


def loader(_data_dir):
    return {"train": [("train",)], "valid": [("valid",)], "test": [("test",)]}


def candidate(_data, _config, _run_dir):
    return CandidateOutput(["u", "u"], [1, 0], [0.9, 0.1])


class AutonomousLoopTests(unittest.TestCase):
    def test_loop_logs_direction_critic_and_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            logger = ResearchLogger(store)
            validator = SafetyValidator(max_runtime_seconds=600)
            controller = ExperimentController(
                logger=logger,
                runner=ExperimentRunner(logger, data_loader=loader),
                validator=validator,
                state=ResearchState(current_best_primary=0.5),
            )
            loop = AutonomousResearchLoop(
                controller=controller,
                logger=logger,
                planner=EvidencePlanner(seed=0),
                search=SearchController(seed=0),
                critic=ProposalCritic(validator),
                reviewer=EvidenceReviewer(),
                fidelity=FidelityManager(),
                candidate=candidate,
            )

            results = loop.run(max_cycles=1)

            self.assertEqual(len(results), 1)
            actions = [event["action"] for event in store.read_events()]
            self.assertIn("research_direction_proposed", actions)
            self.assertIn("proposal_critic_reviewed", actions)
            self.assertEqual(store.read_iterations()[0]["direction_id"], results[0].direction.direction_id)

    def test_near_baseline_rejected_low_fidelity_trial_can_be_promoted_by_asha(self):
        class AlwaysPromote(FidelityManager):
            def should_promote(self, *args, **kwargs): return True

        def near_miss(_data, _config, _run_dir):
            return CandidateOutput(["u", "u"], [1, 0], [0.1, 0.9])

        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            logger = ResearchLogger(store)
            validator = SafetyValidator(max_runtime_seconds=600)
            controller = ExperimentController(
                logger=logger,
                runner=ExperimentRunner(logger, data_loader=loader),
                validator=validator,
                state=ResearchState(current_best_primary=0.61),
            )
            loop = AutonomousResearchLoop(
                controller=controller, logger=logger, planner=EvidencePlanner(seed=0),
                search=SearchController(seed=0), critic=ProposalCritic(validator),
                reviewer=EvidenceReviewer(), fidelity=AlwaysPromote(), candidate=near_miss,
            )
            loop.run(max_cycles=1)
            self.assertIn("asha_promotion", [item["search_strategy"] for item in store.read_iterations()])
