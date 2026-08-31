import tempfile
import unittest

from research_agent.agent_team import BroadProposal
from research_agent.capabilities import CapabilityRegistry
from research_agent.store import ArtifactStore


class CapabilityRegistryTests(unittest.TestCase):
    def test_verified_candidate_is_append_only_capability(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            proposal = BroadProposal("A" * 30, "B" * 30, "training", "loss", "fm", False, ())
            record = CapabilityRegistry(store).register(proposal, "def run_candidate(): pass", {"decision": "verified"})
            self.assertEqual(store.read_capabilities()[0]["capability_id"], record["capability_id"])
            self.assertIn("host_runtime", record)
            self.assertEqual(record["hook"], {})
