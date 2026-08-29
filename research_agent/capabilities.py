"""Append-only verified capabilities created by the autonomous coding role."""
from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any, Mapping

from .agent_team import BroadProposal
from .store import ArtifactStore


class CapabilityRegistry:
    def __init__(self, store: ArtifactStore) -> None:
        self.store = store

    def register(self, proposal: BroadProposal, source: str, verification: Mapping[str, Any]) -> dict[str, Any]:
        record = {"capability_id": f"cap_{hashlib.sha256(source.encode()).hexdigest()[:12]}", "proposal": asdict(proposal), "source_sha256": hashlib.sha256(source.encode()).hexdigest(), "verification": dict(verification)}
        self.store.append_capability(record)
        return record
