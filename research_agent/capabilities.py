"""Append-only verified capabilities created by the autonomous coding role."""
from __future__ import annotations

import hashlib
from pathlib import Path
from dataclasses import asdict
from typing import Any, Mapping

from .agent_team import BroadProposal
from .store import ArtifactStore


class CapabilityRegistry:
    def __init__(self, store: ArtifactStore) -> None:
        self.store = store

    def register(
        self,
        proposal: BroadProposal,
        source: str,
        verification: Mapping[str, Any],
        *,
        config: Mapping[str, Any] | None = None,
        source_path: str | Any | None = None,
        hook_manifest: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        source_hash = hashlib.sha256(source.encode()).hexdigest()
        record = {
            "capability_id": f"cap_{source_hash[:12]}",
            "proposal": asdict(proposal),
            "source_sha256": source_hash,
            "verification": dict(verification),
            "config": dict(config or {}),
            "source_path": str(source_path) if source_path else None,
            "hook": dict(hook_manifest or {}),
            "host_runtime": self._host_runtime_record(),
        }
        self.store.append_capability(record)
        return record

    @staticmethod
    def _host_runtime_record() -> dict[str, str]:
        runtime = Path(__file__).parent / "models" / "torch_fm.py"
        source = runtime.read_bytes()
        return {"path": str(runtime), "sha256": hashlib.sha256(source).hexdigest()}
