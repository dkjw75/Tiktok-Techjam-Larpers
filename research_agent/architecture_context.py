"""Loads the live research-architecture guidance for LLM research roles.

The architecture document guides non-deterministic research decisions.  It is
not a replacement for the immutable benchmark contract enforced in Python.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .contracts import BenchmarkContract


class ArchitectureContextError(RuntimeError):
    """Raised when the required architecture guidance cannot be loaded."""


@dataclass(frozen=True)
class ArchitectureContext:
    """Versioned, read-only guidance injected into every LLM specialist role."""

    source_path: Path
    source_sha256: str
    guidance: str
    benchmark_boundaries: str

    @classmethod
    def from_file(cls, source_path: str | Path, contract: BenchmarkContract) -> "ArchitectureContext":
        path = Path(source_path)
        if not path.is_file():
            raise ArchitectureContextError(f"architecture guidance is missing: {path}")
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            raise ArchitectureContextError(f"architecture guidance is empty: {path}")
        return cls(
            source_path=path,
            source_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            guidance=content,
            benchmark_boundaries=_benchmark_boundaries(contract),
        )

    def prompt_context(self) -> str:
        """Keep decision guidance distinct from non-negotiable code safeguards."""
        return (
            "LIVE ARCHITECTURE GUIDANCE (read and follow for this decision):\n"
            f"{self.guidance}\n\n"
            "IMMUTABLE BENCHMARK BOUNDARIES (do not override):\n"
            f"{self.benchmark_boundaries}"
        )

    def artifact_record(self) -> dict[str, str]:
        return {
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "guidance": self.guidance,
            "benchmark_boundaries": self.benchmark_boundaries,
        }


def _benchmark_boundaries(contract: BenchmarkContract) -> str:
    return "\n".join(
        (
            "- Use only KuaiRand-Pure through data.py; never read raw CSV files directly or use external data.",
            "- Select experiments with validation only; test data is final confirmation only.",
            f"- The target is {contract.label}; primary is ({contract.gauc_metric} + {contract.ndcg_metric}) / 2.",
            "- Keep evaluate.py and baseline.py immutable and preserve the official evaluator.",
            "- Candidate training/model code must use PyTorch and run in an isolated workspace.",
            "- New dependencies, model families, and work outside the approved scope require human review.",
        )
    )
