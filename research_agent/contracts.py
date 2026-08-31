"""Immutable benchmark facts shared by every research-agent component.

This module intentionally contains no data loading, model training, or metric
implementation. Those responsibilities remain in data.py and evaluate.py.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkContract:
    """The fixed KuaiRand-Pure benchmark rules the agent must obey."""

    data_dir: Path = Path("KuaiRand-Pure/data")
    label: str = "long_view"
    train_split: str = "train"
    validation_split: str = "valid"
    test_split: str = "test"
    gauc_metric: str = "GAUC"
    ndcg_metric: str = "nDCG@5"
    primary_metric: str = "primary"
    # A full three-seed confirmation can replace the incumbent with a smaller,
    # still material validation gain. The stricter improvement_threshold below
    # remains the project stop-rule threshold.
    acceptance_threshold: float = 0.001
    improvement_threshold: float = 0.002
    non_improvement_limit: int = 3
    target_primary: float = 0.65
    max_experiments: int = 20

    @property
    def selection_split(self) -> str:
        """Candidates are selected only with validation evidence."""
        return self.validation_split

    @property
    def protected_modules(self) -> tuple[str, ...]:
        """Modules that agent experiments must not modify."""
        return ("evaluate.py", "baseline.py")

    @property
    def metric_names(self) -> tuple[str, str, str]:
        return (self.gauc_metric, self.ndcg_metric, self.primary_metric)


BENCHMARK_CONTRACT = BenchmarkContract()
