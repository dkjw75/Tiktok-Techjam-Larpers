from __future__ import annotations

import csv
import json
import tempfile
import unittest

from research_agent.research_memory import ResearchMemoryProjector, render_cycle_summary
from research_agent.state import ResearchState
from research_agent.store import ArtifactStore


class ResearchMemoryTests(unittest.TestCase):
    def test_projects_resolved_ledger_state_and_negative_lessons(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            accepted = {
                "experiment_id": "exp_001",
                "comparison_incumbent_id": "baseline",
                "hypothesis": "diverse ranks improve the ensemble",
                "direction_id": "rank_ensemble",
                "search_region_id": "region_rank_ensemble",
                "changed_factors": ["member_set"],
                "config": {"loss": "ensemble", "fidelity": "full", "seed": 0},
                "metrics": {"GAUC": 0.67, "nDCG@5": 0.538, "primary": 0.604},
                "delta_primary": 0.0024,
                "runtime_seconds": 10.0,
                "decision": "pending_confirmation",
                "runner_metadata": {
                    "model": "fm_rank_ensemble",
                    "seed": 0,
                    "epochs_run": 10,
                    "best_epoch": 7,
                    "stopped_by": "early_stopping",
                },
            }
            failed = {
                "experiment_id": "exp_002",
                "comparison_incumbent_id": "exp_001",
                "hypothesis": "a failed branch",
                "direction_id": "listwise",
                "search_region_id": "region_listwise",
                "changed_factors": ["loss"],
                "config": {"loss": "listwise", "fidelity": "full", "seed": 0},
                "metrics": {},
                "decision": "failed",
                "error": "worker failed",
                "recovery": "champion preserved",
                "runner_metadata": {},
            }
            store.append_iteration(accepted)
            store.append_iteration(failed)
            state = ResearchState(
                current_best_experiment_id="exp_001",
                current_best_primary=0.6039,
                completed_iterations=2,
            )
            store.write_root_json("state.json", state.as_dict())
            certificate = {
                "candidate_scores": [0.604, 0.603, 0.604],
                "comparator_scores": [0.601468756352959, 0.6017609746263624, 0.6010899806389984],
                "comparator_experiment_id": "baseline",
                "confirmed": True,
                "wins": 2,
            }
            store.write_root_json(
                "promotion_resolutions.json",
                {"exp_001": {"decision": "accepted", "certificate": certificate}},
            )

            ResearchMemoryProjector(store).write_all()

            projected = json.loads((store.root / "research_state.json").read_text())
            self.assertEqual(projected["current_champion"], "exp_001")
            self.assertAlmostEqual(
                projected["baseline_metrics"]["matched_seed_mean_primary"],
                0.6014399038727732,
            )
            self.assertEqual(
                projected["champion_metrics"]["selected_run"]["GAUC"], 0.67
            )
            self.assertAlmostEqual(
                projected["champion_metrics"]["seed_confirmed_primary_mean"],
                0.6036666666666667,
            )
            with (store.root / "experiment_ledger.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["resolved_decision"], "accepted")
            self.assertEqual(rows[1]["resolved_decision"], "failed")
            self.assertAlmostEqual(
                float(rows[0]["delta_vs_baseline"]),
                0.604 - 0.6014399038727732,
            )
            lessons = (store.root / "research_lessons.md").read_text(encoding="utf-8")
            self.assertIn("## exp_002", lessons)
            self.assertIn("worker failed", lessons)

    def test_cycle_summary_contains_every_required_field(self) -> None:
        summary = render_cycle_summary(
            cycle=3,
            hypothesis="one controlled factor",
            record={
                "changed_factors": ["member_set"],
                "config": {"fidelity": "low"},
                "metrics": {"primary": 0.603, "GAUC": 0.67, "nDCG@5": 0.536},
                "delta_primary": 0.0014,
                "decision": "screened",
                "runner_metadata": {},
            },
            state={"completed_iterations": 4, "active_runtime_seconds": 100.0},
        )
        for field in (
            "CYCLE:",
            "HYPOTHESIS:",
            "CHANGE:",
            "FIDELITY:",
            "PRIMARY:",
            "GAUC:",
            "NDCG@5:",
            "DELTA VS BASELINE:",
            "DELTA VS CHAMPION:",
            "ENSEMBLE DELTA:",
            "SEED STATUS:",
            "DECISION:",
            "LESSON:",
            "NEXT HYPOTHESIS:",
            "BUDGET REMAINING:",
        ):
            self.assertIn(field, summary)


if __name__ == "__main__":
    unittest.main()
