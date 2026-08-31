"""Closed-loop orchestration across planner, search, safety, runner, and review."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Callable, Mapping, Sequence

from .controller import ExperimentController, IterationResult
from .critic import ProposalCritic
from .fidelity import FidelityManager
from .logger import ResearchLogger
from .planner import ResearchDirection, ResearchPlanner
from .review import EvidenceReviewer
from .research_memory import (
    FROZEN_CALIBRATED_BASELINE_PRIMARY,
    render_cycle_summary,
)
from .regions import SearchRegionManager
from .runner import CandidateCallable
from .search import SearchController


PromotionConfirmer = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class CycleResult:
    direction: ResearchDirection
    screen_iterations: tuple[IterationResult, ...]
    promoted_iteration: IterationResult | None
    calibration_iterations: tuple[IterationResult, ...] = ()
    promotion_iterations: tuple[IterationResult, ...] = ()

    @property
    def iteration(self) -> IterationResult:
        """Compatibility view for callers written before screen batches."""
        return self.screen_iterations[0]


class AutonomousResearchLoop:
    """Runs evidence-based cycles; no hypothesis sequence is hard-coded."""

    def __init__(
        self,
        *,
        controller: ExperimentController,
        logger: ResearchLogger,
        planner: ResearchPlanner,
        search: SearchController,
        critic: ProposalCritic,
        reviewer: EvidenceReviewer,
        fidelity: FidelityManager,
        regions: SearchRegionManager | None = None,
        candidate: CandidateCallable,
        promotion_confirmer: PromotionConfirmer | None = None,
    ) -> None:
        self.controller = controller
        self.logger = logger
        self.planner = planner
        self.search = search
        self.critic = critic
        self.reviewer = reviewer
        self.fidelity = fidelity
        self.regions = regions or SearchRegionManager()
        self.candidate = candidate
        self.promotion_confirmer = promotion_confirmer

    def run(self, max_cycles: int) -> list[CycleResult]:
        results: list[CycleResult] = []
        self._resume_pending_confirmations()
        for _ in range(max_cycles):
            calibration_due = (self.controller.state.completed_cycles + 1) % 5 == 0
            promotion_width = self.controller.state.screen_promotion_width
            required_trials = 6 if calibration_due else 3 + promotion_width
            if (
                self.controller.state.stopped
                or not self.controller.ensure_cycle_capacity(required_trials)
            ):
                break
            history = self.logger.store.read_iterations()
            resolutions = (
                self.logger.store.read_root_json("promotion_resolutions.json") or {}
            )
            resolved_history = _resolved_history(history, resolutions)
            review = self.reviewer.review(
                history,
                self.controller.state,
                resolutions,
            )
            self.logger.log_action(
                "evidence_reviewed",
                details={
                    **review.as_dict(),
                    "regions": [
                        snapshot.__dict__
                        for snapshot in self.regions.snapshots(resolved_history)
                    ],
                },
            )
            planning_history = [
                *resolved_history,
                {
                    "record_type": "evidence_review",
                    "evidence_review": review.as_dict(),
                },
            ]
            direction = self.planner.propose(planning_history, self.controller.state)
            llm_metadata = getattr(self.planner, "last_metadata", None)
            if llm_metadata:
                self.logger.log_action("llm_hypothesis_generated", details=dict(llm_metadata))
            self.logger.log_action("research_direction_proposed", details=direction.as_dict())
            search_state = self.regions.choose_search_state(
                direction,
                resolved_history,
                self.controller.state,
                review,
            )
            screened: list[tuple[Any, IterationResult]] = []
            for _screen_index in range(3):
                proposals = self.search.propose_batch(
                    direction,
                    self.controller.state,
                    history,
                    count=1,
                    search_state=search_state,
                    reserved_ids=self.logger.store.run_experiment_ids(),
                )
                if not proposals:
                    break
                proposal = proposals[0]
                critic_result = self.critic.review(proposal, history)
                self.logger.log_action(
                    "proposal_critic_reviewed",
                    experiment_id=proposal.experiment_id,
                    details={"approved": critic_result.approved, "reasons": list(critic_result.reasons)},
                )
                if not critic_result.approved:
                    iteration = self.controller.record_critic_rejection(
                        proposal,
                        reasons=critic_result.reasons,
                    )
                else:
                    iteration = self.controller.run_iteration(proposal, self.candidate)
                screened.append((proposal, iteration))
                history = self.logger.store.read_iterations()
                if self.controller.state.stopped:
                    break
            if not screened:
                self.controller.stop_search_exhausted(
                    "search controller found no unique safe configurations"
                )
                break

            promotable = [
                (proposal, iteration)
                for proposal, iteration in screened
                if iteration.decision == "screened"
                and iteration.metrics
                and self.fidelity.should_promote(
                    iteration.metrics,
                    self.controller.state.current_best_primary,
                )
            ]
            promotable.sort(
                key=lambda item: float((item[1].metrics or {})["primary"]),
                reverse=True,
            )
            promoted = None
            promotions: list[IterationResult] = []
            calibrations: list[IterationResult] = []
            if promotable and not self.controller.state.stopped:
                best_proposal, best_iteration = promotable[0]
                selected_for_promotion = promotable[:promotion_width]
                if calibration_due:
                    calibration_candidates = promotable[promotion_width:]
                    required_calibration_seconds = 600.0 * len(calibration_candidates)
                    if (
                        len(promotable) == 3
                        and (
                            not calibration_candidates
                            or self.controller.ensure_wall_clock_capacity(
                                required_calibration_seconds
                            )
                        )
                    ):
                        for calibration_proposal, _iteration in calibration_candidates:
                            calibration = self._calibrate(direction, calibration_proposal)
                            if calibration is not None:
                                calibrations.append(calibration)
                    else:
                        self.logger.log_action(
                            "screen_full_calibration_incomplete",
                            details={
                                "cycle": self.controller.state.completed_cycles + 1,
                                "rankable_screens": len(promotable),
                                "required_rankable_screens": 3,
                            },
                        )
                for selected_proposal, selected_iteration in selected_for_promotion:
                    promoted_result = self._promote(direction, selected_proposal)
                    if promoted_result is not None:
                        promotions.append(promoted_result)
                        if promoted is None:
                            promoted = promoted_result
                    self.logger.log_action(
                        "screen_candidate_selected_for_promotion",
                        experiment_id=selected_proposal.experiment_id,
                        details={
                            "screen_primary": (selected_iteration.metrics or {})["primary"],
                            "screen_candidates": len(promotable),
                            "promotion_width": promotion_width,
                            "rule": "rankable screens only; every selected candidate is judged at full fidelity",
                        },
                    )
                    if self.controller.state.stopped:
                        break
                if calibration_due:
                    self._record_calibration(
                        self.controller.state.completed_cycles + 1,
                        promotable,
                        promotions,
                        calibrations,
                    )
            results.append(
                CycleResult(
                    direction,
                    tuple(iteration for _proposal, iteration in screened),
                    promoted,
                    tuple(calibrations),
                    tuple(promotions),
                )
            )
            self.controller.checkpoint_active_runtime()
            self.controller.complete_cycle()
            cycle_records = self.logger.store.read_iterations()
            preferred_id = promoted.experiment_id if promoted is not None else None
            cycle_record = next(
                (
                    record
                    for record in reversed(cycle_records)
                    if preferred_id is not None
                    and record.get("experiment_id") == preferred_id
                ),
                None,
            )
            if cycle_record is None:
                screen_ids = {result.experiment_id for _, result in screened}
                candidates = [
                    record
                    for record in cycle_records
                    if record.get("experiment_id") in screen_ids
                ]
                if candidates:
                    cycle_record = max(
                        candidates,
                        key=lambda record: float(
                            (record.get("metrics") or {}).get("primary", -math.inf)
                        ),
                    )
                else:
                    last_proposal, last_iteration = screened[-1]
                    cycle_record = {
                        "experiment_id": last_iteration.experiment_id,
                        "hypothesis": last_proposal.hypothesis,
                        "changed_factors": list(last_proposal.changed_factors),
                        "config": dict(last_proposal.config),
                        "metrics": dict(last_iteration.metrics or {}),
                        "decision": last_iteration.decision,
                        "runner_metadata": {},
                    }
            resolutions = (
                self.logger.store.read_root_json("promotion_resolutions.json") or {}
            )
            resolution = resolutions.get(str(cycle_record.get("experiment_id")))
            certificate = (
                resolution.get("certificate")
                if isinstance(resolution, Mapping)
                else None
            )
            seed_status = (
                f"confirmed={certificate.get('confirmed')}; wins={certificate.get('wins')}/3"
                if isinstance(certificate, Mapping)
                else "not confirmed"
            )
            summary_record = dict(cycle_record)
            if isinstance(resolution, Mapping):
                summary_record["decision"] = resolution.get(
                    "decision", summary_record.get("decision")
                )
                if isinstance(certificate, Mapping):
                    summary_record["delta_primary"] = certificate.get(
                        "mean_delta", summary_record.get("delta_primary")
                    )
            baseline_scores = (
                [
                    float(value)
                    for value in certificate.get("comparator_scores", [])
                ]
                if isinstance(certificate, Mapping)
                and certificate.get("comparator_experiment_id") == "baseline"
                else []
            )
            baseline_primary = (
                sum(baseline_scores) / len(baseline_scores)
                if baseline_scores
                else FROZEN_CALIBRATED_BASELINE_PRIMARY
            )
            lesson = (
                "Comparable seed confirmation accepted the hypothesis."
                if summary_record.get("decision") == "accepted"
                else "The result did not clear every applicable promotion gate."
            )
            next_hypothesis = (
                f"A bounded {direction.direction_id} refinement can improve held-user "
                "complementarity without increasing seed variance."
                if summary_record.get("decision") == "accepted"
                else "A materially different approved direction may add decorrelated "
                "ranking errors after this configuration failed its gate."
            )
            print(
                render_cycle_summary(
                    cycle=self.controller.state.completed_cycles,
                    hypothesis=direction.hypothesis,
                    record=summary_record,
                    state=self.controller.state.as_dict(),
                    contract=self.controller.contract,
                    seed_status=seed_status,
                    lesson=lesson,
                    next_hypothesis=next_hypothesis,
                    baseline_primary=baseline_primary,
                )
            )
            if (
                screened
                and all(
                    iteration.decision == "critic_rejected"
                    for _proposal, iteration in screened
                )
                and self.controller.state.planning_rejections >= 3
            ):
                self.controller.stop_search_exhausted(
                    "three proposal-critic rejections produced no executable trial"
                )
                break
        return results
    def _promote(
        self,
        direction: ResearchDirection,
        proposal,
    ) -> IterationResult | None:
        history = self.logger.store.read_iterations()
        if not self.controller.ensure_wall_clock_capacity(4200.0):
            return None
        experiment_id = self.search._next_experiment_id(
            history,
            self.logger.store.run_experiment_ids(),
        )
        promoted = self.fidelity.promote(proposal, direction, experiment_id=experiment_id)
        promoted = replace(
            promoted,
            comparison_incumbent_id=self.controller.state.current_best_experiment_id,
        )
        critic_result = self.critic.review(promoted, history)
        self.logger.log_action(
            "promotion_critic_reviewed",
            experiment_id=promoted.experiment_id,
            details={"approved": critic_result.approved, "reasons": list(critic_result.reasons)},
        )
        if not critic_result.approved:
            self.controller.record_critic_rejection(
                promoted,
                reasons=critic_result.reasons,
            )
            return None
        iteration = self.controller.run_iteration(promoted, self.candidate)
        if iteration.decision != "pending_confirmation":
            return iteration
        if self.promotion_confirmer is None:
            raise RuntimeError("production promotion requires a matched-seed confirmer")
        record = next(
            item
            for item in reversed(self.logger.store.read_iterations())
            if item.get("experiment_id") == promoted.experiment_id
        )
        certificate = dict(self.promotion_confirmer(record))
        return self.controller.resolve_seed_confirmation(
            promoted.experiment_id,
            certificate,
        )

    def _calibrate(
        self,
        direction: ResearchDirection,
        proposal,
    ) -> IterationResult | None:
        """Run one non-promoting full-fidelity point for screen-rank calibration."""
        if self.controller.state.stopped:
            return None
        history = self.logger.store.read_iterations()
        experiment_id = self.search._next_experiment_id(
            history,
            self.logger.store.run_experiment_ids(),
        )
        full = self.fidelity.promote(proposal, direction, experiment_id=experiment_id)
        full = replace(
            full,
            config={**dict(full.config), "calibration_only": True},
            search_strategy="screen_full_calibration",
        )
        critic_result = self.critic.review(full, history)
        self.logger.log_action(
            "calibration_critic_reviewed",
            experiment_id=full.experiment_id,
            details={
                "approved": critic_result.approved,
                "reasons": list(critic_result.reasons),
            },
        )
        if not critic_result.approved:
            self.controller.record_critic_rejection(full, reasons=critic_result.reasons)
            return None
        return self.controller.run_iteration(full, self.candidate)

    def _record_calibration(
        self,
        cycle_number: int,
        screened,
        promotions: list[IterationResult],
        calibrations: list[IterationResult],
    ) -> None:
        full_by_parent = {
            str(
                next(
                    (
                        record.get("parent_experiment_id")
                        for record in reversed(self.logger.store.read_iterations())
                        if record.get("experiment_id") == result.experiment_id
                    ),
                    "",
                )
            ): result
            for result in calibrations
        }
        for promoted in promotions:
            promoted_record = next(
                (
                    record
                    for record in reversed(self.logger.store.read_iterations())
                    if record.get("experiment_id") == promoted.experiment_id
                ),
                {},
            )
            full_by_parent[str(promoted_record.get("parent_experiment_id") or "")] = promoted
        pairs = [
            {
                "screen_experiment_id": proposal.experiment_id,
                "screen_primary": float((screen.metrics or {})["primary"]),
                "full_experiment_id": full_by_parent[proposal.experiment_id].experiment_id,
                "full_primary": float(
                    (full_by_parent[proposal.experiment_id].metrics or {})["primary"]
                ),
            }
            for proposal, screen in screened
            if proposal.experiment_id in full_by_parent
            and full_by_parent[proposal.experiment_id].metrics
            and full_by_parent[proposal.experiment_id].decision
            in {"accepted", "rejected", "inconclusive", "calibrated"}
        ]
        correlation = (
            _spearman_rank_correlation(
                [pair["screen_primary"] for pair in pairs],
                [pair["full_primary"] for pair in pairs],
            )
            if len(pairs) == 3
            else None
        )
        evidence = {
            "cycle": cycle_number,
            "pairs": pairs,
            "spearman_rank_correlation": correlation,
            "complete": len(pairs) == 3 and correlation is not None,
            "screen_ranking_reliable": correlation is not None and correlation >= 0.5,
        }
        records = self.logger.store.read_root_json("screen_full_calibrations.json") or {}
        records[str(cycle_number)] = evidence
        self.logger.store.write_root_json("screen_full_calibrations.json", records)
        self.logger.log_action("screen_full_calibration_recorded", details=evidence)
        self.controller.set_screen_promotion_width(
            1 if evidence["screen_ranking_reliable"] else 2,
            rank_correlation=correlation,
        )

    def _resume_pending_confirmations(self) -> None:
        """Finish a crash-interrupted promotion before planning new work."""
        resolutions = self.logger.store.read_root_json("promotion_resolutions.json") or {}
        pending = [
            record
            for record in self.logger.store.read_iterations()
            if record.get("decision") == "pending_confirmation"
            and record.get("experiment_id") not in resolutions
        ]
        if not pending:
            return
        if self.promotion_confirmer is None:
            raise RuntimeError("pending production promotion requires a matched-seed confirmer")
        for record in pending:
            experiment_id = str(record["experiment_id"])
            self.logger.log_action(
                "pending_seed_confirmation_resumed",
                experiment_id=experiment_id,
            )
            certificate = dict(self.promotion_confirmer(record))
            self.controller.resolve_seed_confirmation(experiment_id, certificate)


def _resolved_history(
    history: Sequence[Mapping[str, Any]],
    resolutions: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Overlay immutable promotion resolutions for allocation-only decisions."""
    resolved: list[dict[str, Any]] = []
    for item in history:
        record = dict(item)
        resolution = resolutions.get(str(record.get("experiment_id")))
        if isinstance(resolution, Mapping):
            record["decision"] = resolution.get("decision", record.get("decision"))
            certificate = resolution.get("certificate")
            if isinstance(certificate, Mapping):
                record["delta_primary"] = certificate.get(
                    "mean_delta", record.get("delta_primary")
                )
        resolved.append(record)
    return resolved


def _spearman_rank_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None

    def ranks(values: list[float]) -> list[float]:
        result = [0.0] * len(values)
        ordered = sorted(range(len(values)), key=values.__getitem__)
        index = 0
        while index < len(ordered):
            end = index + 1
            while end < len(ordered) and values[ordered[end]] == values[ordered[index]]:
                end += 1
            average_rank = (index + 1 + end) / 2
            for position in ordered[index:end]:
                result[position] = average_rank
            index = end
        return result

    x_ranks = ranks(xs)
    y_ranks = ranks(ys)
    x_mean = sum(x_ranks) / len(x_ranks)
    y_mean = sum(y_ranks) / len(y_ranks)
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_ranks, y_ranks)
    )
    x_scale = sum((value - x_mean) ** 2 for value in x_ranks)
    y_scale = sum((value - y_mean) ** 2 for value in y_ranks)
    denominator = math.sqrt(x_scale * y_scale)
    return None if denominator == 0 else numerator / denominator
