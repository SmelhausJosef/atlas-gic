from __future__ import annotations

from datetime import datetime, timezone
import inspect
import logging
from typing import Any

from atlas.backtest.engine import BacktestEngine
from atlas.common.artifacts import ArtifactManager, build_run_id
from atlas.common.config import AppConfig
from atlas.common.models import StrategyContext
from atlas.data.loaders import load_market_data, rolling_test_slices, split_train_oos
from atlas.evaluation.composite_score import composite_score
from atlas.research.candidate_strategies import CandidateValidationResult, validate_candidate_module
from atlas.research.llm_mutator import LLMMutator, MutationCandidate
from atlas.research.promotion import PromotionDecision, evaluate_candidate
from atlas.strategies import get_strategy_class
from atlas.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(self, config: AppConfig):
        self.config = config
        self.artifacts = ArtifactManager(config)
        self.mutator = LLMMutator(config)

    def run(self, baseline_name: str) -> dict[str, Any]:
        frame, manifest, _ = load_market_data(self.config.data)
        experiment_id = build_run_id(f"research_{baseline_name}", manifest.sha256)
        validation_context = self._build_validation_context(frame)
        baseline_cls = get_strategy_class(baseline_name)
        baseline_source = inspect.getsource(baseline_cls)
        current_candidate_name: str | None = None
        current_family_name: str | None = None
        session_dir = self.artifacts.create_research_session(
            experiment_id,
            baseline_name=baseline_name,
            config=self.config,
            manifest=manifest.to_dict(),
        )
        logger.info(
            "research session %s started baseline=%s family_budget=%s tuning_iterations=%s use_llm=%s",
            experiment_id,
            baseline_name,
            self.config.research.family_budget,
            self.config.research.tuning_iterations_per_family,
            self.config.research.use_llm,
        )

        try:
            self._write_heartbeat(
                session_dir,
                status="running",
                stage="baseline",
                baseline_name=baseline_name,
                family_index=0,
                evaluated_iterations=0,
                best_candidate_name=baseline_name,
                best_overall_score=None,
                current_family=None,
                current_iteration=None,
                accepted_families=0,
                rejected_families=0,
                stop_reason=None,
            )
            logger.info("baseline evaluation started for %s", baseline_name)
            baseline = self._evaluate_strategy(frame, manifest, baseline_name, baseline_cls)
            logger.info(
                "baseline evaluation finished overall=%.4f oos=%.4f stressed=%.4f trades=%.0f drawdown=%.4f",
                baseline["overall_score"],
                baseline["oos_score"],
                baseline["stressed_score"],
                baseline["full"]["metrics"]["trade_count"],
                baseline["full"]["metrics"]["max_drawdown"],
            )
            self.artifacts.write_backtest_run(baseline["full"]["result"], self.config)

            best_candidate_name = baseline_name
            best_family_name = baseline_name
            best_summary = baseline
            leaderboard = [self._build_baseline_leaderboard_entry(baseline_name, baseline)]
            lineage: list[dict[str, Any]] = []
            family_results: list[dict[str, Any]] = []
            memory_records: list[dict[str, Any]] = []
            evaluated_iterations = 0
            accepted_family_count = 0
            rejected_family_count = 0

            self._persist_research_state(
                session_dir,
                leaderboard=leaderboard,
                lineage=lineage,
                summary=self._build_running_summary(
                    experiment_id=experiment_id,
                    session_dir=session_dir,
                    baseline_name=baseline_name,
                    baseline=baseline,
                    best_family_name=best_family_name,
                    best_candidate_name=best_candidate_name,
                    best_summary=best_summary,
                    evaluated_iterations=evaluated_iterations,
                    accepted_family_count=accepted_family_count,
                    rejected_family_count=rejected_family_count,
                    family_results=family_results,
                    memory_records=memory_records,
                    stop_reason=None,
                ),
            )

            initial_families = self._discover_initial_families(
                baseline_name=baseline_name,
                baseline_source=baseline_source,
                baseline_summary=baseline,
            )
            if not initial_families:
                stop_reason = "no_strategy_families_generated"
                final_summary = self._build_running_summary(
                    experiment_id=experiment_id,
                    session_dir=session_dir,
                    baseline_name=baseline_name,
                    baseline=baseline,
                    best_family_name=best_family_name,
                    best_candidate_name=best_candidate_name,
                    best_summary=best_summary,
                    evaluated_iterations=evaluated_iterations,
                    accepted_family_count=accepted_family_count,
                    rejected_family_count=rejected_family_count,
                    family_results=family_results,
                    memory_records=memory_records,
                    stop_reason=stop_reason,
                )
                self._persist_research_state(session_dir, leaderboard=leaderboard, lineage=lineage, summary=final_summary)
                self._write_heartbeat(
                    session_dir,
                    status="completed",
                    stage="completed",
                    baseline_name=baseline_name,
                    family_index=0,
                    evaluated_iterations=evaluated_iterations,
                    best_candidate_name=best_candidate_name,
                    best_overall_score=best_summary["overall_score"],
                    current_family=None,
                    current_iteration=None,
                    accepted_families=accepted_family_count,
                    rejected_families=rejected_family_count,
                    stop_reason=stop_reason,
                )
                return final_summary

            family_budget = self.config.research.family_budget
            stop_reason = "family_budget_exhausted"
            for family_index, family_seed in enumerate(initial_families[:family_budget], start=1):
                current_family_name = family_seed.family_name or family_seed.candidate_name
                logger.info(
                    "family %s/%s started family=%s source=%s seed_candidate=%s",
                    family_index,
                    family_budget,
                    current_family_name,
                    family_seed.source,
                    family_seed.candidate_name,
                )
                family_result, new_best_summary, new_best_family_name, new_best_candidate_name, family_evaluated_iterations, memory_record = self._run_strategy_family(
                    session_dir=session_dir,
                    frame=frame,
                    manifest=manifest,
                    validation_context=validation_context,
                    baseline_name=baseline_name,
                    baseline_summary=baseline,
                    family_index=family_index,
                    family_seed=family_seed,
                    leaderboard=leaderboard,
                    lineage=lineage,
                    global_evaluated_iterations=evaluated_iterations,
                    accepted_family_count=accepted_family_count,
                    rejected_family_count=rejected_family_count,
                )
                family_results.append(family_result)
                memory_records.append(memory_record)
                evaluated_iterations += family_evaluated_iterations
                if family_result["verdict"]["accepted"]:
                    accepted_family_count += 1
                else:
                    rejected_family_count += 1
                if new_best_summary is not None and new_best_summary["overall_score"] > best_summary["overall_score"]:
                    best_summary = new_best_summary
                    best_family_name = new_best_family_name
                    best_candidate_name = new_best_candidate_name

                self._persist_research_state(
                    session_dir,
                    leaderboard=leaderboard,
                    lineage=lineage,
                    summary=self._build_running_summary(
                        experiment_id=experiment_id,
                        session_dir=session_dir,
                        baseline_name=baseline_name,
                        baseline=baseline,
                        best_family_name=best_family_name,
                        best_candidate_name=best_candidate_name,
                        best_summary=best_summary,
                        evaluated_iterations=evaluated_iterations,
                        accepted_family_count=accepted_family_count,
                        rejected_family_count=rejected_family_count,
                        family_results=family_results,
                        memory_records=memory_records,
                        stop_reason=None,
                    ),
                )
                self._write_heartbeat(
                    session_dir,
                    status="running",
                    stage="family_completed",
                    baseline_name=baseline_name,
                    family_index=family_index,
                    evaluated_iterations=evaluated_iterations,
                    best_candidate_name=best_candidate_name,
                    best_overall_score=best_summary["overall_score"],
                    current_family=current_family_name,
                    current_iteration=family_result["iterations_run"],
                    accepted_families=accepted_family_count,
                    rejected_families=rejected_family_count,
                    stop_reason=None,
                )

            final_summary = self._build_running_summary(
                experiment_id=experiment_id,
                session_dir=session_dir,
                baseline_name=baseline_name,
                baseline=baseline,
                best_family_name=best_family_name,
                best_candidate_name=best_candidate_name,
                best_summary=best_summary,
                evaluated_iterations=evaluated_iterations,
                accepted_family_count=accepted_family_count,
                rejected_family_count=rejected_family_count,
                family_results=family_results,
                memory_records=memory_records,
                stop_reason=stop_reason,
            )
            self._persist_research_state(session_dir, leaderboard=leaderboard, lineage=lineage, summary=final_summary)
            self._write_heartbeat(
                session_dir,
                status="completed",
                stage="completed",
                baseline_name=baseline_name,
                family_index=min(family_budget, len(family_results)),
                evaluated_iterations=evaluated_iterations,
                best_candidate_name=best_candidate_name,
                best_overall_score=best_summary["overall_score"],
                current_family=None,
                current_iteration=None,
                accepted_families=accepted_family_count,
                rejected_families=rejected_family_count,
                stop_reason=stop_reason,
            )
            logger.info(
                "research session %s completed stop_reason=%s evaluated_iterations=%s accepted_families=%s rejected_families=%s best_family=%s best_candidate=%s best_overall=%.4f",
                experiment_id,
                stop_reason,
                evaluated_iterations,
                accepted_family_count,
                rejected_family_count,
                best_family_name,
                best_candidate_name,
                best_summary["overall_score"],
            )
            return final_summary
        except Exception as exc:
            logger.exception("research session %s failed", experiment_id)
            failure_summary = {
                "experiment_id": experiment_id,
                "artifact_dir": str(session_dir),
                "heartbeat_path": str(session_dir / "heartbeat.json"),
                "leaderboard_path": str(session_dir / "leaderboard.json"),
                "lineage_path": str(session_dir / "lineage.json"),
                "baseline": {"strategy": baseline_name},
                "best_family_name": baseline_name,
                "best_candidate_name": baseline_name,
                "best_overall_score": None,
                "evaluated_iterations": 0,
                "accepted_family_count": 0,
                "rejected_family_count": 0,
                "stop_reason": f"failed:{exc.__class__.__name__}",
                "error": str(exc),
                "failed_family_name": current_family_name,
                "failed_candidate_name": current_candidate_name,
                "families": [],
                "strategy_memory_records": [],
            }
            self._persist_research_state(session_dir, leaderboard=[], lineage=[], summary=failure_summary)
            self._write_heartbeat(
                session_dir,
                status="failed",
                stage="failed",
                baseline_name=baseline_name,
                family_index=0,
                evaluated_iterations=0,
                best_candidate_name=baseline_name,
                best_overall_score=None,
                current_family=current_family_name,
                current_iteration=None,
                accepted_families=0,
                rejected_families=0,
                stop_reason=f"failed:{exc.__class__.__name__}",
            )
            raise

    def _run_strategy_family(
        self,
        *,
        session_dir,
        frame,
        manifest,
        validation_context: StrategyContext,
        baseline_name: str,
        baseline_summary: dict[str, Any],
        family_index: int,
        family_seed: MutationCandidate,
        leaderboard: list[dict[str, Any]],
        lineage: list[dict[str, Any]],
        global_evaluated_iterations: int,
        accepted_family_count: int,
        rejected_family_count: int,
    ) -> tuple[dict[str, Any], dict[str, Any] | None, str, str, int, dict[str, Any]]:
        family_name = family_seed.family_name or family_seed.candidate_name
        family_dir = self.artifacts.create_family_bundle(
            session_dir,
            family_name=family_name,
            metadata={
                "family_name": family_name,
                "family_index": family_index,
                "seed_candidate_name": family_seed.candidate_name,
                "source": family_seed.source,
                "baseline_name": baseline_name,
            },
        )

        family_history: list[dict[str, Any]] = []
        family_best_summary: dict[str, Any] | None = None
        family_best_candidate: MutationCandidate | None = None
        family_best_source: str | None = None
        family_best_validation: CandidateValidationResult | None = None
        family_best_iteration_dir = None
        iterations: list[dict[str, Any]] = []
        evaluated_iterations = 0
        no_improvement_rounds = 0
        pending_candidate = family_seed
        stop_reason = "tuning_iterations_exhausted"

        for iteration_index in range(0, self.config.research.tuning_iterations_per_family + 1):
            if pending_candidate is None:
                stop_reason = "no_tuning_candidate_generated"
                break

            evaluated_iterations += 1
            logger.info(
                "family %s iteration=%s candidate=%s parent=%s source=%s",
                family_name,
                iteration_index,
                pending_candidate.candidate_name,
                pending_candidate.parent_candidate_name,
                pending_candidate.source,
            )
            self._write_heartbeat(
                session_dir,
                status="running",
                stage="evaluating_family_iteration",
                baseline_name=baseline_name,
                family_index=family_index,
                evaluated_iterations=global_evaluated_iterations + evaluated_iterations,
                best_candidate_name=family_best_candidate.candidate_name if family_best_candidate else baseline_name,
                best_overall_score=family_best_summary["overall_score"] if family_best_summary else baseline_summary["overall_score"],
                current_family=family_name,
                current_iteration=iteration_index,
                accepted_families=accepted_family_count,
                rejected_families=rejected_family_count,
                stop_reason=None,
            )

            iteration_dir = self.artifacts.create_family_iteration_bundle(
                family_dir,
                family_name=family_name,
                candidate_name=pending_candidate.candidate_name,
                iteration_index=iteration_index,
                metadata=self._build_iteration_metadata(pending_candidate, family_index, iteration_index),
                strategy_code=pending_candidate.strategy_code,
            )
            validation, strategy_cls = self._validate_candidate(
                candidate_spec=pending_candidate,
                candidate_dir=iteration_dir,
                validation_context=validation_context,
            )
            self.artifacts.write_candidate_validation(iteration_dir, validation.to_dict())

            candidate_summary: dict[str, Any] | None = None
            if validation.valid and strategy_cls is not None:
                strategy_name = pending_candidate.strategy_name or pending_candidate.candidate_name
                candidate_summary = self._evaluate_strategy(
                    frame,
                    manifest,
                    strategy_name,
                    strategy_cls,
                    candidate_label=pending_candidate.candidate_name,
                )

            iteration_improved = self._did_family_iteration_improve(family_best_summary, candidate_summary)
            if iteration_improved and candidate_summary is not None:
                family_best_summary = candidate_summary
                family_best_candidate = pending_candidate
                family_best_source = pending_candidate.strategy_code or inspect.getsource(strategy_cls)
                family_best_validation = validation
                family_best_iteration_dir = iteration_dir
                no_improvement_rounds = 0
            else:
                no_improvement_rounds += 1

            iteration_reasons = []
            if candidate_summary is None:
                iteration_reasons = ["candidate validation failed", *validation.issues]
            elif not iteration_improved:
                iteration_reasons = ["candidate did not improve family best"]

            iteration_record = self._build_iteration_result(
                family_name=family_name,
                family_index=family_index,
                iteration_index=iteration_index,
                candidate_spec=pending_candidate,
                candidate_summary=candidate_summary,
                validation=validation,
                candidate_dir=iteration_dir,
                improved_family_best=iteration_improved,
                reasons=iteration_reasons,
            )
            iterations.append(iteration_record)
            family_history.append(self._build_family_history_entry(iteration_record))
            lineage.append(
                {
                    "family_name": family_name,
                    "family_index": family_index,
                    "parent_candidate_name": pending_candidate.parent_candidate_name,
                    "candidate_name": pending_candidate.candidate_name,
                    "generation": family_index,
                    "iteration_index": iteration_index,
                    "source": pending_candidate.source,
                    "accepted": iteration_improved,
                    "validation_status": validation.status,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            leaderboard.append(
                self._build_candidate_leaderboard_entry(
                    family_name=family_name,
                    family_index=family_index,
                    iteration_index=iteration_index,
                    candidate_spec=pending_candidate,
                    candidate_summary=candidate_summary,
                    validation=validation,
                    candidate_dir=iteration_dir,
                    improved_family_best=iteration_improved,
                )
            )

            if iteration_index >= self.config.research.tuning_iterations_per_family:
                stop_reason = "tuning_iterations_exhausted"
                break
            if not self.config.research.use_llm:
                stop_reason = "llm_disabled_no_tuning"
                break
            if no_improvement_rounds >= self.config.research.max_family_stagnation_iterations:
                stop_reason = "family_early_stop_no_improvement"
                break

            tuning_parent_source = family_best_source or pending_candidate.strategy_code
            tuning_parent_name = (
                family_best_candidate.candidate_name
                if family_best_candidate is not None
                else pending_candidate.candidate_name
            )
            if tuning_parent_source is None:
                stop_reason = "missing_parent_strategy_source"
                break
            pending_candidate = self.mutator.tune_strategy_family(
                family_name=family_name,
                iteration_index=iteration_index + 1,
                parent_candidate_name=tuning_parent_name,
                parent_strategy_source=tuning_parent_source,
                baseline_summary=self._build_llm_summary(baseline_summary),
                family_history=family_history,
            )

        verdict = self._build_family_verdict(baseline_summary, family_best_summary)
        memory_record = self._write_family_memory(
            session_dir=session_dir,
            family_name=family_name,
            family_index=family_index,
            family_seed=family_seed,
            family_best_candidate=family_best_candidate,
            family_best_summary=family_best_summary,
            verdict=verdict,
            family_history=family_history,
        )
        family_summary = {
            "family_name": family_name,
            "family_index": family_index,
            "seed_candidate_name": family_seed.candidate_name,
            "source": family_seed.source,
            "iterations_run": len(iterations),
            "stop_reason": stop_reason,
            "best_candidate_name": family_best_candidate.candidate_name if family_best_candidate else None,
            "best_overall_score": family_best_summary["overall_score"] if family_best_summary else None,
            "best_oos_score": family_best_summary["oos_score"] if family_best_summary else None,
            "best_stressed_score": family_best_summary["stressed_score"] if family_best_summary else None,
            "best_validation_status": family_best_validation.status if family_best_validation else "failed",
            "best_strategy_source_path": str(family_best_iteration_dir / "strategy.py") if family_best_iteration_dir and family_best_candidate and family_best_candidate.strategy_code else None,
            "family_artifact_dir": str(family_dir),
            "verdict": verdict.to_dict(),
            "memory_record": memory_record,
            "iterations": iterations,
        }
        self.artifacts.write_family_verdict(family_dir, family_summary)

        best_summary = family_best_summary if verdict.accepted else None
        best_family_name = family_name if verdict.accepted else baseline_name
        best_candidate_name = family_best_candidate.candidate_name if verdict.accepted and family_best_candidate else baseline_name
        logger.info(
            "family %s completed accepted=%s best_candidate=%s best_overall=%s stop_reason=%s",
            family_name,
            verdict.accepted,
            family_summary["best_candidate_name"],
            family_summary["best_overall_score"],
            stop_reason,
        )
        return family_summary, best_summary, best_family_name, best_candidate_name, evaluated_iterations, memory_record

    def _discover_initial_families(
        self,
        *,
        baseline_name: str,
        baseline_source: str,
        baseline_summary: dict[str, Any],
    ) -> list[MutationCandidate]:
        family_specs = [
            MutationCandidate(
                candidate_name=name,
                family_name=name,
                strategy_name=name,
                source="config",
                notes="registered seed strategy family",
                parent_candidate_name=baseline_name,
                iteration_index=0,
            )
            for name in self.config.research.candidate_strategies
            if name != baseline_name
        ]
        llm_families = self.mutator.propose_strategy_families(
            baseline_name,
            baseline_source,
            self._build_llm_summary(baseline_summary),
            remaining_budget=max(0, self.config.research.family_budget - len(family_specs)),
        )
        family_specs.extend(llm_families)
        deduped: list[MutationCandidate] = []
        seen_family_names: set[str] = set()
        for spec in family_specs:
            family_name = spec.family_name or spec.candidate_name
            if family_name in seen_family_names:
                continue
            seen_family_names.add(family_name)
            deduped.append(spec)
        return deduped

    def _evaluate_strategy(
        self,
        frame,
        manifest,
        strategy_name: str,
        strategy_cls: type[BaseStrategy],
        candidate_label: str | None = None,
    ) -> dict[str, Any]:
        train_frame, oos_frame = split_train_oos(frame, self.config.research.oos_fraction)
        run_label = candidate_label or strategy_name
        full_result = self._run_backtest(frame, manifest, strategy_name, strategy_cls, self.config, run_label)
        train_result = self._run_backtest(train_frame, manifest, strategy_name, strategy_cls, self.config, run_label)
        oos_result = self._run_backtest(oos_frame, manifest, strategy_name, strategy_cls, self.config, run_label)

        stressed_config = self.config.model_copy(deep=True)
        stressed_config.costs.trading_fee_bps *= self.config.research.stressed_cost_multiplier
        stressed_config.costs.slippage_bps *= self.config.research.stressed_cost_multiplier
        stressed_config.costs.latency_penalty_bps *= self.config.research.stressed_cost_multiplier
        stressed_result = self._run_backtest(
            oos_frame,
            manifest,
            strategy_name,
            strategy_cls,
            stressed_config,
            f"{run_label}_stressed",
        )

        walk_forward_scores = []
        for window in rolling_test_slices(frame, self.config.research.walk_forward_splits):
            walk_forward_scores.append(
                composite_score(
                    self._run_backtest(
                        window,
                        manifest,
                        strategy_name,
                        strategy_cls,
                        self.config,
                        f"{run_label}_wf",
                    ).metrics
                )
            )

        overall_score = (
            composite_score(full_result.metrics) * 0.35
            + composite_score(oos_result.metrics) * 0.35
            + composite_score(stressed_result.metrics) * 0.15
            + (sum(walk_forward_scores) / len(walk_forward_scores)) * 0.15
        )

        return {
            "train": {"result": train_result, "metrics": train_result.metrics},
            "oos": {"result": oos_result, "metrics": oos_result.metrics},
            "full": {"result": full_result, "metrics": full_result.metrics},
            "stressed": {"result": stressed_result, "metrics": stressed_result.metrics},
            "walk_forward_scores": walk_forward_scores,
            "overall_score": overall_score,
            "oos_score": composite_score(oos_result.metrics),
            "stressed_score": composite_score(stressed_result.metrics),
        }

    def _run_backtest(
        self,
        frame,
        manifest,
        strategy_name: str,
        strategy_cls: type[BaseStrategy],
        config: AppConfig,
        run_label: str,
    ):
        strategy = strategy_cls(config.strategy)
        engine = BacktestEngine(config)
        return engine.run(frame, manifest, strategy, run_label)

    def _build_baseline_leaderboard_entry(self, baseline_name: str, baseline: dict[str, Any]) -> dict[str, Any]:
        return {
            "rank_candidate_name": baseline_name,
            "family_name": baseline_name,
            "family_index": 0,
            "iteration_index": 0,
            "strategy": baseline_name,
            "generation": 0,
            "parent_candidate_name": None,
            "source": "baseline",
            "class_name": None,
            "validation_status": "passed",
            "accepted": True,
            "overall_score": baseline["overall_score"],
            "oos_score": baseline["oos_score"],
            "stressed_score": baseline["stressed_score"],
            "trade_count": baseline["full"]["metrics"]["trade_count"],
            "max_drawdown": baseline["full"]["metrics"]["max_drawdown"],
            "net_return": baseline["full"]["metrics"]["net_return"],
            "notes": "baseline",
            "strategy_source_path": None,
            "candidate_artifact_dir": None,
        }

    def _build_iteration_result(
        self,
        *,
        family_name: str,
        family_index: int,
        iteration_index: int,
        candidate_spec: MutationCandidate,
        candidate_summary: dict[str, Any] | None,
        validation: CandidateValidationResult,
        candidate_dir,
        improved_family_best: bool,
        reasons: list[str],
    ) -> dict[str, Any]:
        full_metrics = candidate_summary["full"]["metrics"] if candidate_summary is not None else {}
        return {
            "family_name": family_name,
            "family_index": family_index,
            "iteration_index": iteration_index,
            "strategy": candidate_spec.strategy_name,
            "candidate_name": candidate_spec.candidate_name,
            "parent_candidate_name": candidate_spec.parent_candidate_name,
            "source": candidate_spec.source,
            "notes": candidate_spec.notes,
            "mutation_spec": candidate_spec.mutation_spec,
            "class_name": candidate_spec.class_name or validation.class_name,
            "validation": validation.to_dict(),
            "candidate_artifact_dir": str(candidate_dir),
            "strategy_source_path": str(candidate_dir / "strategy.py") if candidate_spec.strategy_code else None,
            "improved_family_best": improved_family_best,
            "reasons": reasons,
            "overall_score": candidate_summary["overall_score"] if candidate_summary is not None else None,
            "oos_score": candidate_summary["oos_score"] if candidate_summary is not None else None,
            "stressed_score": candidate_summary["stressed_score"] if candidate_summary is not None else None,
            "trade_count": full_metrics.get("trade_count"),
            "max_drawdown": full_metrics.get("max_drawdown"),
            "net_return": full_metrics.get("net_return"),
        }

    def _build_candidate_leaderboard_entry(
        self,
        *,
        family_name: str,
        family_index: int,
        iteration_index: int,
        candidate_spec: MutationCandidate,
        candidate_summary: dict[str, Any] | None,
        validation: CandidateValidationResult,
        candidate_dir,
        improved_family_best: bool,
    ) -> dict[str, Any]:
        full_metrics = candidate_summary["full"]["metrics"] if candidate_summary is not None else {}
        return {
            "rank_candidate_name": candidate_spec.candidate_name,
            "family_name": family_name,
            "family_index": family_index,
            "iteration_index": iteration_index,
            "strategy": candidate_spec.strategy_name or candidate_spec.candidate_name,
            "generation": family_index,
            "parent_candidate_name": candidate_spec.parent_candidate_name,
            "source": candidate_spec.source,
            "class_name": candidate_spec.class_name or validation.class_name,
            "validation_status": validation.status,
            "accepted": improved_family_best,
            "overall_score": candidate_summary["overall_score"] if candidate_summary is not None else None,
            "oos_score": candidate_summary["oos_score"] if candidate_summary is not None else None,
            "stressed_score": candidate_summary["stressed_score"] if candidate_summary is not None else None,
            "trade_count": full_metrics.get("trade_count"),
            "max_drawdown": full_metrics.get("max_drawdown"),
            "net_return": full_metrics.get("net_return"),
            "notes": candidate_spec.notes,
            "strategy_source_path": str(candidate_dir / "strategy.py") if candidate_spec.strategy_code else None,
            "candidate_artifact_dir": str(candidate_dir),
        }

    def _build_running_summary(
        self,
        *,
        experiment_id: str,
        session_dir,
        baseline_name: str,
        baseline: dict[str, Any],
        best_family_name: str,
        best_candidate_name: str,
        best_summary: dict[str, Any],
        evaluated_iterations: int,
        accepted_family_count: int,
        rejected_family_count: int,
        family_results: list[dict[str, Any]],
        memory_records: list[dict[str, Any]],
        stop_reason: str | None,
    ) -> dict[str, Any]:
        return {
            "experiment_id": experiment_id,
            "artifact_dir": str(session_dir),
            "heartbeat_path": str(session_dir / "heartbeat.json"),
            "leaderboard_path": str(session_dir / "leaderboard.json"),
            "lineage_path": str(session_dir / "lineage.json"),
            "baseline": {
                "strategy": baseline_name,
                "overall_score": baseline["overall_score"],
                "oos_score": baseline["oos_score"],
                "stressed_score": baseline["stressed_score"],
            },
            "best_family_name": best_family_name,
            "best_candidate_name": best_candidate_name,
            "best_overall_score": best_summary["overall_score"],
            "evaluated_iterations": evaluated_iterations,
            "family_budget": self.config.research.family_budget,
            "tuning_iterations_per_family": self.config.research.tuning_iterations_per_family,
            "accepted_family_count": accepted_family_count,
            "rejected_family_count": rejected_family_count,
            "stop_reason": stop_reason,
            "families": family_results,
            "strategy_memory_records": memory_records,
        }

    def _persist_research_state(
        self,
        run_dir,
        *,
        leaderboard: list[dict[str, Any]],
        lineage: list[dict[str, Any]],
        summary: dict[str, Any],
    ) -> None:
        self.artifacts.write_research_leaderboard(run_dir, leaderboard)
        self.artifacts.write_research_lineage(run_dir, lineage)
        self.artifacts.write_research_summary(run_dir, summary)

    def _write_heartbeat(
        self,
        run_dir,
        *,
        status: str,
        stage: str,
        baseline_name: str,
        family_index: int,
        evaluated_iterations: int,
        best_candidate_name: str,
        best_overall_score: float | None,
        current_family: str | None,
        current_iteration: int | None,
        accepted_families: int,
        rejected_families: int,
        stop_reason: str | None,
    ) -> None:
        self.artifacts.write_research_heartbeat(
            run_dir,
            {
                "status": status,
                "stage": stage,
                "baseline_name": baseline_name,
                "family_index": family_index,
                "family_budget": self.config.research.family_budget,
                "evaluated_iterations": evaluated_iterations,
                "tuning_iterations_per_family": self.config.research.tuning_iterations_per_family,
                "best_candidate_name": best_candidate_name,
                "best_overall_score": best_overall_score,
                "current_family": current_family,
                "current_iteration": current_iteration,
                "accepted_families": accepted_families,
                "rejected_families": rejected_families,
                "max_family_stagnation_iterations": self.config.research.max_family_stagnation_iterations,
                "stop_reason": stop_reason,
            },
        )

    def _build_iteration_metadata(
        self,
        candidate_spec: MutationCandidate,
        family_index: int,
        iteration_index: int,
    ) -> dict[str, Any]:
        return {
            "candidate_name": candidate_spec.candidate_name,
            "strategy_name": candidate_spec.strategy_name,
            "class_name": candidate_spec.class_name,
            "family_name": candidate_spec.family_name,
            "family_index": family_index,
            "iteration_index": iteration_index,
            "parent_candidate_name": candidate_spec.parent_candidate_name,
            "source": candidate_spec.source,
            "notes": candidate_spec.notes,
            "mutation_spec": candidate_spec.mutation_spec,
            "generated": candidate_spec.is_generated,
        }

    def _validate_candidate(
        self,
        *,
        candidate_spec: MutationCandidate,
        candidate_dir,
        validation_context: StrategyContext,
    ) -> tuple[CandidateValidationResult, type[BaseStrategy] | None]:
        if candidate_spec.strategy_code is None:
            strategy_cls = get_strategy_class(candidate_spec.strategy_name or candidate_spec.candidate_name)
            return (
                CandidateValidationResult(status="passed", class_name=strategy_cls.__name__, module_path=None),
                strategy_cls,
            )
        return validate_candidate_module(
            candidate_name=candidate_spec.candidate_name,
            module_path=candidate_dir / "strategy.py",
            validation_context=validation_context,
            strategy_config=self.config.strategy,
        )

    def _build_validation_context(self, frame) -> StrategyContext:
        sample_size = min(
            len(frame),
            max(
                self.config.strategy.lookback_bars,
                self.config.strategy.mean_window,
                self.config.strategy.std_window,
                self.config.strategy.vol_window,
                240,
            ),
        )
        sample = frame.iloc[:sample_size].copy()
        return StrategyContext(
            bars=sample,
            current_position=0.0,
            gross_exposure=0.0,
            equity=self.config.backtest.initial_equity,
            bars_since_entry=None,
            bars_since_exit=None,
            expected_bar_seconds=self.config.data.expected_bar_seconds,
        )

    def _did_family_iteration_improve(
        self,
        reference_summary: dict[str, Any] | None,
        candidate_summary: dict[str, Any] | None,
    ) -> bool:
        if candidate_summary is None:
            return False
        if reference_summary is None:
            return True
        epsilon = self.config.research.improvement_epsilon
        if candidate_summary["overall_score"] <= reference_summary["overall_score"] + epsilon:
            return False
        if candidate_summary["oos_score"] + epsilon < reference_summary["oos_score"]:
            return False
        if candidate_summary["stressed_score"] + epsilon < reference_summary["stressed_score"]:
            return False
        return True

    def _build_family_verdict(
        self,
        baseline_summary: dict[str, Any],
        family_best_summary: dict[str, Any] | None,
    ) -> PromotionDecision:
        if family_best_summary is None:
            return PromotionDecision(accepted=False, reasons=["family never produced a valid backtest candidate"])
        return evaluate_candidate(
            baseline=baseline_summary,
            candidate=family_best_summary,
            minimum_trade_count=self.config.research.minimum_trade_count,
            max_drawdown_pct=self.config.research.max_drawdown_pct,
            improvement_epsilon=self.config.research.improvement_epsilon,
        )

    def _build_family_history_entry(self, iteration_record: dict[str, Any]) -> dict[str, Any]:
        return {
            "candidate_name": iteration_record["candidate_name"],
            "iteration_index": iteration_record["iteration_index"],
            "validation_status": iteration_record["validation"]["status"],
            "improved_family_best": iteration_record["improved_family_best"],
            "overall_score": iteration_record["overall_score"],
            "oos_score": iteration_record["oos_score"],
            "stressed_score": iteration_record["stressed_score"],
            "trade_count": iteration_record["trade_count"],
            "max_drawdown": iteration_record["max_drawdown"],
            "net_return": iteration_record["net_return"],
            "reasons": iteration_record["reasons"],
            "mutation_spec": iteration_record["mutation_spec"],
        }

    def _write_family_memory(
        self,
        *,
        session_dir,
        family_name: str,
        family_index: int,
        family_seed: MutationCandidate,
        family_best_candidate: MutationCandidate | None,
        family_best_summary: dict[str, Any] | None,
        verdict: PromotionDecision,
        family_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        category = "accepted" if verdict.accepted else "rejected"
        payload = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "session_dir": str(session_dir),
            "family_name": family_name,
            "family_index": family_index,
            "seed_candidate_name": family_seed.candidate_name,
            "best_candidate_name": family_best_candidate.candidate_name if family_best_candidate else None,
            "best_overall_score": family_best_summary["overall_score"] if family_best_summary else None,
            "best_oos_score": family_best_summary["oos_score"] if family_best_summary else None,
            "best_stressed_score": family_best_summary["stressed_score"] if family_best_summary else None,
            "verdict": verdict.to_dict(),
            "history": family_history,
        }
        memory_path = self.artifacts.append_strategy_memory(category, payload)
        return {"category": category, "path": str(memory_path), "family_name": family_name}

    def _build_llm_summary(self, summary: dict[str, Any]) -> dict[str, Any]:
        full_metrics = summary["full"]["metrics"]
        oos_metrics = summary["oos"]["metrics"]
        stressed_metrics = summary["stressed"]["metrics"]
        return {
            "overall_score": summary["overall_score"],
            "oos_score": summary["oos_score"],
            "stressed_score": summary["stressed_score"],
            "full_metrics": {
                "trade_count": full_metrics["trade_count"],
                "max_drawdown": full_metrics["max_drawdown"],
                "net_return": full_metrics["net_return"],
                "annualized_sharpe": full_metrics["annualized_sharpe"],
            },
            "oos_metrics": {
                "trade_count": oos_metrics["trade_count"],
                "max_drawdown": oos_metrics["max_drawdown"],
                "net_return": oos_metrics["net_return"],
            },
            "stressed_metrics": {
                "trade_count": stressed_metrics["trade_count"],
                "max_drawdown": stressed_metrics["max_drawdown"],
                "net_return": stressed_metrics["net_return"],
            },
            "walk_forward_scores": summary["walk_forward_scores"],
        }
