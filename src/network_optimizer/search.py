"""Two-phase local search optimizer.

Phase 1: Greedy additions — add entities one at a time, picking the best scorer.
Phase 2: Swap refinement — swap one network entity for one outside entity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .config import OptimizerConfig
from .scoring import adequacy_score


@dataclass
class SearchResult:
    """Result from the optimization run."""
    network: pd.DataFrame
    score: float
    phases: list[dict] = field(default_factory=list)
    elapsed: float = 0.0
    entities_added: list[str] = field(default_factory=list)
    entities_swapped: list[tuple[str, str]] = field(default_factory=list)
    network_entities: set[str] = field(default_factory=set)


class NetworkOptimizer:
    """Two-phase local search for provider network optimization."""

    def __init__(
        self,
        pool: pd.DataFrame,
        members: pd.DataFrame,
        thresholds: dict,
        initial_network: pd.DataFrame,
        config: OptimizerConfig | None = None,
        objective: Callable[[pd.DataFrame], float] | None = None,
    ):
        self.pool = pool
        self.members = members
        self.thresholds = thresholds
        self.config = config or OptimizerConfig()
        self._objective = objective

        # Current network state (entity names)
        self.network = initial_network.copy()
        self.network_entities = set(self.network["entity"].dropna().str.lower().unique())
        all_entities = set(pool["entity"].dropna().str.lower().unique())
        self.outside_entities = all_entities - self.network_entities

        # Pre-compute entity -> provider mapping
        self.entity_map = {}
        for entity, group in pool.dropna(subset=["entity"]).groupby("entity"):
            self.entity_map[entity.lower()] = group

        # Scoring cache
        self._last_score: float | None = None
        self._last_network_hash: int | None = None
        self.phases: list[dict] = []

    def _get_network_df(self) -> pd.DataFrame:
        """Reconstruct network DataFrame from current entity set."""
        entities = sorted(self.network_entities)
        if not entities:
            return self.pool.iloc[:0].copy()
        return self.pool[self.pool["entity"].dropna().str.lower().isin(entities)].reset_index(drop=True)

    def _score(self) -> float:
        """Score current network, with caching."""
        network_df = self._get_network_df()
        if self._objective is not None:
            return self._objective(network_df)
        return adequacy_score(self.members, self.thresholds, network_df)

    def _add_entity(self, entity: str) -> None:
        """Add an entity to the network."""
        self.network_entities.add(entity.lower())
        self.outside_entities.discard(entity.lower())

    def _remove_entity(self, entity: str) -> None:
        """Remove an entity from the network."""
        self.network_entities.discard(entity.lower())
        self.outside_entities.add(entity.lower())

    def _log(self, message: str) -> None:
        """Log message based on verbosity level."""
        if self.config.verbosity >= 1:
            print(message)

    def phase1_additions(self) -> list[str]:
        """Phase 1: Greedy additions.

        Iteratively add the entity that gives the biggest score improvement.
        Stops when no entity improves the score or max_rounds is reached.
        """
        self._log("\n--- Phase 1: Greedy Additions ---")
        added = []
        no_improve_count = 0
        start_time = time.time()
        initial_score = self._score()
        self._log(f"  Initial score: {initial_score:.2f}%")

        for round_num in range(self.config.max_rounds):
            # Check time budget
            if self.config.time_budget and (time.time() - start_time) > self.config.time_budget:
                self._log(f"  Time budget reached ({self.config.time_budget}s)")
                break

            best_entity = None
            best_score = initial_score
            best_improvement = 0.0

            for entity in sorted(self.outside_entities):
                self._add_entity(entity)
                score = self._score()
                improvement = score - initial_score
                self._remove_entity(entity)

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_score = score
                    best_entity = entity

            # Check convergence
            if self.config.convergence_threshold > 0:
                relative = best_improvement / max(initial_score, 0.001)
                if relative < self.config.convergence_threshold:
                    self._log(f"  Converged at round {round_num + 1} (relative improvement {relative:.4f} < {self.config.convergence_threshold})")
                    break

            if best_entity is None or best_improvement <= 0:
                no_improve_count += 1
                self._log(f"  Round {round_num + 1}: No improvement, {no_improve_count}/{self.config.patience} no-improve")
                if no_improve_count >= self.config.patience:
                    self._log(f"  Stopping after {no_improve_count} rounds without improvement")
                    break
                continue

            # Commit best entity
            self._add_entity(best_entity)
            added.append(best_entity)
            initial_score = best_score
            no_improve_count = 0

            if self.config.verbosity >= 2:
                self._log(f"  Round {round_num + 1}: Added '{best_entity}' → score {best_score:.2f}% (+{best_improvement:.2f}%)")

        self._log(f"  Phase 1 complete: {len(added)} entities added, score {initial_score:.2f}%")

        self.phases.append({
            "phase": 1,
            "entities_added": len(added),
            "final_score": initial_score,
            "elapsed": time.time() - start_time,
        })

        return added

    def phase2_swaps(self) -> list[tuple[str, str]]:
        """Phase 2: Swap refinement.

        Iteratively swap one network entity for one outside entity.
        Stops when no swap improves the score or max_rounds is reached.
        """
        if not self.config.enable_swaps:
            self._log("\n--- Phase 2: Swaps disabled ---")
            return []

        if not self.network_entities or not self.outside_entities:
            self._log("\n--- Phase 2: Skipped (no entities to swap) ---")
            return []

        self._log("\n--- Phase 2: Swap Refinement ---")
        swapped = []
        no_improve_count = 0
        start_time = time.time()
        initial_score = self._score()
        self._log(f"  Starting score: {initial_score:.2f}%")

        for round_num in range(self.config.max_rounds):
            # Check time budget
            if self.config.time_budget and (time.time() - start_time) > self.config.time_budget:
                self._log(f"  Time budget reached ({self.config.time_budget}s)")
                break

            best_swap = None
            best_score = initial_score
            best_improvement = 0.0

            for in_entity in sorted(self.network_entities):
                for out_entity in sorted(self.outside_entities):
                    # Try swap
                    self._remove_entity(in_entity)
                    self._add_entity(out_entity)
                    score = self._score()
                    improvement = score - initial_score
                    # Undo swap
                    self._add_entity(in_entity)
                    self._remove_entity(out_entity)

                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_score = score
                        best_swap = (in_entity, out_entity)

            # Check convergence
            if self.config.convergence_threshold > 0:
                relative = best_improvement / max(initial_score, 0.001)
                if relative < self.config.convergence_threshold:
                    self._log(f"  Converged at round {round_num + 1}")
                    break

            if best_swap is None or best_improvement <= 0:
                no_improve_count += 1
                self._log(f"  Round {round_num + 1}: No improvement, {no_improve_count}/{self.config.patience} no-improve")
                if no_improve_count >= self.config.patience:
                    self._log(f"  Stopping after {no_improve_count} rounds without improvement")
                    break
                continue

            # Commit best swap
            in_e, out_e = best_swap
            self._remove_entity(in_e)
            self._add_entity(out_e)
            swapped.append((in_e, out_e))
            initial_score = best_score
            no_improve_count = 0

            if self.config.verbosity >= 2:
                self._log(f"  Round {round_num + 1}: Swapped '{in_e}' → '{out_e}' → score {best_score:.2f}% (+{best_improvement:.2f}%)")

        self._log(f"  Phase 2 complete: {len(swapped)} swaps, score {initial_score:.2f}%")

        self.phases.append({
            "phase": 2,
            "swaps": len(swapped),
            "final_score": initial_score,
            "elapsed": time.time() - start_time,
        })

        return swapped

    def optimize(self) -> SearchResult:
        """Run the full optimization."""
        overall_start = time.time()
        entities_added = []
        entities_swapped = []

        if not self.config.swap_only:
            entities_added = self.phase1_additions()

        entities_swapped = self.phase2_swaps()

        final_network = self._get_network_df()
        final_score = self._score()
        elapsed = time.time() - overall_start

        self._log(f"\n=== Final Score: {final_score:.2f}% ===")
        self._log(f"  Entities in network: {len(self.network_entities)}")
        self._log(f"  Providers in network: {len(final_network)}")
        self._log(f"  Time: {elapsed:.1f}s")

        return SearchResult(
            network=final_network,
            score=final_score,
            elapsed=elapsed,
            entities_added=entities_added,
            entities_swapped=entities_swapped,
            network_entities=set(self.network_entities),
        )
