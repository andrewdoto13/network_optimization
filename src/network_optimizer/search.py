"""Two-phase local search optimizer."""

import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from network_optimizer.adequacy import compute_adequacy
from network_optimizer.config import OptimizerConfig
from network_optimizer.distance import compute_access


@dataclass
class Move:
    """A single move in the search space."""

    move_type: str  # "add" or "swap"
    add_group_id: int
    remove_group_id: Optional[int] = None


@dataclass
class OptimizationResult:
    """Result of the optimization run."""

    best_network: pd.DataFrame
    final_score: float
    initial_score: float
    performance_history: list[float] = field(default_factory=list)
    moves: list["Move"] = field(default_factory=list)
    phase1_rounds: int = 0
    phase2_rounds: int = 0
    total_time: float = 0.0
    adequacy_detail: Optional[pd.DataFrame] = None


class NetworkOptimizer:
    """Two-phase local search: greedy additions then swap refinement."""

    def __init__(
        self,
        pool: pd.DataFrame,
        members: pd.DataFrame,
        adequacy_reqs: pd.DataFrame,
        network: Optional[pd.DataFrame] = None,
        config: Optional[OptimizerConfig] = None,
    ):
        self.pool = pool
        self.members = members
        self.adequacy_reqs = adequacy_reqs
        self.network = network if network is not None else pd.DataFrame(pool.columns)
        self.config = config or OptimizerConfig()

    def evaluate(self, network: pd.DataFrame) -> tuple:
        """Evaluate a network state. Returns (score, detail)."""
        access_df = compute_access(network, self.members, self.adequacy_reqs)
        return compute_adequacy(access_df, self.members, self.adequacy_reqs)

    def successors_additions(self, network: pd.DataFrame) -> list["Move"]:
        """Generate addition moves: one per group in the pool not in the network."""
        pool_groups = set(self.pool["group_id"])
        network_groups = set(network["group_id"])
        candidate_groups = pool_groups - network_groups
        return [Move("add", gid) for gid in candidate_groups]

    def successors_swaps(self, network: pd.DataFrame) -> list["Move"]:
        """Generate swap moves: one per (network_group, pool_group) pair."""
        pool_groups = set(self.pool["group_id"])
        network_groups = set(network["group_id"])
        candidate_groups = pool_groups - network_groups
        return [
            Move("swap", add_gid, remove_gid)
            for remove_gid in network_groups
            for add_gid in candidate_groups
        ]

    def apply_move(self, network: pd.DataFrame, move: Move) -> pd.DataFrame:
        """Apply a move and return the new network."""
        new_network = network.copy()
        if move.move_type == "add":
            group = self.pool[self.pool["group_id"] == move.add_group_id]
            new_network = pd.concat([new_network, group], ignore_index=True)
        elif move.move_type == "swap":
            new_network = new_network[new_network["group_id"] != move.remove_group_id]
            group = self.pool[self.pool["group_id"] == move.add_group_id]
            new_network = pd.concat([new_network, group], ignore_index=True)
        return new_network

    def _phase_additions(self, network: pd.DataFrame, history: list[float], moves: list["Move"]) -> pd.DataFrame:
        """Phase 1: Greedy additions until convergence."""
        score, _ = self.evaluate(network)
        no_improve = 0

        for round_num in range(1, self.config.max_rounds + 1):
            successors = self.successors_additions(network)
            if not successors:
                break

            best_move = None
            best_score = score

            for move in successors:
                candidate = self.apply_move(network, move)
                cand_score, _ = self.evaluate(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_move = move

            if best_move is None:
                no_improve += 1
                if no_improve >= self.config.patience:
                    break
                continue

            network = self.apply_move(network, best_move)
            history.append(best_score)
            moves.append(best_move)
            print(f"  Phase 1 Round {round_num}: +group {best_move.add_group_id} -> {best_score:.4f}")

        return network

    def _phase_swaps(self, network: pd.DataFrame, history: list[float], moves: list["Move"]) -> pd.DataFrame:
        """Phase 2: Swap refinement until convergence."""
        score, _ = self.evaluate(network)

        for round_num in range(1, self.config.max_rounds + 1):
            successors = self.successors_swaps(network)
            if not successors:
                break

            best_move = None
            best_score = score

            for move in successors:
                candidate = self.apply_move(network, move)
                cand_score, _ = self.evaluate(candidate)
                if cand_score > best_score:
                    best_score = cand_score
                    best_move = move

            if best_move is None:
                break

            network = self.apply_move(network, best_move)
            history.append(best_score)
            moves.append(best_move)
            print(f"  Phase 2 Round {round_num}: swap {best_move.remove_group_id} -> {best_move.add_group_id} -> {best_score:.4f}")

        return network

    def optimize(self) -> OptimizationResult:
        """Run the full two-phase optimization."""
        start_time = time.time()

        initial_score, _ = self.evaluate(self.network)
        print(f"Initial score: {initial_score:.4f}")
        print(f"Initial network: {len(self.network)} groups")

        history = [initial_score]
        moves = []

        # Phase 1: Additions
        print("\nPhase 1: Greedy additions")
        self.network = self._phase_additions(self.network, history, moves)
        phase1_rounds = len([m for m in moves if m.move_type == "add"])

        # Phase 2: Swaps
        phase2_rounds = 0
        if self.config.enable_swaps:
            print("\nPhase 2: Swap refinement")
            self.network = self._phase_swaps(self.network, history, moves)
            phase2_rounds = len([m for m in moves if m.move_type == "swap"])

        final_score, detail = self.evaluate(self.network)
        total_time = time.time() - start_time

        print(f"\nFinal score: {final_score:.4f}")
        print(f"Final network: {len(self.network)} groups")
        print(f"Moves: {len(moves)} ({phase1_rounds} additions, {phase2_rounds} swaps)")
        print(f"Time: {total_time:.2f}s")

        return OptimizationResult(
            best_network=self.network,
            final_score=final_score,
            initial_score=initial_score,
            performance_history=history,
            moves=moves,
            phase1_rounds=phase1_rounds,
            phase2_rounds=phase2_rounds,
            total_time=total_time,
            adequacy_detail=detail,
        )
