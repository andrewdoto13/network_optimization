"""Optimizer configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

# Required columns for provider pool (county is optional - BallTree handles geographic matching)
POOL_REQUIRED = frozenset({
    "id", "entity", "specialty", "lat", "lon", "state",
    "effectiveness", "efficiency",
})

MEMBER_REQUIRED = frozenset({"id", "state", "county", "lat", "lon"})


@dataclass
class OptimizerConfig:
    """Configuration for the network optimizer."""

    # Search control
    max_rounds: int = 100
    patience: int = 5
    time_budget: float | None = None
    convergence_threshold: float = 0.0  # Stop when improvement < this % of current score
    n_jobs: int = 1  # Parallel workers for candidate scoring (1 = sequential)

    # Search strategy
    search_mode: str = "first_improvement"  # "steepest" | "first_improvement"
    metric_weights: dict[str, float] = field(default_factory=dict)  # {"efficiency": 0.3}

    # Phase control
    enable_swaps: bool = False
    swap_only: bool = False  # Skip additions, only do swap refinement

    # Output
    verbosity: int = 1  # 0=silent, 1=summary, 2=per-round details

    # Coordinate bounds
    lat_min: float = -90.0
    lat_max: float = 90.0
    lon_min: float = -180.0
    lon_max: float = 180.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        if self.patience < 1:
            raise ValueError("patience must be >= 1")
        if self.time_budget is not None and self.time_budget <= 0:
            raise ValueError("time_budget must be > 0")
        if not 0 <= self.convergence_threshold <= 1:
            raise ValueError("convergence_threshold must be in [0, 1]")
        if self.verbosity not in (0, 1, 2):
            raise ValueError("verbosity must be 0, 1, or 2")
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be >= 1")
        if self.search_mode not in ("steepest", "first_improvement"):
            raise ValueError("search_mode must be 'steepest' or 'first_improvement'")
        if self.metric_weights:
            if any(v < 0 or v > 1 for v in self.metric_weights.values()):
                raise ValueError("metric_weights values must be in [0, 1]")
            if sum(self.metric_weights.values()) >= 1.0:
                raise ValueError("sum of metric_weights must be < 1.0 (adequacy gets remainder)")

    @classmethod
    def quick(cls) -> OptimizerConfig:
        """Quick test config: small rounds, no swaps."""
        return cls(max_rounds=10, patience=3, enable_swaps=False, verbosity=1)

    @classmethod
    def production(cls) -> OptimizerConfig:
        """Production config: full search with swaps."""
        return cls(
            max_rounds=500,
            patience=20,
            enable_swaps=True,
            time_budget=3600,
            convergence_threshold=0.001,
            verbosity=1,
        )
