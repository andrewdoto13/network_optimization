"""Optimizer configuration."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OptimizerConfig:
    """Configuration for the network optimizer."""

    max_rounds: int = 100
    patience: int = 5
    time_budget: Optional[float] = None
    enable_swaps: bool = False

    # Default column names
    pool_columns: dict = field(
        default_factory=lambda: {
            "npi": "npi",
            "specialty": "specialty",
            "group_id": "group_id",
            "county": "county",
            "latitude": "latitude",
            "longitude": "longitude",
        }
    )

    member_columns: dict = field(
        default_factory=lambda: {
            "member_id": "member_id",
            "county": "county",
            "latitude": "latitude",
            "longitude": "longitude",
        }
    )

    adequacy_columns: dict = field(
        default_factory=lambda: {
            "specialty": "specialty",
            "county": "county",
            "provider_count": "provider_count",
            "distance_req": "distance_req",
            "min_access_pct": "min_access_pct",
            "min_providers": "min_providers",
        }
    )

    # Coordinate bounds
    lat_min: float = -90.0
    lat_max: float = 90.0
    lon_min: float = -180.0
    lon_max: float = 180.0
