"""Candidate ranking and pre-filtering for the optimizer.

Filters out entities that cannot improve coverage and ranks remaining
candidates by weighted contribution (coverage + column metrics).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from .config import OptimizerConfig
from .scoring import miles_to_radians, normalize_to_100


@dataclass
class UncoveredData:
    """Pre-computed data about uncovered members from coverage results."""

    member_indices: np.ndarray
    member_coords: np.ndarray
    uncovered_specialties: set[str]
    uncovered_county_specs: set[tuple[str, str, str]]


def _extract_uncovered(members: pd.DataFrame, coverage: list[dict]) -> UncoveredData:
    """Extract uncovered member data from coverage results.

    Members are considered uncovered if they belong to a (county, specialty)
    pair with coverage < 100%.
    """
    uncovered_county_specs: set[tuple[str, str, str]] = {
        (r["state"], r["county"], r["specialty"])
        for r in coverage
        if r["coverage_percentage"] < 100.0
    }

    if not uncovered_county_specs:
        return UncoveredData(
            member_indices=np.array([], dtype=int),
            member_coords=np.empty((0, 2)),
            uncovered_specialties=set(),
            uncovered_county_specs=set(),
        )

    county_groups = members.groupby(["state", "county"], sort=False).groups
    uncovered_indices: list[int] = []
    for state, county, _spec in uncovered_county_specs:
        key = (state.lower(), county.lower())
        idx = county_groups.get(key)
        if idx is not None:
            uncovered_indices.extend(idx)

    unique_indices = np.unique(uncovered_indices)
    member_coords = members.loc[unique_indices, ["lat", "lon"]].values * (np.pi / 180.0)

    uncovered_specialties = {spec for _, _, spec in uncovered_county_specs}

    return UncoveredData(
        member_indices=unique_indices,
        member_coords=member_coords,
        uncovered_specialties=uncovered_specialties,
        uncovered_county_specs=uncovered_county_specs,
    )


class CandidateRanker:
    """Filters and ranks candidate entities for the optimizer.

    Uses a BallTree on uncovered members to estimate how many each entity
    could cover, combined with weighted column metrics for ranking.
    """

    def __init__(
        self,
        members: pd.DataFrame,
        thresholds: dict,
        entity_map: dict[str, pd.DataFrame],
        config: OptimizerConfig,
        pool_stats: dict[str, dict[str, float]],
    ) -> None:
        self.members = members
        self.thresholds = thresholds
        self.entity_map = entity_map
        self.config = config
        self.pool_stats = pool_stats
        self.weights = config.metric_weights
        self.entity_specialties: dict[str, set[str]] = {}

        for entity, df in entity_map.items():
            self.entity_specialties[entity] = set(df["specialty"].unique())

    def _coverage_count(
        self,
        entity_df: pd.DataFrame,
        tree: BallTree,
        uncovered: UncoveredData,
    ) -> int:
        """Count how many uncovered members an entity could cover."""
        if tree.data.shape[0] == 0:
            return 0

        provider_specialty = entity_df["specialty"].astype(str).str.lower().values
        provider_pts = entity_df[["lat", "lon"]].values * (np.pi / 180.0)

        all_thresholds = self._get_max_thresholds()
        reachable: set[int] = set()

        for spec in np.unique(provider_specialty):
            spec_mask = provider_specialty == spec
            spec_pts = provider_pts[spec_mask]
            radius = all_thresholds.get(spec)
            if radius is None:
                continue
            indices = tree.query_radius(spec_pts, r=radius)
            for idx in indices:
                for local_idx in idx:
                    global_idx = uncovered.member_indices[local_idx]
                    reachable.add(int(global_idx))

        return len(reachable)

    def _get_max_thresholds(self) -> dict[str, float]:
        """Get max threshold distance in radians per specialty."""
        thresholds: dict[str, float] = {}
        for state_data in self.thresholds.values():
            for county_data in state_data.values():
                for specialty, distance in county_data.items():
                    spec_lower = specialty.lower()
                    rad = miles_to_radians(distance)
                    if spec_lower not in thresholds or rad > thresholds[spec_lower]:
                        thresholds[spec_lower] = rad
        return thresholds

    def _rank_entity(self, entity: str, entity_df: pd.DataFrame, tree: BallTree, uncovered: UncoveredData) -> float:
        """Compute rank score for a single entity."""
        cov_count = self._coverage_count(entity_df, tree, uncovered)
        if cov_count == 0:
            return 0.0

        cov_pct = cov_count / len(uncovered.member_indices) * 100.0

        metric_score = 0.0
        total_weight = sum(self.weights.values())
        for col, w in self.weights.items():
            if col in entity_df.columns:
                col_mean = entity_df[col].mean()
                norm = normalize_to_100(col_mean, self.pool_stats[col]["min"], self.pool_stats[col]["max"])
                metric_score += w * norm

        adequacy_w = 1.0 - total_weight
        return adequacy_w * cov_pct + metric_score

    def get_relevant_candidates(
        self,
        outside_entities: set[str],
        coverage: list[dict],
    ) -> list[str]:
        """Filter and rank candidates.

        Returns entities sorted by rank score descending.
        Entities with score 0 (cannot improve coverage) are excluded.
        """
        uncovered = _extract_uncovered(self.members, coverage)

        if len(uncovered.member_indices) == 0:
            return []

        tree = BallTree(uncovered.member_coords, leaf_size=40, metric="haversine")

        scored: list[tuple[str, float]] = []
        for entity in outside_entities:
            if not self.entity_specialties.get(entity, set()) & uncovered.uncovered_specialties:
                continue

            entity_df = self.entity_map[entity]
            rank_score = self._rank_entity(entity, entity_df, tree, uncovered)
            if rank_score > 0:
                scored.append((entity, rank_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored]
