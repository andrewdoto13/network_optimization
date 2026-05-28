"""Scoring — coverage computation and adequacy scoring.

Provides building blocks for network adequacy evaluation and
supports custom objective functions via composition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


def miles_to_radians(miles: float, earth_radius: float = 3958.8) -> float:
    """Convert miles to radians for haversine distance."""
    return miles / earth_radius


def compute_coverage(
    members: pd.DataFrame,
    thresholds: dict,
    network: pd.DataFrame,
) -> list[dict]:
    """Compute per-county-and-specialty member coverage.

    Optimized approach:
        - Build per-specialty BallTree (N builds instead of N×C)
        - Per (county, specialty): query only county members
        - No post-filtering needed (tree is specialty-specific)

    Returns list of coverage result dicts matching agent format.
    """
    if network.empty:
        return [
            {
                "state": state, "county": county, "specialty": specialty,
                "members_with_access": 0, "total_members": 0,
                "coverage_percentage": 0.0,
            }
            for state, counties in thresholds.items()
            for county, specialties in counties.items()
            for specialty in specialties
        ]

    # Pre-compute per-(state, county) member index groups
    county_groups = members.groupby(["state", "county"], sort=False).groups
    member_pts = members[["lat", "lon"]].values * (np.pi / 180.0)
    provider_specialty = network["specialty"].astype(str).str.lower().values

    # Build per-specialty BallTrees
    provider_pts = network[["lat", "lon"]].values * (np.pi / 180.0)
    spec_trees: dict[str, BallTree] = {}
    for spec in np.unique(provider_specialty):
        spec_pts = provider_pts[provider_specialty == spec]
        spec_trees[spec] = BallTree(spec_pts, leaf_size=40, metric="haversine")

    # Per (county, specialty): query county members with specialty-specific tree
    coverage_results = []

    for state_val, counties in thresholds.items():
        for county_val, specialties in counties.items():
            state_key = state_val.lower()
            county_key = county_val.lower()
            county_key_pair = (state_key, county_key)
            member_idx = county_groups.get(county_key_pair)
            total_in_county = len(member_idx) if member_idx is not None else 0

            if total_in_county == 0:
                for specialty in specialties:
                    coverage_results.append({
                        "state": state_key, "county": county_key,
                        "specialty": specialty.lower(),
                        "members_with_access": 0, "total_members": 0,
                        "coverage_percentage": 0.0,
                    })
                continue

            county_pts = member_pts[member_idx]

            for specialty, threshold in specialties.items():
                spec_key = specialty.lower()
                spec_tree = spec_trees.get(spec_key)

                if spec_tree is None:
                    coverage_results.append({
                        "state": state_key, "county": county_key,
                        "specialty": spec_key,
                        "members_with_access": 0, "total_members": total_in_county,
                        "coverage_percentage": 0.0,
                    })
                    continue

                radius_rad = miles_to_radians(threshold)
                indices = spec_tree.query_radius(county_pts, r=radius_rad)
                members_with_access = int(np.array([len(idx) > 0 for idx in indices]).sum())
                coverage_pct = round(members_with_access / total_in_county * 100, 2)

                coverage_results.append({
                    "state": state_key, "county": county_key,
                    "specialty": spec_key,
                    "members_with_access": members_with_access,
                    "total_members": total_in_county,
                    "coverage_percentage": coverage_pct,
                })

    coverage_results.sort(key=lambda x: (x["state"], x["county"], x["specialty"]))
    return coverage_results


def adequacy_score(
    members: pd.DataFrame,
    thresholds: dict,
    network: pd.DataFrame,
) -> float:
    """Compute overall adequacy score for a given network.

    Score = mean coverage percentage across all (county, specialty) thresholds.
    """
    coverage_results = compute_coverage(members, thresholds, network)
    if not coverage_results:
        return 0.0
    return sum(r["coverage_percentage"] for r in coverage_results) / len(coverage_results)
