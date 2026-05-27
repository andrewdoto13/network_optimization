"""Distance computation and BallTree coverage queries."""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


def build_balltree(df: pd.DataFrame) -> tuple[dict, dict]:
    """Build a BallTree per specialty from provider data.

    Returns:
        specialty_trees: dict mapping specialty -> BallTree
        specialty_indices: dict mapping specialty -> list of provider indices
    """
    specialty_trees = {}
    specialty_indices = {}

    for specialty, group in df.groupby("specialty"):
        coords = group[["lat", "lon"]].values * (np.pi / 180.0)
        tree = BallTree(coords, leaf_size=40, metric="haversine")
        specialty_trees[specialty] = tree
        specialty_indices[specialty] = list(group.index)

    return specialty_trees, specialty_indices


def miles_to_radians(miles: float, earth_radius: float = 3958.8) -> float:
    """Convert miles to radians for haversine distance."""
    return miles / earth_radius


def compute_coverage(
    pool: pd.DataFrame,
    members: pd.DataFrame,
    thresholds: dict,
    network: pd.DataFrame,
) -> list[dict]:
    """Compute per-county-and-specialty member coverage.

    For each (state, county, specialty) threshold:
        - Filter network to matching specialty
        - Build BallTree for those providers
        - Query each member's location for providers within distance threshold
        - Count members with at least one provider in range

    Returns list of coverage result dicts matching agent format.
    """
    coverage_results = []

    for state_val, counties in thresholds.items():
        for county_val, specialties in counties.items():
            # Filter members to this county
            county_mask = members["county"].astype(str).str.lower() == county_val.lower()
            if "state" in members.columns:
                state_mask = members["state"].astype(str).str.lower() == state_val.lower()
                county_mask = county_mask & state_mask
            county_members = members[county_mask]

            if county_members.empty:
                for specialty, _threshold in specialties.items():
                    coverage_results.append({
                        "state": state_val,
                        "county": county_val,
                        "specialty": specialty,
                        "members_with_access": 0,
                        "total_members": 0,
                        "coverage_percentage": 0.0,
                    })
                continue

            group_pts = county_members[["lat", "lon"]].values * (np.pi / 180.0)

            for specialty, threshold in specialties.items():
                # Filter network to this specialty
                specialty_network = network[network["specialty"].str.lower() == specialty.lower()]

                if specialty_network.empty:
                    coverage_results.append({
                        "state": state_val,
                        "county": county_val,
                        "specialty": specialty,
                        "members_with_access": 0,
                        "total_members": len(county_members),
                        "coverage_percentage": 0.0,
                    })
                    continue

                # Build BallTree and query
                radius_rad = miles_to_radians(threshold)
                tree = BallTree(
                    specialty_network[["lat", "lon"]].values * (np.pi / 180.0),
                    leaf_size=40,
                    metric="haversine",
                )
                indices, _ = tree.query_radius(group_pts, r=radius_rad, return_distance=True)

                members_with_access = int(np.array([len(lst) > 0 for lst in indices]).sum())
                total_members = len(county_members)
                coverage_pct = round(members_with_access / total_members * 100, 2) if total_members > 0 else 0.0

                coverage_results.append({
                    "state": state_val,
                    "county": county_val,
                    "specialty": specialty,
                    "members_with_access": members_with_access,
                    "total_members": total_members,
                    "coverage_percentage": coverage_pct,
                })

    coverage_results.sort(key=lambda x: (x.get("state", ""), x["county"], x["specialty"]))
    return coverage_results


def compute_score(coverage_results: list[dict]) -> float:
    """Compute overall adequacy score from coverage results.

    Score = mean coverage percentage across all (county, specialty) thresholds.
    """
    if not coverage_results:
        return 0.0
    return sum(r["coverage_percentage"] for r in coverage_results) / len(coverage_results)
