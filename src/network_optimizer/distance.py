"""Distance computation with BallTree-based coverage queries."""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute haversine distance in miles between two points."""
    radius_earth = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return radius_earth * 2 * np.arcsin(np.sqrt(a))


def build_balltree(providers: pd.DataFrame) -> BallTree:
    """Build BallTree from provider coordinates.

    Args:
        providers: DataFrame with 'latitude' and 'longitude' columns.

    Returns:
        BallTree instance with radians coordinates.
    """
    coords = np.radians(providers[["latitude", "longitude"]].values)
    return BallTree(coords, metric="haversine")


def compute_access(
    providers: pd.DataFrame,
    members: pd.DataFrame,
    adequacy_reqs: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-member accessible provider counts per (county, specialty).

    Uses BallTree radius queries instead of cross-join for scalability.

    Args:
        providers: Provider DataFrame with specialty, county, latitude, longitude.
        members: Member DataFrame with county, latitude, longitude.
        adequacy_reqs: Requirements DataFrame with specialty, county, distance_req, provider_count.

    Returns:
        DataFrame with columns: member_id, county, specialty, accessible_provider_count.
    """
    results = []

    for _, req_row in adequacy_reqs.iterrows():
        specialty = req_row["specialty"]
        county = req_row["county"]
        distance_deg = req_row["distance_req"] / 3958.8  # miles to radians

        # Filter providers by specialty
        spec_providers = providers[providers["specialty"] == specialty]
        if spec_providers.empty:
            continue

        # Filter members by county
        county_members = members[members["county"] == county]
        if county_members.empty:
            continue

        # Build BallTree on specialty providers
        tree = build_balltree(spec_providers)

        # Query each member for providers within distance
        member_coords = np.radians(county_members[["latitude", "longitude"]].values)
        indices, _ = tree.query_radius(member_coords, r=distance_deg, return_distance=True)

        for i, member_id in enumerate(county_members["member_id"]):
            results.append(
                {
                    "member_id": member_id,
                    "county": county,
                    "specialty": specialty,
                    "accessible_provider_count": len(indices[i]),
                }
            )

    return pd.DataFrame(results)
