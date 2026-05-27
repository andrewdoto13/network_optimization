"""Adequacy scoring and objective functions."""

import pandas as pd


def compute_adequacy(
    access_df: pd.DataFrame,
    members: pd.DataFrame,
    adequacy_reqs: pd.DataFrame,
) -> tuple:
    """Compute network adequacy score.

    For each (county, specialty) combination, calculates the percentage of members
    who have access to at least `provider_count` providers. Returns the weighted
    average across all combinations.

    Args:
        access_df: Output of compute_access() with member_id, county, specialty,
                   accessible_provider_count.
        members: Full members DataFrame.
        adequacy_reqs: Requirements with specialty, county, provider_count,
                       min_access_pct, min_providers.

    Returns:
        Tuple of (overall_score: float, detail: DataFrame).
        Detail has per (county, specialty) breakdown.
    """
    if access_df.empty:
        return 0.0, pd.DataFrame()

    detail_rows = []

    for _, req in adequacy_reqs.iterrows():
        specialty = req["specialty"]
        county = req["county"]
        provider_count = req["provider_count"]

        # Members in this county
        county_members = members[members["county"] == county]
        total_members = len(county_members)

        if total_members == 0:
            continue

        # Access for this county/specialty
        subset = access_df[
            (access_df["county"] == county) & (access_df["specialty"] == specialty)
        ]

        # Members with sufficient access
        if subset.empty:
            members_with_access = 0
            servicing_providers = 0
        else:
            members_with_access = int(
                (subset["accessible_provider_count"] >= provider_count).sum()
            )
            servicing_providers = int(subset["accessible_provider_count"].sum())

        pct_with_access = members_with_access / total_members if total_members > 0 else 0.0

        # Adequacy index: min(pct_with_access / target, 1.0)
        target = req.get("min_access_pct", 1.0)
        adequacy_index = min(pct_with_access / target, 1.0) if target > 0 else pct_with_access

        detail_rows.append(
            {
                "county": county,
                "specialty": specialty,
                "total_members": total_members,
                "members_with_access": members_with_access,
                "pct_with_access": pct_with_access,
                "servicing_providers": servicing_providers,
                "adequacy_index": adequacy_index,
            }
        )

    detail = pd.DataFrame(detail_rows)

    if detail.empty:
        return 0.0, detail

    # Overall score: mean of adequacy indices
    overall_score = float(detail["adequacy_index"].mean())

    return overall_score, detail


def default_objective(network: pd.DataFrame, data_ctx: dict) -> float:
    """Default objective: network adequacy score.

    Args:
        network: Current network DataFrame.
        data_ctx: Dict with 'access_df', 'members', 'adequacy_reqs'.

    Returns:
        Adequacy score.
    """
    score, _ = compute_adequacy(
        data_ctx["access_df"],
        data_ctx["members"],
        data_ctx["adequacy_reqs"],
    )
    return score
