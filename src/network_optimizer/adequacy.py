"""Adequacy scoring for the network optimizer.

Imports coverage computation from distance module and provides
scoring functions for the optimizer.
"""

from __future__ import annotations

import pandas as pd

from .distance import compute_coverage


def score_network(
    pool: pd.DataFrame,
    members: pd.DataFrame,
    thresholds: dict,
    network: pd.DataFrame,
) -> float:
    """Compute overall adequacy score for a given network.

    Score = mean coverage percentage across all (county, specialty) thresholds.
    """
    coverage_results = compute_coverage(pool, members, thresholds, network)
    return compute_score(coverage_results)


def score_network_detailed(
    pool: pd.DataFrame,
    members: pd.DataFrame,
    thresholds: dict,
    network: pd.DataFrame,
) -> tuple[float, list[dict]]:
    """Compute adequacy score with detailed coverage breakdown.

    Returns:
        (score, coverage_results)
    """
    coverage_results = compute_coverage(pool, members, thresholds, network)
    return compute_score(coverage_results), coverage_results


def compute_score(coverage_results: list[dict]) -> float:
    """Compute overall adequacy score from coverage results.

    Score = mean coverage percentage across all (county, specialty) thresholds.
    """
    if not coverage_results:
        return 0.0
    return sum(r["coverage_percentage"] for r in coverage_results) / len(coverage_results)
