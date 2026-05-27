"""Data loading and validation for the network optimizer."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import MEMBER_REQUIRED, POOL_REQUIRED


def load_pool(path: str | Path) -> pd.DataFrame:
    """Load provider pool data.

    Expects columns matching the agent schema:
      id, entity, specialty, lat, lon, state, county, effectiveness, efficiency, ...

    Returns a validated DataFrame with required columns.
    """
    df = pd.read_csv(path)

    # Normalize column names (lowercase, strip whitespace)
    df.columns = [c.strip().lower() for c in df.columns]

    # Validate required columns
    missing = POOL_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(
            f"Pool data missing required columns: {missing}. "
            f"Found: {sorted(df.columns)}"
        )

    # Convert coordinates to float
    for col in ("lat", "lon"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with invalid coordinates
    invalid_mask = df["lat"].isna() | df["lon"].isna()
    if invalid_mask.any():
        print(f"  Dropping {invalid_mask.sum()} rows with invalid coordinates")
        df = df[~invalid_mask].reset_index(drop=True)

    # Normalize entity to string
    df["entity"] = df["entity"].astype(str).str.strip()

    # Normalize specialty
    df["specialty"] = df["specialty"].astype(str).str.strip()

    # Normalize state/county
    for col in ("state", "county"):
        df[col] = df[col].astype(str).str.strip().str.lower()

    return df


def load_members(path: str | Path) -> pd.DataFrame:
    """Load member location data.

    Expects columns: id, state, county, lat, lon

    Returns a validated DataFrame with required columns.
    """
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Validate required columns
    missing = MEMBER_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(
            f"Members data missing required columns: {missing}. "
            f"Found: {sorted(df.columns)}"
        )

    # Convert coordinates to float
    for col in ("lat", "lon"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with invalid coordinates
    invalid_mask = df["lat"].isna() | df["lon"].isna()
    if invalid_mask.any():
        print(f"  Dropping {invalid_mask.sum()} rows with invalid coordinates")
        df = df[~invalid_mask].reset_index(drop=True)

    # Normalize state/county
    for col in ("state", "county"):
        df[col] = df[col].astype(str).str.strip().str.lower()

    return df


def load_thresholds(path: str | Path) -> dict:
    """Load adequacy thresholds from JSON file.

    Expects nested structure matching agent format:
    {
        "mi": {
            "wayne": {
                "general practice": 20.0,
                "cardiology": 15.0,
                ...
            },
            ...
        }
    }

    Returns the parsed dict.
    """
    with open(path) as f:
        return json.load(f)


def load_initial_network(path: str | Path, pool: pd.DataFrame) -> pd.DataFrame:
    """Load initial network from CSV or select from pool.

    If path exists, loads entity names from CSV and filters pool.
    If path does not exist, returns empty network.
    """
    path = Path(path)
    if not path.exists():
        return pool.iloc[:0].copy()

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "entity" in df.columns:
        entities = set(df["entity"].str.strip().str.lower().unique())
        return pool[pool["entity"].str.lower().isin(entities)].reset_index(drop=True)
    else:
        raise ValueError(f"Initial network CSV missing 'entity' column. Found: {sorted(df.columns)}")


def load_all(
    pool_path: str | Path,
    members_path: str | Path,
    thresholds_path: str | Path,
    network_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """Load all data files and return them as a tuple.

    Returns:
        (pool, members, thresholds, initial_network)
    """
    pool = load_pool(pool_path)
    members = load_members(members_path)
    thresholds = load_thresholds(thresholds_path)

    initial_network = load_initial_network(network_path, pool) if network_path is not None else pool.iloc[:0].copy()

    print(f"  Pool: {len(pool)} providers, {pool['entity'].nunique()} entities, {pool['specialty'].nunique()} specialties")
    print(f"  Members: {len(members)} across {members['county'].nunique()} counties")
    print(f"  Initial network: {len(initial_network)} providers, {initial_network['entity'].nunique()} entities")

    # Count threshold requirements
    total_reqs = sum(
        len(specialties)
        for counties in thresholds.values()
        for specialties in counties.values()
    )
    print(f"  Thresholds: {total_reqs} requirements across {len(thresholds)} states")

    return pool, members, thresholds, initial_network
