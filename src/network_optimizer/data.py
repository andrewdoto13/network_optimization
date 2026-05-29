"""Data loading and validation for the network optimizer."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import MEMBER_REQUIRED, POOL_REQUIRED

# Column rename map: raw CSV column name -> canonical name
POOL_COLUMN_RENAMES = {
    "latitude": "lat",
    "longitude": "lon",
    "primary contract entity": "entity",
    "location confidence score": "location_confidence",
    "total claims amount": "total_claims_amount",
    "medicare total claims amount": "medicare_total_claims_amount",
    "medicare new patient claims": "new_patient_claims",
    "medicare claims volume": "medicare_claims_volume",
    "total claims volume": "total_claims_volume",
}

MEMBER_COLUMN_RENAMES = {
    "countyname": "county",
    "latitude": "lat",
    "longitude": "lon",
    "rowid": "id",
}


def _normalize_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize scaled integer coordinates in-place.

    If coordinates are scaled integers (abs(mean) > 1000), divide by 1_000_000.
    Negate longitudes if all positive (Western hemisphere convention).
    """
    for col in ("lat", "lon"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["lat"].abs().mean() > 1000:
        df["lat"] = df["lat"] / 1_000_000.0
        df["lon"] = df["lon"] / 1_000_000.0

    if df["lon"].gt(0).any() and df["lon"].lt(0).sum() == 0:
        df["lon"] = -df["lon"]

    return df


def _normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and lowercase all string columns. Replace 'nan' with NA."""
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].replace("nan", pd.NA)
    return df


def load_pool(path: str | Path) -> pd.DataFrame:
    """Load provider pool data.

    Handles raw CSV format with column normalization:
      - Renames raw columns to canonical names
      - Normalizes scaled integer coordinates
      - Lowercases string values

    Returns a validated DataFrame with required columns.
    """
    df = pd.read_csv(path)

    # Normalize column names (lowercase, strip whitespace)
    df.columns = [c.strip().lower() for c in df.columns]

    # Apply column renames
    df.rename(columns=POOL_COLUMN_RENAMES, inplace=True)

    # Generate id from row index if missing
    if "id" not in df.columns:
        df.insert(0, "id", range(len(df)))

    # Normalize coordinates (handles both raw scaled int and standard decimal)
    df = _normalize_coordinates(df)

    # Drop rows with invalid coordinates
    invalid_mask = df["lat"].isna() | df["lon"].isna()
    if invalid_mask.any():
        print(f"  Dropping {invalid_mask.sum()} rows with invalid coordinates")
        df = df[~invalid_mask].reset_index(drop=True)

    # Normalize strings
    df = _normalize_strings(df)

    # Validate required columns
    missing = POOL_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(
            f"Pool data missing required columns: {missing}. "
            f"Found: {sorted(df.columns)}"
        )

    # Drop rows with missing entity
    na_entities = df["entity"].isna().sum()
    if na_entities:
        print(f"  Dropping {na_entities} rows with missing entity")
        df = df[df["entity"].notna()].reset_index(drop=True)

    # Normalize entity to string
    df["entity"] = df["entity"].astype(str).str.strip()

    # Drop rows with missing specialty
    na_specialties = df["specialty"].isna().sum()
    if na_specialties:
        print(f"  Dropping {na_specialties} rows with missing specialty")
        df = df[df["specialty"].notna()].reset_index(drop=True)

    # Normalize specialty
    df["specialty"] = df["specialty"].astype(str).str.strip()

    # Normalize state (county is optional for pool)
    if "state" in df.columns:
        df["state"] = df["state"].astype(str).str.strip().str.lower()
    if "county" in df.columns:
        df["county"] = df["county"].astype(str).str.strip().str.lower()

    return df


def load_members(path: str | Path) -> pd.DataFrame:
    """Load member location data.

    Handles raw CSV format with column normalization:
      - Renames raw columns to canonical names
      - Normalizes scaled integer coordinates
      - Lowercases string values

    Returns a validated DataFrame with required columns.
    """
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Apply column renames
    df.rename(columns=MEMBER_COLUMN_RENAMES, inplace=True)

    # Normalize coordinates (handles both raw scaled int and standard decimal)
    df = _normalize_coordinates(df)

    # Drop rows with invalid coordinates
    invalid_mask = df["lat"].isna() | df["lon"].isna()
    if invalid_mask.any():
        print(f"  Dropping {invalid_mask.sum()} rows with invalid coordinates")
        df = df[~invalid_mask].reset_index(drop=True)

    # Normalize strings
    df = _normalize_strings(df)

    # Validate required columns
    missing = MEMBER_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(
            f"Members data missing required columns: {missing}. "
            f"Found: {sorted(df.columns)}"
        )

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
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Thresholds must be a JSON object, got {type(data).__name__}")

    for state, counties in data.items():
        if not isinstance(state, str) or not state.strip():
            raise ValueError(f"Invalid state key: {state!r}")
        if not isinstance(counties, dict):
            raise ValueError(f"State {state!r} value must be a dict, got {type(counties).__name__}")

        for county, specialties in counties.items():
            if not isinstance(county, str) or not county.strip():
                raise ValueError(f"Invalid county key in state {state!r}: {county!r}")
            if not isinstance(specialties, dict):
                raise ValueError(
                    f"County {county!r} in state {state!r} must be a dict, got {type(specialties).__name__}"
                )

            for specialty, distance in specialties.items():
                if not isinstance(specialty, str) or not specialty.strip():
                    raise ValueError(
                        f"Invalid specialty key in {state!r}/{county!r}: {specialty!r}"
                    )
                if not isinstance(distance, (int, float)) or distance <= 0:
                    raise ValueError(
                        f"Distance for {specialty!r} in {state!r}/{county!r} must be a positive number, got {distance!r}"
                    )

    return data


def load_weights(path: str | Path | None) -> dict[str, float]:
    """Load metric weights from JSON file.

    Expects: {"efficiency": 0.3, "effectiveness": 0.2}
    Returns empty dict if path is None or file doesn't exist.
    """
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Weights must be a JSON object, got {type(data).__name__}")
    for key, val in data.items():
        if not isinstance(key, str) or not isinstance(val, (int, float)):
            raise ValueError(f"Invalid weight: {key}: {val}")
    return {k: float(v) for k, v in data.items()}


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

    # Cross-validate thresholds against pool/members data
    pool_states = set(pool["state"].unique())
    pool_specialties = set(pool["specialty"].unique())
    member_states = set(members["state"].unique())
    member_counties = set(members["county"].unique())

    known_states = pool_states | member_states
    known_counties = member_counties

    for state, counties in thresholds.items():
        state_lower = state.lower()
        if state_lower not in known_states:
            raise ValueError(
                f"Threshold state {state!r} not found in pool or members data. "
                f"Known states: {sorted(known_states)}"
            )

        for county in counties:
            county_lower = county.lower()
            if county_lower not in known_counties:
                raise ValueError(
                    f"Threshold county {county!r} in state {state!r} not found in members data. "
                    f"Known counties: {sorted(known_counties)}"
                )

            for specialty in counties[county]:
                spec_lower = specialty.lower()
                if spec_lower not in {s.lower() for s in pool_specialties}:
                    raise ValueError(
                        f"Threshold specialty {specialty!r} in {state!r}/{county!r} not found in pool data. "
                        f"Known specialties: {sorted(pool_specialties)}"
                    )

    pool_counties = pool["county"].nunique() if "county" in pool.columns else 0
    print(f"  Pool: {len(pool)} providers, {pool['entity'].nunique()} entities, {pool['specialty'].nunique()} specialties, {pool_counties} counties")
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
