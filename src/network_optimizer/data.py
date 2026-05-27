"""Data loading, validation, and normalization."""

import pandas as pd


def load_pool(path: str) -> pd.DataFrame:
    """Load provider pool CSV."""
    return pd.read_csv(path)


def load_members(path: str) -> pd.DataFrame:
    """Load members CSV."""
    return pd.read_csv(path)


def load_network(path: str) -> pd.DataFrame:
    """Load initial network CSV."""
    return pd.read_csv(path)


def load_adequacy_reqs(path: str) -> pd.DataFrame:
    """Load adequacy requirements CSV."""
    return pd.read_csv(path)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize string columns to lowercase."""
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.lower().str.strip()
    return df


def validate_pool(df: pd.DataFrame) -> None:
    """Validate provider pool DataFrame."""
    required = {"npi", "specialty", "group_id", "county", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Pool missing columns: {missing}")
    if df.empty:
        raise ValueError("Pool is empty")


def validate_members(df: pd.DataFrame) -> None:
    """Validate members DataFrame."""
    required = {"member_id", "county", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Members missing columns: {missing}")
    if df.empty:
        raise ValueError("Members is empty")


def validate_adequacy_reqs(df: pd.DataFrame) -> None:
    """Validate adequacy requirements DataFrame."""
    required = {"specialty", "county", "provider_count", "distance_req"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Adequacy requirements missing columns: {missing}")
    if df.empty:
        raise ValueError("Adequacy requirements is empty")
