#!/usr/bin/env python3
"""Generate synthetic data matching the network_manager_agent schema.

Produces:
  data/synth/candidates.csv  — provider-level market data
  data/synth/members.csv     — member locations
  data/synth/thresholds.json — nested county/specialty distance thresholds
  data/synth/network.csv     — initial network (subset of candidates)
"""

import json
import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pathlib import Path


# Michigan counties with realistic centers (lat, lon)
COUNTIES = {
    "wayne": (42.15, -83.35),
    "oakland": (42.50, -83.25),
    "kalamazoo": (42.30, -85.00),
}

# Counties and their member population weights
MEMBER_WEIGHTS = {
    "wayne": 0.50,
    "oakland": 0.30,
    "kalamazoo": 0.20,
}

# Specialties with realistic provider counts
SPECIALTIES = [
    "general practice",
    "family medicine",
    "internal medicine",
    "cardiology",
    "orthopedic surgery",
    "obstetrics/gynecology",
    "pediatrics",
    "psychiatry",
    "dermatology",
    "endocrinology",
]

# Entity names (contracting organizations)
ENTITY_NAMES = [
    "Beaumont Health",
    "Corewell Health",
    "Spectrum Health",
    "Henry Ford Health",
    "Providence Health",
    "Ascension Michigan",
    "Michigan Health & Hospital Corp",
    "Trinity Health",
    "Priority Health",
    "Independence Blue Cross",
    "Careington Health",
    "UnitedHealth Community Plan",
    "MultiCare Health",
    "Advance Health",
    "Envision Healthcare",
    "Encompass Health",
    "LifePoint Health",
    "Prime Healthcare",
    "CommonSpirit Health",
    "Bon Secours Mercy Health",
]

# Thresholds matching agent format: {state: {county: {specialty: distance_miles}}}
THRESHOLDS = {
    "mi": {
        "wayne": {
            "general practice": 20.0,
            "family medicine": 20.0,
            "internal medicine": 20.0,
            "cardiology": 15.0,
            "orthopedic surgery": 20.0,
            "obstetrics/gynecology": 15.0,
            "pediatrics": 20.0,
            "psychiatry": 25.0,
            "dermatology": 20.0,
            "endocrinology": 20.0,
        },
        "oakland": {
            "general practice": 20.0,
            "family medicine": 20.0,
            "internal medicine": 20.0,
            "cardiology": 15.0,
            "orthopedic surgery": 20.0,
            "obstetrics/gynecology": 15.0,
            "pediatrics": 20.0,
            "psychiatry": 25.0,
            "dermatology": 20.0,
            "endocrinology": 20.0,
        },
        "kalamazoo": {
            "general practice": 25.0,
            "family medicine": 25.0,
            "internal medicine": 25.0,
            "cardiology": 20.0,
            "orthopedic surgery": 25.0,
            "obstetrics/gynecology": 20.0,
            "pediatrics": 25.0,
            "psychiatry": 30.0,
            "dermatology": 25.0,
            "endocrinology": 25.0,
        },
    }
}


def generate_entities(num_entities: int) -> List[Dict]:
    """Generate entity metadata."""
    entities = []
    for i in range(num_entities):
        # Assign entity to 1-3 counties (geographic reach)
        num_counties = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
        counties = list(np.random.choice(list(COUNTIES.keys()), size=num_counties, replace=False))
        # Assign 1-3 specialties per entity
        num_specs = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        specs = list(np.random.choice(SPECIALTIES, size=num_specs, replace=False))
        entities.append({
            "entity": ENTITY_NAMES[i % len(ENTITY_NAMES)] + (f" {i // len(ENTITY_NAMES) + 1}" if i >= len(ENTITY_NAMES) else ""),
            "counties": counties,
            "specialties": specs,
            "num_providers": np.random.randint(1, 8),
        })
    return entities


def generate_candidates(entities: List[Dict], num_providers: Optional[int] = None) -> pd.DataFrame:
    """Generate provider-level candidate data."""
    records = []
    id_counter = 0

    for entity in entities:
        for _ in range(entity["num_providers"]):
            county = np.random.choice(entity["counties"])
            center_lat, center_lon = COUNTIES[county]
            # Cluster around county center with some spread
            lat = center_lat + np.random.normal(0, 0.15)
            lon = center_lon + np.random.normal(0, 1.5)
            specialty = np.random.choice(entity["specialties"])
            city = county.capitalize()

            records.append({
                "id": id_counter,
                "entity": entity["entity"],
                "specialty": specialty,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "state": "mi",
                "county": county,
                "city": city,
                "effectiveness": np.random.randint(1, 6),
                "efficiency": np.random.randint(1, 6),
                "location_confidence": round(np.random.uniform(0.5, 1.0), 2),
                "total_claims_volume": np.random.randint(10, 500),
                "medicare_claims_volume": np.random.randint(5, 200),
                "total_claims_amount": round(np.random.uniform(10000, 500000), 2),
                "medicare_total_claims_amount": round(np.random.uniform(5000, 200000), 2),
                "new_patient_claims": np.random.randint(0, 50),
            })
            id_counter += 1

    return pd.DataFrame(records)


def generate_members(num_members: int) -> pd.DataFrame:
    """Generate member location data."""
    records = []
    counties = list(MEMBER_WEIGHTS.keys())
    weights = list(MEMBER_WEIGHTS.values())

    chosen_counties = np.random.choice(counties, size=num_members, p=weights)

    for i, county in enumerate(chosen_counties):
        center_lat, center_lon = COUNTIES[county]
        # Members spread across the county
        lat = center_lat + np.random.normal(0, 0.25)
        lon = center_lon + np.random.normal(0, 2.0)
        records.append({
            "id": i,
            "state": "mi",
            "county": county,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
        })

    return pd.DataFrame(records)


def generate_network(candidates: pd.DataFrame, num_entities: int = 5) -> pd.DataFrame:
    """Generate initial network: select a subset of entities."""
    all_entities = candidates["entity"].unique()
    network_entities = np.random.choice(all_entities, size=min(num_entities, len(all_entities)), replace=False)
    return candidates[candidates["entity"].isin(network_entities)].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data matching agent schema")
    parser.add_argument("--num-entities", type=int, default=50, help="Number of contracting entities")
    parser.add_argument("--num-members", type=int, default=1000, help="Number of members")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=Path, default=Path("data/synth"), help="Output directory")
    args = parser.parse_args()

    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_entities} entities, {args.num_members} members...")

    entities = generate_entities(args.num_entities)
    candidates = generate_candidates(entities, None)
    members = generate_members(args.num_members)
    network = generate_network(candidates, num_entities=5)

    # Save
    candidates.to_csv(args.output_dir / "candidates.csv", index=False)
    members.to_csv(args.output_dir / "members.csv", index=False)
    network.to_csv(args.output_dir / "network.csv", index=False)
    with open(args.output_dir / "thresholds.json", "w") as f:
        json.dump(THRESHOLDS, f, indent=2)

    print(f"Generated:")
    print(f"  candidates.csv: {len(candidates)} providers, {candidates['entity'].nunique()} entities")
    print(f"  members.csv: {len(members)} members across {members['county'].nunique()} counties")
    print(f"  network.csv: {len(network)} providers, {network['entity'].nunique()} entities")
    print(f"  thresholds.json: {len(THRESHOLDS['mi'])} counties, {len(THRESHOLDS['mi']['wayne'])} specialties")

    # Print entity stats
    entity_stats = candidates.groupby("entity").agg(
        provider_count=("id", "count"),
        specialties=("specialty", "nunique"),
        counties=("county", "nunique"),
        avg_effectiveness=("effectiveness", "mean"),
        avg_efficiency=("efficiency", "mean"),
    ).round(2)
    print(f"\nEntity stats (top 10 by provider count):")
    print(entity_stats.sort_values("provider_count", ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
