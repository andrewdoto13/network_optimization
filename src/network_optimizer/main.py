"""CLI entry point for the network optimizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import OptimizerConfig
from .data import load_all, load_weights
from .search import NetworkOptimizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Provider network optimizer — two-phase local search"
    )
    parser.add_argument("--pool", type=Path, required=True, help="Provider pool CSV")
    parser.add_argument("--members", type=Path, required=True, help="Members CSV")
    parser.add_argument("--thresholds", type=Path, required=True, help="Thresholds JSON")
    parser.add_argument("--network", type=Path, default=None, help="Initial network CSV (optional)")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path (default: stdout)")
    parser.add_argument("--max-rounds", type=int, default=100, help="Max rounds per phase")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--enable-swaps", action="store_true", help="Enable Phase 2 swap refinement")
    parser.add_argument("--swap-only", action="store_true", help="Skip Phase 1, only do swaps")
    parser.add_argument("--time-budget", type=float, default=None, help="Time budget in seconds")
    parser.add_argument("--convergence", type=float, default=0.0, help="Convergence threshold (0-1)")
    parser.add_argument("--verbosity", type=int, choices=[0, 1, 2], default=1, help="Verbosity level")
    parser.add_argument("--min-entity-size", type=int, default=None, help="Min providers per entity (filters pool)")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers for candidate scoring (1 = sequential)")
    parser.add_argument("--search-mode", choices=["steepest", "first-improvement"], default="first-improvement",
                        help="Search strategy (default: first-improvement)")
    parser.add_argument("--weights", type=Path, default=None, help="Weights JSON path, e.g. data/weights.json")
    parser.add_argument("--quick", action="store_true", help="Quick test mode (10 rounds, no swaps)")

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.max_rounds = 10
        args.patience = 3
        args.enable_swaps = False

    print("Loading data...")
    weights = load_weights(args.weights)
    pool, members, thresholds, initial_network = load_all(
        args.pool, args.members, args.thresholds, args.network
    )

    # Filter pool to entities with min N providers
    if args.min_entity_size is not None:
        entity_sizes = pool.groupby("entity").size()
        qualifying = set(entity_sizes[entity_sizes >= args.min_entity_size].index)
        before = len(pool)
        pool = pool[pool["entity"].isin(qualifying)].reset_index(drop=True)
        initial_network = initial_network[initial_network["entity"].isin(qualifying)].reset_index(drop=True)
        print(f"  Filtered pool: {before} -> {len(pool)} providers ({pool['entity'].nunique()} entities with >= {args.min_entity_size} providers)")

    config = OptimizerConfig(
        max_rounds=args.max_rounds,
        patience=args.patience,
        enable_swaps=args.enable_swaps,
        swap_only=args.swap_only,
        time_budget=args.time_budget,
        convergence_threshold=args.convergence,
        verbosity=args.verbosity,
        n_jobs=args.n_jobs,
        search_mode=args.search_mode.replace("-", "_"),
        metric_weights=weights,
    )

    optimizer = NetworkOptimizer(pool, members, thresholds, initial_network, config)
    result = optimizer.optimize()

    # Build output
    output = {
        "score": round(result.score, 2),
        "network_entities": sorted(result.network_entities),
        "num_entities": len(result.network_entities),
        "num_providers": len(result.network),
        "entities_added": result.entities_added,
        "entities_swapped": [list(s) for s in result.entities_swapped],
        "phases": optimizer.phases,
        "elapsed": round(result.elapsed, 1),
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {args.output}")
    else:
        print(f"\n{json.dumps(output, indent=2)}")


if __name__ == "__main__":
    main()
