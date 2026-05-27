"""CLI entry point for the network optimizer."""

import argparse
import json

from network_optimizer.config import OptimizerConfig
from network_optimizer.data import (
    load_adequacy_reqs,
    load_members,
    load_network,
    load_pool,
    normalize_df,
    validate_adequacy_reqs,
    validate_members,
    validate_pool,
)
from network_optimizer.search import NetworkOptimizer


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Deterministic provider network optimization via local search"
    )
    parser.add_argument("--pool", required=True, help="Provider pool CSV path")
    parser.add_argument("--members", required=True, help="Members CSV path")
    parser.add_argument("--adequacy-reqs", required=True, help="Adequacy requirements CSV path")
    parser.add_argument("--network", default=None, help="Initial network CSV path (optional)")
    parser.add_argument("--max-rounds", type=int, default=100, help="Max rounds per phase")
    parser.add_argument("--patience", type=int, default=5, help="No-improvement rounds to stop")
    parser.add_argument("--enable-swaps", action="store_true", help="Enable swap refinement phase")
    parser.add_argument("--time-budget", type=float, default=None, help="Max runtime in seconds")
    parser.add_argument("--output", default=None, help="Write results to JSON file")
    return parser.parse_args(args)


def main(args=None):
    parsed = parse_args(args)

    # Load data
    pool = normalize_df(load_pool(parsed.pool))
    members = normalize_df(load_members(parsed.members))
    adequacy_reqs = normalize_df(load_adequacy_reqs(parsed.adequacy_reqs))

    # Validate
    validate_pool(pool)
    validate_members(members)
    validate_adequacy_reqs(adequacy_reqs)

    # Optional initial network
    network = None
    if parsed.network:
        network = normalize_df(load_network(parsed.network))

    # Config
    config = OptimizerConfig(
        max_rounds=parsed.max_rounds,
        patience=parsed.patience,
        time_budget=parsed.time_budget,
        enable_swaps=parsed.enable_swaps,
    )

    # Run
    optimizer = NetworkOptimizer(pool, members, adequacy_reqs, network, config)
    result = optimizer.optimize()

    # Output
    if parsed.output:
        output = {
            "initial_score": result.initial_score,
            "final_score": result.final_score,
            "num_groups": len(result.best_network),
            "phase1_rounds": result.phase1_rounds,
            "phase2_rounds": result.phase2_rounds,
            "total_time": result.total_time,
            "performance_history": result.performance_history,
            "moves": [
                {"type": m.move_type, "add_group_id": m.add_group_id, "remove_group_id": m.remove_group_id}
                for m in result.moves
            ],
        }
        with open(parsed.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {parsed.output}")

    return result


if __name__ == "__main__":
    main()
