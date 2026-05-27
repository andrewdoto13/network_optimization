# Network Optimizer

Deterministic provider network optimization via local search.

Given a pool of candidate provider groups, a member population, and adequacy requirements per county/specialty, find the network that maximizes member access to adequate care.

## Approach

Two-phase local search (steepest ascent):

1. **Phase 1 (Construct):** Greedy additions from the candidate pool until convergence.
2. **Phase 2 (Refine):** Fixed-network-size swap refinement — trade poorly-placed groups for better ones.

Uses `sklearn.neighbors.BallTree` for efficient haversine distance queries — O(M log P) instead of O(M × P) cross joins.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
run-optimizer \
  --pool data/synth/pool.csv \
  --members data/synth/members.csv \
  --adequacy-reqs data/synth/adequacy_reqs.csv \
  --network data/synth/network.csv \
  --max-rounds 100 \
  --enable-swaps
```

Or start with an empty network:

```bash
run-optimizer \
  --pool data/synth/pool.csv \
  --members data/synth/members.csv \
  --adequacy-reqs data/synth/adequacy_reqs.csv
```

## Data

Synthetic test data in `data/synth/`. Real-world member data available from the [Medicare Sample Census](https://data.medicare.gov/).

## Project Structure

```
network_optimization/
├── src/network_optimizer/     # Package
│   ├── config.py              # OptimizerConfig dataclass
│   ├── data.py                # Data loading, validation
│   ├── distance.py            # BallTree-based coverage
│   ├── adequacy.py            # Adequacy scoring
│   ├── search.py              # Two-phase local search
│   └── main.py                # CLI entry point
├── tests/                     # pytest test suite
├── notebooks/                 # Interactive demos
├── scripts/                   # Data generation utilities
└── reports/                   # Benchmark results
```

## Development

```bash
pytest              # Run tests
ruff check .        # Lint
mypy src/network_optimizer  # Type check
```
