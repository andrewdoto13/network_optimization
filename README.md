# Network Optimizer

Deterministic provider network optimization via two-phase local search.

Given a pool of candidate contracting entities, a member population, and distance-based adequacy thresholds per county/specialty, find the network that maximizes member access to adequate care.

## Approach

Two-phase local search (steepest ascent):

1. **Phase 1 (Construct):** Greedy additions — add entities one at a time, picking the biggest score improvement.
2. **Phase 2 (Refine):** Fixed-network-size swap refinement — trade poorly-placed entities for better ones.

Uses `sklearn.neighbors.BallTree` for efficient haversine distance queries — O(M log P) instead of O(M × P) cross joins. Per-specialty BallTree + per-county query approach avoids N×C tree rebuilds.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.9+. Venv (`.venv/`) is pre-configured in the repo.

## Usage

### Real data (Michigan market)

```bash
run-optimizer \
  --pool data/mi_market_data.csv \
  --members data/mi_members_sample.csv \
  --thresholds data/thresholds.json \
  --min-entity-size 20 \
  --max-rounds 50 \
  --enable-swaps
```

Quick test mode (10 rounds, no swaps):

```bash
run-optimizer \
  --pool data/mi_market_data.csv \
  --members data/mi_members_sample.csv \
  --thresholds data/thresholds.json \
  --quick
```

### CLI Options

| Option | Description |
|---|---|
| `--pool` | Provider pool CSV (required) |
| `--members` | Member locations CSV (required) |
| `--thresholds` | Distance thresholds JSON (required) |
| `--network` | Initial network CSV (optional) |
| `--min-entity-size` | Minimum providers per entity to include in pool |
| `--max-rounds` | Max rounds per phase (default: 100) |
| `--patience` | No-improvement rounds before stopping (default: 5) |
| `--enable-swaps` | Enable Phase 2 swap refinement |
| `--swap-only` | Skip Phase 1, only do swaps |
| `--time-budget` | Time budget in seconds |
| `--convergence` | Convergence threshold 0-1 (default: 0) |
| `--verbosity` | 0=silent, 1=summary, 2=per-round (default: 1) |
| `--output` | Write results to JSON file (default: stdout) |
| `--quick` | Quick test mode (10 rounds, no swaps) |

## Data

### Schema

**Pool / Network** — provider-level data with entity grouping:
- Required: `id`, `entity`, `specialty`, `lat`, `lon`, `state`, `effectiveness`, `efficiency`
- Optional: `county`, `city`, `location_confidence`, claims volume/amount columns

**Members** — member locations:
- Required: `id`, `state`, `county`, `lat`, `lon`

**Thresholds** — nested JSON format:
```json
{
  "mi": {
    "wayne": {
      "general practice": 20.0,
      "cardiology": 15.0
    },
    "oakland": { ... }
  }
}
```

Binary coverage: member has access if **any** provider of matching specialty is within the distance threshold.

### Included datasets

| File | Description |
|---|---|
| `data/mi_market_data.csv` | 124K providers, 13.8K entities, 46 specialties (MI market) |
| `data/MedicareSampleCensus2023Q4.csv` | 2.2M members, 83 counties (national sample) |
| `data/mi_members_sample.csv` | 10K MI members (testing subset) |
| `data/thresholds.json` | 10 MI counties × 10 specialties (100 thresholds) |

Raw pool data is normalized automatically: column renames (`Primary Contract Entity` → `entity`), scaled coordinates (÷1M, Western hemisphere longitude negation), string lowercasing, NaN entity/specialty filtering.

## Performance

### Real data benchmarks (MI market, 10K members, 100 thresholds)

| Pool filter | Entities | Score | Time |
|---|---|---|---|
| 20+ providers | 479 | 95.33% | 116s (2 rounds) |
| 5+ providers | 2,122 | — | ~28 min/round |

Scoring: `adequacy_score` call = 0.79s for full 124K provider pool.

Score = mean coverage % across all (county, specialty) thresholds.

## Custom Objectives

The optimizer supports custom objective functions. Pass any `(network) -> float` callable:

```python
from network_optimizer import NetworkOptimizer, adequacy_score, compute_coverage

# Example: combine adequacy with provider effectiveness
def my_objective(network):
    if len(network) == 0:
        return 0.0
    adequacy = adequacy_score(members, thresholds, network)
    effectiveness = network["effectiveness"].mean()
    return adequacy * 0.7 + effectiveness * 0.3

optimizer = NetworkOptimizer(pool, members, thresholds, initial_network, objective=my_objective)
result = optimizer.optimize()
```

Use `compute_coverage()` to build more complex objectives with per-county/specialty breakdowns.

## Project Structure

```
network_optimization/
├── src/network_optimizer/     # Package
│   ├── config.py              # OptimizerConfig dataclass + validation
│   ├── data.py                # Data loading, validation, raw-format normalization
│   ├── scoring.py             # BallTree coverage queries + adequacy scoring
│   ├── search.py              # Two-phase local search
│   └── main.py                # CLI entry point
├── data/                      # Real datasets
│   ├── mi_market_data.csv     # 124K providers
│   ├── MedicareSampleCensus2023Q4.csv  # 2.2M members
│   ├── mi_members_sample.csv  # 10K MI subset
│   └── thresholds.json        # 100 thresholds
├── scripts/
│   └── generate_data.py       # Synthetic data generator (legacy)
└── tests/                     # [TODO] pytest suite
```

## Development

```bash
ruff check .                   # Lint (use --fix --unsafe-fixes)
mypy src/network_optimizer     # Type check (disallow_untyped_defs = false)
```

## License

MIT
