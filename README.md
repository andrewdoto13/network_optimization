# Network Optimizer

Deterministic provider network optimization via two-phase local search.

Given a pool of candidate contracting entities, a member population, and distance-based adequacy thresholds per county/specialty, find the network that maximizes member access to adequate care.

## Approach

Two-phase local search (steepest ascent):

1. **Phase 1 (Construct):** Greedy additions — add entities one at a time, picking the biggest score improvement.
2. **Phase 2 (Refine):** Fixed-network-size swap refinement — trade poorly-placed entities for better ones.

Uses `sklearn.neighbors.BallTree` for efficient haversine distance queries — O(M log P) instead of O(M × P) cross joins.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.9+. Venv (`.venv/`) is pre-configured in the repo.

## Usage

```bash
run-optimizer \
  --pool data/synth/candidates.csv \
  --members data/synth/members.csv \
  --thresholds data/synth/thresholds.json \
  --network data/synth/network.csv \
  --max-rounds 100 \
  --enable-swaps
```

Or start with an empty network:

```bash
run-optimizer \
  --pool data/synth/candidates.csv \
  --members data/synth/members.csv \
  --thresholds data/synth/thresholds.json
```

Quick test mode (10 rounds, no swaps):

```bash
run-optimizer \
  --pool data/synth/candidates.csv \
  --members data/synth/members.csv \
  --thresholds data/synth/thresholds.json \
  --quick
```

### CLI Options

| Option | Description |
|---|---|
| `--pool` | Provider pool CSV (required) |
| `--members` | Member locations CSV (required) |
| `--thresholds` | Distance thresholds JSON (required) |
| `--network` | Initial network CSV (optional) |
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

### Schema (agent-compatible)

**Pool / Network** — provider-level data with entity grouping:
- `id`, `entity`, `specialty`, `lat`, `lon`, `state`, `county`, `city`
- `effectiveness`, `efficiency`, `location_confidence`
- Claims volume/amount columns

**Members** — member locations:
- `id`, `state`, `county`, `lat`, `lon`

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

### Generate synthetic data

```bash
python scripts/generate_data.py \
  --num-entities 50 \
  --num-members 1000 \
  --seed 42 \
  --output-dir data/synth/
```

## Performance

Synthetic data: 233 providers, 50 entities, 1000 members, 3 counties, 30 thresholds.

| Starting network | Final score | Entities | Time |
|---|---|---|---|
| Empty | 31.52% | 10 | 16s |
| 5 entities | 38.62% | 15 | 17s |

Score = mean coverage % across all (county, specialty) thresholds.

## Project Structure

```
network_optimization/
├── src/network_optimizer/     # Package
│   ├── config.py              # OptimizerConfig dataclass + validation
│   ├── data.py                # Data loading, validation
│   ├── distance.py            # BallTree-based coverage queries
│   ├── adequacy.py            # Adequacy scoring
│   ├── search.py              # Two-phase local search
│   └── main.py                # CLI entry point
├── tests/                     # [TODO] pytest suite
├── notebooks/                 # [TODO] demo.ipynb
├── scripts/
│   └── generate_data.py       # Synthetic data generator
└── reports/                   # [TODO] benchmark results
```

## Development

```bash
pytest              # Run tests
ruff check .        # Lint
mypy src/network_optimizer  # Type check
```

## License

MIT
