# Network Optimizer

Deterministic provider network optimization via two-phase local search.

Given a pool of candidate contracting entities, a member population, and distance-based adequacy thresholds per county/specialty, find the network that maximizes member access to adequate care.

## Approach

Two-phase local search with configurable strategy:

1. **Phase 1 (Construct):** Greedy additions — add entities one at a time.
2. **Phase 2 (Refine):** Fixed-network-size swap refinement — trade poorly-placed entities for better ones.

### Search Modes

- **First-improvement** (default): Accept the first candidate that improves the score. Candidates are pre-ranked by estimated contribution, so the best is evaluated first. Fast, typically evaluates far fewer candidates per round.
- **Steepest ascent** (`--search-mode steepest`): Evaluate all candidates, pick the best. Guaranteed locally optimal per round but slower.

### Candidate Prefiltering

A BallTree-based prefilter eliminates entities that cannot improve coverage before scoring. It checks two conditions:
1. **Specialty gap:** Entity has at least one specialty in a sub-100% coverage (county, specialty) pair.
2. **Geographic reach:** Entity has at least one provider within threshold distance of an uncovered member.

Typically reduces candidates by 80-90% (e.g. 13,842 → 1,658). Filter is lossless — never eliminates a candidate that could improve the score.

### Weighted Objectives

Combine adequacy with provider metrics via a weights JSON file:

```json
{"efficiency": 0.3, "effectiveness": 0.2}
```

Adequacy weight is implicit: `1.0 - sum(weights)`. Column metrics are normalized to 0-100 using pool min/max, then combined. Pass via `--weights path/to/weights.json`.

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

With weighted objectives (70% adequacy, 30% efficiency):

```bash
echo '{"efficiency": 0.3}' > weights.json
run-optimizer \
  --pool data/mi_market_data.csv \
  --members data/mi_members_sample.csv \
  --thresholds data/thresholds.json \
  --weights weights.json
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
| `--weights` | Metric weights JSON, e.g. `{"efficiency": 0.3}` (optional) |
| `--search-mode` | `first-improvement` (default) or `steepest` |
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

**Weights** — metric weights JSON (optional):
```json
{"efficiency": 0.3, "effectiveness": 0.2}
```
Values must be in [0, 1], sum must be < 1.0. Adequacy gets the remaining weight (`1.0 - sum`). Column metrics are normalized to 0-100 using pool min/max, then combined as a weighted sum.

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

## Weighted Objectives

Combine adequacy with provider metrics via a weights JSON file. Create a weights file:

```json
{"efficiency": 0.3, "effectiveness": 0.2}
```

Then pass it on the CLI:
```bash
run-optimizer --pool pool.csv --members members.csv --thresholds thresholds.json --weights weights.json
```

The objective becomes: `0.5 * adequacy + 0.3 * efficiency + 0.2 * effectiveness` (adequacy gets the remaining weight).

Column metrics are normalized to 0-100 using pool min/max. Available columns depend on your pool data — typically `effectiveness` and `efficiency` (1-5 scale).

For programmatic use:

```python
from network_optimizer import NetworkOptimizer, OptimizerConfig, weighted_objective

config = OptimizerConfig(metric_weights={"efficiency": 0.3})
optimizer = NetworkOptimizer(pool, members, thresholds, initial_network, config)
result = optimizer.optimize()
```

## Project Structure

```
network_optimization/
├── src/network_optimizer/     # Package
│   ├── config.py              # OptimizerConfig dataclass + validation
│   ├── data.py                # Data loading, validation, raw-format normalization
│   ├── scoring.py             # BallTree coverage queries + adequacy scoring
│   ├── ranking.py             # Candidate prefiltering and ranking
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
